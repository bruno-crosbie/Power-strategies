"""Live paper trading of the month/quarter-ahead weather fade against REAL market quotes.

How it works
  1. Every run refreshes the data and works out our fair price for next month's and next quarter's
     UK baseload power and NBP gas (weather-neutral price level x normal weather for the period).
  2. You type the market price you see for each contract into live/quotes.json (on GitHub's website).
     Saving that file starts a run straight away.
  3. The first quote entered for a contract (before its delivery starts) is the trade price. If our fair
     price is at least 2% away, we paper-trade toward fair: buy if fair is higher, sell if lower.
  4. After the delivery period ends, the trade settles against the actual average spot price
     (power: Elexon daily average; gas: National Gas System Average Price).
  5. docs/month-ahead.html shows fair prices, quotes, trades and results.

Size: power 1 MW baseload; gas 25,000 therms/day. Costs: £1/MWh, 0.5p/therm. Paper trading only.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from config import COST_GAS_P_THERM, COST_POWER_GBP_MWH, GAS_POSITION_THERMS_PER_DAY, POWER_POSITION_MW
from lib.sources import download_all
from strategies.nbp_week_ahead import daily_inputs as gas_inputs
from strategies.weather_month_ahead import MIN_EDGE_PCT, gas_fair, power_fair, power_inputs

HERE = Path(__file__).resolve().parent
LIVE = HERE / "live"
LIVE.mkdir(exist_ok=True)
QUOTES = LIVE / "quotes.json"
LEDGER = LIVE / "month_ahead_trades.csv"
FAIR_LOG = LIVE / "fair_values.csv"
PAGE = HERE.parent / "docs" / "month-ahead.html"
RUN_LOG = LIVE / "last_run.log"                    # committed by the workflow, so failures are readable
EDIT_URL = "https://github.com/bruno-crosbie/Power-strategies/edit/main/weather-fade/live/quotes.json"
HOW_TO = ("Type the market price you see for each contract into 'price' (power in GBP/MWh, gas in pence/therm), "
          "then Commit changes. A run starts automatically, compares it with our fair price and logs a paper trade "
          "if they differ by 2% or more. Only the first price entered for a contract counts, and it must be entered "
          "before delivery starts. Keep the quotes and commas exactly as they are; leave null to skip.")
LEDGER_COLS = ["contract", "product", "start", "end", "decided_at", "quote", "fair", "spot14", "edge_pct", "position",
               "status", "outcome", "pnl_gbp", "note"]


# ------------------------------------------------------------------ contracts

def upcoming(today: pd.Timestamp) -> list[dict]:
    """Next calendar month and next calendar quarter, for power and gas."""
    m0 = (today + pd.offsets.MonthBegin(1)).normalize()
    q0 = (today + pd.offsets.QuarterBegin(1, startingMonth=1)).normalize()
    periods = [("month", m0, m0 + pd.offsets.MonthEnd(0), m0.strftime("%Y-%m")),
               ("quarter", q0, q0 + pd.offsets.QuarterEnd(0), f"{q0.year}-Q{q0.quarter}")]
    out = []
    for kind, start, end, label in periods:
        for market in ("power", "gas"):
            out.append({"contract": f"{market}_{kind} {label}", "product": f"{market}_{kind}", "market": market,
                        "start": start, "end": end,
                        "unit": "GBP/MWh" if market == "power" else "p/therm",
                        "what": (f"UK baseload power, {label}" if market == "power" else f"NBP gas, {label}")
                                + (" (e.g. ICE UK Base Load futures)" if market == "power" else " (e.g. ICE UK NBP futures)")})
    return out


def read_quotes() -> tuple[dict, str]:
    if not QUOTES.exists():
        return {}, ""
    try:
        return json.loads(QUOTES.read_text(encoding="utf-8")).get("contracts", {}), ""
    except (json.JSONDecodeError, AttributeError) as e:
        return None, f"quotes.json could not be read ({e}). Check for a missing comma or quote mark."


def write_quotes(quotes: dict, contracts: list[dict]):
    """Add blank entries for new contracts; keep everything the user typed."""
    for c in contracts:
        quotes.setdefault(c["contract"], {"price": None, "unit": c["unit"], "what": c["what"],
                                          "trade_by": str((c["start"] - pd.Timedelta(days=1)).date())})
    QUOTES.write_text(json.dumps({"how_to_use": HOW_TO, "contracts": quotes}, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")


# ------------------------------------------------------------------ trading

def size_and_cost(market: str, days: int):
    if market == "power":
        units = POWER_POSITION_MW * 24 * days          # MWh
        return units, COST_POWER_GBP_MWH, 1.0           # P&L in £ = Δ(£/MWh) x MWh
    therms = GAS_POSITION_THERMS_PER_DAY * days
    return therms, COST_GAS_P_THERM, 0.01               # pence -> £


def main():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    errors = []
    for src in ("elexon", "gas", "temps"):             # one source failing shouldn't stop the others
        try:
            download_all({src})
        except Exception as e:
            errors.append(f"{src} download failed: {e!r}")
    pw = gs = None
    try:
        pw = power_inputs()
    except Exception as e:
        errors.append(f"power data unavailable: {e!r}")
    try:
        gs = gas_inputs()
    except Exception as e:
        errors.append(f"gas data unavailable: {e!r}")
    contracts = upcoming(today)

    fair = {}
    for c in contracts:
        if (pw if c["market"] == "power" else gs) is None:
            continue
        days = pd.date_range(c["start"], c["end"], freq="D")
        try:
            f = power_fair(pw, today, days) if c["market"] == "power" else gas_fair(gs, today, days)
        except Exception as e:
            errors.append(f"fair price failed for {c['contract']}: {e}")
            continue
        series = pw.price if c["market"] == "power" else gs.sap
        spot = series[(series.index >= today - pd.Timedelta(days=14)) & (series.index < today)].mean()
        fair[c["contract"]] = {"fair": round(f, 2), "spot14": round(float(spot), 2)}

    log = pd.read_csv(FAIR_LOG) if FAIR_LOG.exists() else pd.DataFrame(columns=["date", "contract", "fair", "spot14"])
    new = pd.DataFrame([{"date": str(today.date()), "contract": k, **v} for k, v in fair.items()])
    log = pd.concat([log[log.date != str(today.date())], new]).sort_values(["date", "contract"])
    log.to_csv(FAIR_LOG, index=False)

    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=LEDGER_COLS)
    quotes, qerr = read_quotes()
    if qerr:
        errors.append(qerr)
    else:
        by_name = {c["contract"]: c for c in contracts}
        for name, q in quotes.items():
            price = q.get("price") if isinstance(q, dict) else None
            if price in (None, "") or name in set(ledger.contract):
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                errors.append(f"{name}: price '{price}' is not a number"); continue
            c = by_name.get(name)
            if c is None or name not in fair:   # delivery already started (or fair price unavailable)
                errors.append(f"{name}: quote seen too late (delivery started or contract no longer upcoming) - not traded"); continue
            f = fair[name]["fair"]
            edge = f / price - 1
            pos = int(np.sign(edge)) if abs(edge) >= MIN_EDGE_PCT else 0
            ledger.loc[len(ledger)] = {
                "contract": name, "product": c["product"], "start": str(c["start"].date()), "end": str(c["end"].date()),
                "decided_at": str(pd.Timestamp.now(tz="UTC").floor("min")), "quote": price, "fair": f,
                "spot14": fair[name]["spot14"], "edge_pct": round(edge * 100, 2), "position": pos,
                "status": "open" if pos else "no trade", "outcome": np.nan, "pnl_gbp": np.nan,
                "note": ("buy: fair above market" if pos > 0 else "sell: fair below market" if pos < 0
                         else f"gap under {MIN_EDGE_PCT:.0%} - no trade")}
        write_quotes(quotes, contracts)

    # settle trades whose delivery period has finished and has (almost) all its prices
    for i, r in ledger[ledger.status == "open"].iterrows():
        start, end = pd.Timestamp(r.start), pd.Timestamp(r.end)
        market = r["product"].split("_")[0]
        if today <= end or (pw if market == "power" else gs) is None:
            continue
        series = pw.price if market == "power" else gs.sap
        vals = series.reindex(pd.date_range(start, end, freq="D"))
        if vals.notna().mean() < 0.9:
            continue
        outcome = float(vals.mean())
        units, cost, to_gbp = size_and_cost(market, len(vals))
        pnl = (r.position * (outcome - r.quote) - cost) * units * to_gbp
        ledger.loc[i, ["outcome", "pnl_gbp", "status"]] = [round(outcome, 2), round(pnl, 0), "settled"]
    ledger.to_csv(LEDGER, index=False)
    build_page(today, contracts, fair, quotes or {}, ledger, log, errors)
    RUN_LOG.write_text(f"{pd.Timestamp.now(tz='UTC')}: ok, {len(fair)} fair prices\n" + "".join(f"- {e}\n" for e in errors),
                       encoding="utf-8")


# ------------------------------------------------------------------ page

def build_page(today, contracts, fair, quotes, ledger, log, errors):
    def money(v):
        return "–" if pd.isna(v) else f"{'-' if v < 0 else ''}£{abs(v):,.0f}"
    rows = ""
    for c in contracts:
        k = c["contract"]
        f = fair.get(k, {})
        q = (quotes.get(k) or {}).get("price")
        traded = ledger[ledger.contract == k]
        status = traded.iloc[0].note if len(traded) else ("waiting for your quote" if q in (None, "") else "will trade next run")
        rows += (f"<tr><td><b>{c['what'].split(' (')[0]}</b><div class='muted'>trade by {(c['start'] - pd.Timedelta(days=1)).date()}</div></td>"
                 f"<td>{f.get('fair', '–')}</td><td>{f.get('spot14', '–')}</td><td>{q if q not in (None, '') else '–'}</td>"
                 f"<td>{c['unit']}</td><td>{status}</td></tr>")
    trades = ""
    for _, r in ledger.sort_values("decided_at", ascending=False).iterrows():
        side = {1: "Buy", -1: "Sell", 0: "–"}.get(int(r.position), "–")
        trades += (f"<tr><td>{r.contract}</td><td>{str(r.decided_at)[:16]}</td><td>{r.quote}</td><td>{r.fair}</td>"
                   f"<td>{r.edge_pct:+.1f}%</td><td>{side}</td><td>{r.status}</td>"
                   f"<td>{'' if pd.isna(r.outcome) else r.outcome}</td>"
                   f"<td class='{'pos' if (r.pnl_gbp or 0) > 0 else 'neg' if (r.pnl_gbp or 0) < 0 else ''}'>{money(r.pnl_gbp)}</td></tr>")
    total = ledger.pnl_gbp.sum() if len(ledger) else 0
    settled = int((ledger.status == "settled").sum()) if len(ledger) else 0
    hist = {k: g.sort_values("date")[["date", "fair", "spot14"]].values.tolist() for k, g in log.groupby("contract")}
    err = "".join(f"<li>{e}</li>" for e in errors)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Month-Ahead Fade</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>:root{{--bg:#f6f7f9;--card:#fff;--text:#16191d;--muted:#5f6670;--border:#e3e6ea;--accent:#2563eb;--pos:#15803d;--neg:#b91c1c;--warn:#fef3c7;--warnt:#92400e}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1216;--card:#171b21;--text:#e8eaed;--muted:#9aa3ad;--border:#2a313a;--accent:#60a5fa;--pos:#4ade80;--neg:#f87171;--warn:#422006;--warnt:#fde68a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:22px 16px 50px}}h1{{font-size:24px;margin:0}}h2{{font-size:17px;margin:0 0 8px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:14px 0}}
.muted{{color:var(--muted);font-size:13px}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border);vertical-align:top}}th{{color:var(--muted);font-weight:500}}
a{{color:var(--accent)}}.btn{{display:inline-block;background:var(--accent);color:#fff;padding:8px 14px;border-radius:8px;text-decoration:none;font-weight:600}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}.warn{{background:var(--warn);color:var(--warnt);border-radius:8px;padding:10px 14px}}.chart{{position:relative;height:260px}}</style></head><body><main>
<h1>Month-ahead weather fade: live</h1>
<div class="muted">Updated {pd.Timestamp.now(tz='UTC').strftime('%a %d %b %Y %H:%M')} UTC · <a href="index.html">dashboard</a> · <a href="weather-fade.html">research write-up</a></div>
{f'<div class="card warn"><b>Needs attention</b><ul>{err}</ul></div>' if err else ''}
<div class="card"><h2>Fair prices right now</h2>
<p class="muted">Our fair price assumes the recent weather won't carry on and next period's weather is normal for the time of year. The 14-day spot average is what a market anchored on recent prices would lean toward.</p>
<table><thead><tr><th>Contract</th><th>Our fair price</th><th>Last 14 days' spot</th><th>Your quote</th><th>Unit</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>
<p style="margin-top:12px"><a class="btn" href="{EDIT_URL}">Enter market prices</a></p>
<p class="muted">Opens quotes.json on GitHub: type the prices you see into "price", then <b>Commit changes</b>. A run starts within a minute or two; refresh this page after ~3 minutes.</p></div>
<div class="card"><h2>Trades</h2><p class="muted">{settled} settled · total {money(total)} · power 1 MW baseload, gas 25,000 therms/day · costs £1/MWh and 0.5p/therm</p>
<table><thead><tr><th>Contract</th><th>Decided (UTC)</th><th>Market quote</th><th>Our fair</th><th>Gap</th><th>Side</th><th>Status</th><th>Actual average</th><th>P&L</th></tr></thead>
<tbody>{trades or '<tr><td colspan="9" class="muted">No trades yet - enter a quote to make the first one.</td></tr>'}</tbody></table></div>
<div class="card"><h2>How our fair price has moved</h2><div class="chart"><canvas id="c"></canvas></div></div>
<p class="muted">Paper trading only; trades settle against daily spot averages (Elexon APX index for power, National Gas SAP for gas), a stand-in for the futures' own settlement. Not financial advice.</p>
</main><script>
const H = {json.dumps(hist)}; const names = Object.keys(H); const labels = [...new Set(names.flatMap(n => H[n].map(r => r[0])))].sort();
const colors = ['#2563eb', '#e0662a', '#16a34a', '#9333ea'];
new Chart(document.getElementById('c'), {{type: 'line', data: {{labels, datasets: names.map((n, i) => {{
  const m = Object.fromEntries(H[n].map(r => [r[0], r[1]])); return {{label: n + ' fair', data: labels.map(d => m[d] ?? null),
  borderColor: colors[i % 4], backgroundColor: colors[i % 4], pointRadius: 3, spanGaps: true}}; }})}},
  options: {{responsive: true, maintainAspectRatio: false, animation: false, plugins: {{legend: {{labels: {{boxWidth: 12}}}}}}}}}});
</script></body></html>"""
    PAGE.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RUN_LOG.write_text(f"{pd.Timestamp.now(tz='UTC')}: FAILED\n{traceback.format_exc()}", encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1)
