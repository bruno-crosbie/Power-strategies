"""Build results/report.html from the per-strategy JSON files. Plain HTML + Chart.js, opens offline
(apart from loading Chart.js from a CDN). Plain-English conclusions live in findings.py."""
from __future__ import annotations

import html
import json

import pandas as pd

from config import RESULTS

ORDER = ["power_week_ahead", "nbp_week_ahead", "weather_month_ahead", "norway_supply", "brent_trend", "oil_carry", "brent_wti_spread"]
SCENARIO_KEYS = {"power_week_ahead", "nbp_week_ahead", "weather_month_ahead"}


def load() -> dict:
    out = {}
    for k in ORDER:
        f = RESULTS / f"{k}.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    return out


def headline(r: dict) -> dict:
    """The numbers the scoreboard shows. For the fade strategies that is the pure fade (market trusts the
    forecast 100%) on the main product, plus the 50% case."""
    if r["key"] == "weather_month_ahead":
        sc = r["stats"]["power_month"]
        return {"selection": sc["100% on last 14 days"]["selection"], "holdout": sc["100% on last 14 days"]["holdout"],
                "alt": sc["50% on last 14 days"]["holdout"], "alt_label": "if the market only half-anchors on spot"}
    if r["key"] in SCENARIO_KEYS:
        sc = r["stats"]["week_ahead"]
        return {"selection": sc["100%"]["selection"], "holdout": sc["100%"]["holdout"],
                "alt": sc["50%"]["holdout"], "alt_label": "if the market only half-trusts forecasts"}
    return {"selection": r["stats"]["selection"], "holdout": r["stats"]["holdout"]}


def verdict(h: dict) -> tuple[str, str]:
    s, t = h["selection"], h["holdout"]
    if not t.get("trades"):
        return "No signal", "grey"
    if t["pnl"] <= 0:
        return "Doesn't work", "red"
    if t.get("t_stat", 0) >= 2 and s.get("t_stat", 0) >= 1.5:
        return "Works in testing", "green"
    if t.get("t_stat", 0) >= 1:
        return "Promising, unproven", "amber"
    return "Inconclusive", "amber"


def money(v, cur):
    if v is None:
        return "–"
    return f"{'-' if v < 0 else ''}{cur}{abs(v):,.0f}"


def esc(x):
    return html.escape(str(x))


def stats_table(r):
    cur = r["currency"]
    cols = [("Period", lambda s: f"{s.get('from', '')} → {s.get('to', '')}"), ("Trades", lambda s: s.get("trades", 0)),
            ("Total P&L", lambda s: money(s.get("pnl"), cur)), ("P&L per year", lambda s: money(s.get("pnl_per_year"), cur)),
            ("Win rate", lambda s: f"{s.get('hit_rate', 0):.0%}"), ("t-stat", lambda s: s.get("t_stat", 0)),
            ("Sharpe", lambda s: s.get("sharpe", 0)), ("Worst drawdown", lambda s: money(s.get("max_drawdown"), cur))]
    h = headline(r)
    rows = {"Tuning period": h["selection"], "Test period (untouched)": h["holdout"]}
    head = "".join(f"<th>{c}</th>" for c, _ in cols)
    body = "".join(f"<tr><td><b>{k}</b></td>" + "".join(f"<td>{esc(f(v))}</td>" for _, f in cols[0:]) + "</tr>"
                   for k, v in rows.items())
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"


def scenario_tables(r):
    cur = r["currency"]
    out = []
    for prod, info in r["extra"]["products"].items():
        rows = ""
        for m, st in info["scenarios"].items():
            s, t = st["selection"], st["holdout"]
            label = ({"0%": "ignores the forecast", "100%": "fully trusts the forecast (pure fade)"}.get(m, f"trusts it {m}")
                     if r["key"] != "weather_month_ahead" else f"is anchored {m}")
            rows += (f"<tr><td>{esc(label)}</td><td>{s.get('trades', 0)}</td><td class='{('pos' if s.get('pnl', 0) > 0 else 'neg')}'>{money(s.get('pnl'), cur)}</td>"
                     f"<td>{s.get('t_stat', 0)}</td><td>{t.get('trades', 0)}</td><td class='{('pos' if t.get('pnl', 0) > 0 else 'neg')}'>{money(t.get('pnl'), cur)}</td><td>{t.get('t_stat', 0)}</td></tr>")
        acc = info["accuracy"]
        if r["key"] == "weather_month_ahead":
            names = {"fair": "Our fair price (normal weather)", "spot14": "Last 14 days' spot", "spot28": "Last 28 days' spot"}
            acc_rows = "".join(f"<tr><td>{names[k]}</td><td>{acc['selection'][k]['rmse_pct']}%</td><td>{acc['holdout'][k]['rmse_pct']}%</td></tr>"
                               for k in names)
            acc_rows += (f"<tr><td>Fair closer than 14-day spot</td><td>{acc['selection']['fair_closer_share']:.0%} of the time</td>"
                         f"<td>{acc['holdout']['fair_closer_share']:.0%} of the time</td></tr>")
            out.append(f"""<h4>{esc(info['description'])}</h4>
<p class="muted">Setting chosen on the tuning period: {esc(info['chosen_model'])}</p>
<div class="two"><div><table><thead><tr><th>If the market price…</th><th>Tuning trades</th><th>Tuning P&L</th><th>t</th><th>Test trades</th><th>Test P&L</th><th>t</th></tr></thead><tbody>{rows}</tbody></table></div>
<div><table><thead><tr><th>Typical % miss predicting the period</th><th>Tuning</th><th>Test</th></tr></thead><tbody>{acc_rows}</tbody></table></div></div>""")
            continue
        acc_rows = "".join(f"<tr><td>{esc({'fair': 'Our fair price (forecast pulled back)', 'face': 'Forecast at face value', 'naive': 'Recent prices (ignore weather)', 'level': 'Weather-neutral price'}.get(k, k))}</td>"
                           f"<td>{acc['selection'][k]['rmse']}</td><td>{acc['holdout'][k]['rmse']}</td></tr>" for k in acc["selection"])
        out.append(f"""<h4>{esc(info['description'])}</h4>
<p class="muted">Model variant chosen on the tuning period: {esc(info['chosen_model'])}</p>
<div class="two"><div><table><thead><tr><th>If the market price…</th><th>Tuning trades</th><th>Tuning P&L</th><th>t</th><th>Test trades</th><th>Test P&L</th><th>t</th></tr></thead><tbody>{rows}</tbody></table></div>
<div><table><thead><tr><th>Forecast accuracy (typical error)</th><th>Tuning</th><th>Test</th></tr></thead><tbody>{acc_rows}</tbody></table>
<p class="muted">Lower is better. Error is the root-mean-square miss in {esc('£/MWh' if r['key'] == 'power_week_ahead' else 'p/therm')}.</p></div></div>""")
    return "".join(out)


def grid_table(r):
    if not r.get("grid"):
        return ""
    cur = r["currency"]
    rows = "".join(
        f"<tr{' class=chosen' if g['setting'] == r['chosen'] else ''}><td>{esc(g['setting'])}</td><td>{g.get('sel_trades', 0)}</td>"
        f"<td>{money(g.get('sel_pnl'), cur)}</td><td>{g.get('sel_t_stat', 0)}</td><td>{g.get('hold_trades', 0)}</td>"
        f"<td>{money(g.get('hold_pnl'), cur)}</td><td>{g.get('hold_t_stat', 0)}</td></tr>"
        for g in sorted(r["grid"], key=lambda g: -g.get("sel_t_stat", 0)))
    return f"""<details><summary>Every setting tried ({len(r['grid'])}), best tuning score first — the highlighted row was chosen</summary>
<table><thead><tr><th>Setting</th><th>Tuning trades</th><th>Tuning P&L</th><th>t</th><th>Test trades</th><th>Test P&L</th><th>t</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">Look down the test columns: if most settings make money in the test period, the idea is robust; if only the chosen one does, it is probably luck.</p></details>"""


def extras(r):
    k, e, cur = r["key"], r.get("extra", {}), r["currency"]
    if k == "power_week_ahead":
        rows = "".join(f"<tr><td>{s['days_ahead']}</td><td>{s['forecast_vs_actual_corr']}</td><td>{s['trust_factor']}</td>"
                       f"<td>{s['error_raw']}</td><td>{s['error_pulled_back']}</td></tr>" for s in e["skill"])
        return f"""<h4>How much can you trust a wind forecast?</h4><table><thead><tr><th>Days ahead</th><th>Correlation with what happened</th>
<th>Trust factor (1 = take at face value)</th><th>Error, raw (capacity factor)</th><th>Error, pulled back</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">The trust factor falling below 1 at longer leads is the whole basis of the fade: a forecast for 6-7 days out should be pulled part of the way back toward normal.</p>"""
    if k == "nbp_week_ahead":
        s = e["price_sensitivity"]
        return f"""<h4>What moves NBP (learned from 2018-2026)</h4><ul>
<li>Each extra GB heating degree day vs the last week: <b>{s['per_extra_GB_HDD_pct']:+.2f}%</b></li>
<li>Each extra Continental heating degree day: <b>{s['per_extra_EU_HDD_pct']:+.2f}%</b></li>
<li>GB wind capacity factor 0.1 higher: <b>{s['per_+0.1_wind_CF_pct']:+.2f}%</b></li>
<li>Weekend day vs weekday mix: <b>{s['weekend_pct']:+.2f}%</b></li></ul>"""
    if k == "norway_supply":
        rows = "".join(f"<tr><td>{x['after_days']}</td><td>{esc(x['group'])}</td><td>{x['n']}</td><td>{x['avg_change_pct']:+.2f}%</td></tr>" for x in e["event_study"])
        return f"""<h4>What gas prices did after flow changes (all days, no costs)</h4><table><thead><tr><th>Days later</th><th>After…</th><th>Cases</th><th>Average gas price change</th></tr></thead><tbody>{rows}</tbody></table>"""
    if k == "brent_trend":
        b = e["buy_and_hold"]["holdout"]
        return f"<p>For comparison, simply holding 1 contract long over the test period: {money(b['pnl'], cur)} (Sharpe {b['sharpe']}, worst drawdown {money(b['max_drawdown'], cur)}).</p>"
    if k == "oil_carry":
        b = e["long_only"]["holdout"]
        f = e["curve_facts"]
        return (f"<p>Always long the rolled futures over the test period: {money(b['pnl'], cur)} (Sharpe {b['sharpe']}). "
                f"The curve was backwardated {f['share_of_weeks_backwardated']:.0%} of weeks.</p>")
    if k == "brent_wti_spread":
        yrs = " · ".join(f"{y}: ${v:.1f}" for y, v in list(e["gap_by_year"].items())[-12:])
        reasons = ", ".join(f"{k2}: {v}" for k2, v in e["exit_reasons"].items())
        return f"<p><b>Average Brent-WTI gap by year:</b> {yrs}. Latest: <b>${e['latest_gap']:.2f}</b>.</p><p>How trades ended: {esc(reasons)}.</p>"
    return ""


def recent_table(r):
    t = r.get("recent_trades") or []
    if not t:
        return ""
    cols = list(t[0].keys())
    body = "".join("<tr>" + "".join(f"<td>{esc(x[c])}</td>" for c in cols) + "</tr>" for x in t)
    return f"<details><summary>Most recent trades</summary><table><thead><tr>{''.join(f'<th>{esc(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table></details>"


def section(r, findings):
    h = headline(r)
    v, colour = verdict(h)
    chart_id = f"c_{r['key']}"
    alt = ""
    if "alt" in h:
        alt = f"<p class='muted'>Test-period result {esc(h['alt_label'])}: {money(h['alt'].get('pnl'), r['currency'])} (t {h['alt'].get('t_stat', 0)}).</p>"
    body = scenario_tables(r) if r["key"] in SCENARIO_KEYS else stats_table(r)
    find = findings.get(r["key"], "")
    return f"""
<section id="{r['key']}">
  <div class="head"><h2>{esc(r['name'])}</h2><span class="badge {colour}">{v}</span></div>
  <p class="lead">{esc(r['summary'])}</p>
  {f'<div class="finding"><b>What we found.</b> {find}</div>' if find else ''}
  <div class="two">
    <div><h3>How it works</h3><ol>{''.join(f'<li>{esc(x)}</li>' for x in r['how_it_works'])}</ol>
      <p class="muted">Size: {esc(r['unit'])}. Tuning period ends {esc(r['split'])}; everything after is the untouched test.</p></div>
    <div><h3>Growth of profit &amp; loss</h3><div class="chart"><canvas id="{chart_id}"></canvas></div>
      <p class="muted">Cumulative P&L after costs. The test period starts at {esc(r['split'])}.</p></div>
  </div>
  <h3>Results</h3>{body}{alt}
  {extras(r)}
  {grid_table(r)}
  <h3>Things to be aware of</h3><ul>{''.join(f'<li>{esc(x)}</li>' for x in r['caveats'])}</ul>
  <p class="muted"><b>Data:</b> {esc('; '.join(r['data']))}</p>
  {recent_table(r)}
</section>"""


def scoreboard(res, findings):
    rows = ""
    for r in res.values():
        h = headline(r)
        v, colour = verdict(h)
        t = h["holdout"]
        rows += (f"<tr><td><a href='#{r['key']}'>{esc(r['name'])}</a></td><td>{esc(r['asset'])}</td>"
                 f"<td>{esc(t.get('from', ''))} → {esc(t.get('to', ''))}</td><td>{t.get('trades', 0)}</td>"
                 f"<td class='{('pos' if t.get('pnl', 0) > 0 else 'neg')}'>{money(t.get('pnl'), r['currency'])}</td>"
                 f"<td>{t.get('t_stat', 0)}</td><td><span class='badge {colour}'>{v}</span></td></tr>")
    return f"""<table class="score"><thead><tr><th>Strategy</th><th>Market</th><th>Test period</th><th>Trades</th><th>Test P&L</th><th>t-stat</th><th>Verdict</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">Fade strategies are scored as a pure fade (a market that fully trusts the forecast). Verdicts: <b>works</b> = test t-stat ≥ 2 and tuning t ≥ 1.5; <b>promising</b> = profitable in test with t ≥ 1; <b>inconclusive</b> = profitable but could easily be luck; <b>doesn't work</b> = lost money in the test.</p>"""


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#16191d;--muted:#5f6670;--border:#e3e6ea;--accent:#2563eb;--pos:#15803d;--neg:#b91c1c;
--green:#dcfce7;--greent:#166534;--amber:#fef3c7;--ambert:#92400e;--red:#fee2e2;--redt:#991b1b;--grey:#e5e7eb;--greyt:#374151;--hl:#eff6ff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1216;--card:#171b21;--text:#e8eaed;--muted:#9aa3ad;--border:#2a313a;
--accent:#60a5fa;--pos:#4ade80;--neg:#f87171;--green:#14532d;--greent:#bbf7d0;--amber:#422006;--ambert:#fde68a;--red:#450a0a;--redt:#fecaca;--grey:#374151;--greyt:#e5e7eb;--hl:#172554}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:21px;margin:0}h3{font-size:16px;margin:22px 0 8px}h4{font-size:15px;margin:18px 0 6px}
section,.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin:18px 0}
.head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.lead{font-size:16px}.muted{color:var(--muted);font-size:13px}
.badge{font-size:12px;font-weight:600;padding:3px 9px;border-radius:999px;white-space:nowrap}
.green{background:var(--green);color:var(--greent)}.amber{background:var(--amber);color:var(--ambert)}.red{background:var(--red);color:var(--redt)}.grey{background:var(--grey);color:var(--greyt)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13.5px;margin:6px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}th{color:var(--muted);font-weight:500}
tr.chosen td{background:var(--hl);font-weight:600}.pos{color:var(--pos)}.neg{color:var(--neg)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:20px}
.chart{position:relative;height:280px}.finding{background:var(--hl);border-radius:8px;padding:12px 14px;margin:10px 0}
details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}a{color:var(--accent)}
.toc a{margin-right:14px;white-space:nowrap}table{display:block;overflow-x:auto}
"""


def build_report():
    res = load()
    try:
        from findings import FINDINGS, OVERVIEW
    except ImportError:
        FINDINGS, OVERVIEW = {}, ""
    curves = {k: r.get("curve", {}) for k, r in res.items()}
    toc = "".join(f"<a href='#{k}'>{esc(r['name'])}</a>" for k, r in res.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy Strategy Lab</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>{CSS}</style></head><body><main>
<h1>Energy Strategy Lab</h1><p class="muted">Six paper-trading strategies across UK power, NBP gas and crude oil · built {pd.Timestamp.now().strftime('%d %b %Y %H:%M')} · nothing here is financial advice</p>
<div class="card"><h3 style="margin-top:0">How to read this</h3><ul>
<li><b>Paper trading only.</b> No real money; trades are simulated on historical prices, after an allowance for trading costs.</li>
<li><b>Two periods.</b> Every setting was chosen using the <i>tuning</i> period only. The <i>test</i> period was never used to pick anything, so it is the honest estimate of how the strategy would do.</li>
<li><b>t-stat</b> says how unlikely a result is to be luck: around 2 or more is hard to get by chance; below 1 could easily be noise.</li>
<li><b>Sharpe</b> is profit per unit of risk, per year: 0.5 is decent for a single strategy, 1+ is excellent.</li></ul>
{OVERVIEW}</div>
<div class="card"><h3 style="margin-top:0">Scoreboard</h3>{scoreboard(res, FINDINGS)}<p class="toc">{toc}</p></div>
{''.join(section(r, FINDINGS) for r in res.values())}
<p class="muted">Data: Elexon, Open-Meteo (ECMWF, GFS, ICON forecasts; ERA5), ONS / National Gas, ENTSOG, FRED, US EIA. Built locally; not published anywhere.</p>
</main>
<script>
const CURVES = {json.dumps(curves)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const palette = ['#2563eb', '#e0662a', '#16a34a', '#9333ea'];
for (const [key, series] of Object.entries(CURVES)) {{
  const el = document.getElementById('c_' + key); if (!el) continue;
  const names = Object.keys(series); if (!names.length) continue;
  const labels = [...new Set(names.flatMap(n => series[n].map(p => p[0])))].sort();
  const datasets = names.map((n, i) => {{ const m = Object.fromEntries(series[n]); let last = null;
    return {{label: n, data: labels.map(d => (m[d] !== undefined ? (last = m[d]) : last)), borderColor: palette[i % 4],
             backgroundColor: palette[i % 4], borderWidth: 1.8, pointRadius: 0, tension: 0}}; }});
  new Chart(el, {{type: 'line', data: {{labels, datasets}}, options: {{responsive: true, maintainAspectRatio: false, animation: false,
    interaction: {{mode: 'index', intersect: false}}, plugins: {{legend: {{labels: {{color: css('--muted'), boxWidth: 12}}}}}},
    scales: {{x: {{ticks: {{color: css('--muted'), maxTicksLimit: 8, maxRotation: 0}}, grid: {{color: css('--border')}}}},
             y: {{ticks: {{color: css('--muted')}}, grid: {{color: css('--border')}}}}}}}}}});
}}
</script></body></html>"""
    path = RESULTS / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_report())
