"""Builds results/dashboard.html from the backtest and live trade log."""
import json

import pandas as pd

from common import COST_PER_MWH, ROOT

RESULTS = ROOT / "results"
NAMES = {"weather_fade": "Weather fade", "remit_outage": "REMIT outage", "imbalance_momentum": "Imbalance momentum"}


def build():
    bt = json.loads((RESULTS / "backtest.json").read_text())
    state = json.loads((RESULTS / "live_state.json").read_text()) if (RESULTS / "live_state.json").exists() else {}
    trades = pd.read_csv(RESULTS / "live_trades.csv") if (RESULTS / "live_trades.csv").exists() else pd.DataFrame(
        columns=["strategy", "period_start", "position", "note", "mid", "sip", "pnl_gross", "pnl_net", "status"])
    trades["period_start"] = pd.to_datetime(trades.period_start, utc=True)

    live = {}
    for key, name in NAMES.items():
        t = trades[trades.strategy == key].sort_values("period_start")
        settled = t[t.status == "settled"]
        last = t.iloc[-1] if len(t) else None
        live[key] = {
            "name": name,
            "net": round(float(settled.pnl_net.sum()), 2),
            "gross": round(float(settled.pnl_gross.sum()), 2),
            "settled": int(len(settled)),
            "open": int((t.status == "open").sum()),
            "decisions": int(len(t)),
            "wins": int((settled.pnl_gross > 0).sum()),
            "last_pos": int(last.position) if last is not None else None,
            "last_period": last.period_start.tz_convert("Europe/London").strftime("%a %H:%M") if last is not None else "",
            "last_note": str(last.note) if last is not None else "",
            "curve": [[r.period_start.isoformat(), round(float(c), 2)]
                      for r, c in zip(settled.itertuples(), settled.pnl_net.cumsum())],
        }

    recent = trades[trades.position != 0].sort_values("period_start", ascending=False).head(40)
    recent_rows = [{
        "strategy": NAMES.get(r.strategy, r.strategy),
        "period": r.period_start.tz_convert("Europe/London").strftime("%a %d %b %H:%M"),
        "pos": "Long" if r.position > 0 else "Short",
        "mid": None if pd.isna(r.mid) else round(float(r.mid), 2),
        "sip": None if pd.isna(r.sip) else round(float(r.sip), 2),
        "pnl": None if pd.isna(r.pnl_net) else round(float(r.pnl_net), 2),
        "note": str(r.note),
    } for r in recent.itertuples()]

    payload = {
        "updated": pd.Timestamp.now(tz="Europe/London").strftime("%a %d %b %Y %H:%M"),
        "live_start": pd.Timestamp(state["start"]).tz_convert("Europe/London").strftime("%a %d %b %H:%M") if state.get("start") else "not started",
        "live_end": (pd.Timestamp(state["start"]) + pd.Timedelta(days=10)).tz_convert("Europe/London").strftime("%a %d %b %H:%M") if state.get("start") else "",
        "cost": COST_PER_MWH,
        "backtest": {k: {kk: v[kk] for kk in ("name", "chosen_params", "train", "test", "full_year", "equity_curve")}
                     for k, v in bt["strategies"].items()},
        "split": bt["split"][:10],
        "live": live,
        "recent": recent_rows,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    (RESULTS / "dashboard.html").write_text(html, encoding="utf-8")
    docs = ROOT / "docs"          # served by GitHub Pages
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Power Strategy Tracker</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#16191d;--muted:#646b75;--border:#e3e6ea;--grid:rgba(0,0,0,.06);
 --pos:#15803d;--neg:#dc2626;--s1:#2563eb;--s2:#e0662a;--s3:#9333ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1216;--card:#171b21;--text:#e8eaed;--muted:#9aa3ad;
 --border:#262c34;--grid:rgba(255,255,255,.07);--pos:#4ade80;--neg:#f87171;--s1:#60a5fa;--s2:#fb923c;--s3:#c084fc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 40px}h1{font-size:22px;margin:0}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--muted);font-size:13px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.big{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}.pos{color:var(--pos)}.neg{color:var(--neg)}
.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.chart{position:relative;height:300px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);font-size:13px;vertical-align:top}th{color:var(--muted);font-weight:500}
.scroll{overflow-x:auto}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.note{color:var(--muted);font-size:12px}
</style></head><body><main>
<h1>Power Strategy Tracker</h1>
<div class="sub" id="meta"></div>

<h2>Live paper trading (10 days)</h2>
<div class="tiles" id="tiles"></div>
<div class="card" style="margin-top:12px"><div class="chart"><canvas id="live"></canvas></div></div>

<h2>Recent trades</h2>
<div class="card scroll"><table id="recent"><thead><tr><th>Strategy</th><th>Period</th><th>Side</th><th>Entry (MID)</th><th>Settle (SIP)</th><th>Net £</th><th>Why</th></tr></thead><tbody></tbody></table></div>

<h2>Backtest (12 months)</h2>
<div class="card scroll"><table id="bt"><thead><tr><th>Strategy</th><th>Settings chosen on train</th><th>Train net £ (trades, win%)</th><th>Test net £ (trades, win%)</th><th>Test Sharpe</th><th>Full-year max drawdown</th></tr></thead><tbody></tbody></table></div>
<div class="card" style="margin-top:12px"><div class="chart"><canvas id="eq"></canvas></div>
<div class="note" id="eqnote"></div></div>

<p class="note">Paper trading only. Each trade is 1 MW for one half-hour, bought or sold at the Elexon market index price (MID) and settled at the
system imbalance price (SIP); P&amp;L = position &times; (SIP &minus; MID) &times; 0.5 MWh, net of an assumed £<span id="cost"></span>/MWh cost.
Real trading would face extra costs, limited liquidity and access requirements. Not financial advice. Page reloads every 5 minutes.</p>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLORS = {weather_fade: css('--s1'), remit_outage: css('--s2'), imbalance_momentum: css('--s3')};
const gbp = v => (v < 0 ? '-£' : '£') + Math.abs(v).toLocaleString('en-GB', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const cls = v => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
document.getElementById('meta').textContent = `Live window ${D.live_start} → ${D.live_end} · updated ${D.updated}`;
document.getElementById('cost').textContent = D.cost;

const tiles = document.getElementById('tiles');
for (const [k, s] of Object.entries(D.live)) {
  const pos = s.last_pos === null ? '–' : s.last_pos > 0 ? 'Long' : s.last_pos < 0 ? 'Short' : 'Flat';
  tiles.insertAdjacentHTML('beforeend', `<div class="card">
    <div class="label"><span class="dot" style="background:${COLORS[k]}"></span>${s.name}</div>
    <div class="big ${cls(s.net)}">${gbp(s.net)}</div>
    <div class="sub">${s.settled} settled trades · ${s.settled ? Math.round(100 * s.wins / s.settled) : 0}% wins · ${s.open} awaiting prices</div>
    <div class="sub" style="margin-top:6px">Latest (${s.last_period || '–'}): <b>${pos}</b><br>${s.last_note}</div></div>`);
}

const grid = css('--grid'), muted = css('--muted');
const axes = (y) => ({x: {type: 'category', ticks: {color: muted, maxTicksLimit: 10, maxRotation: 0}, grid: {color: grid}},
                      y: {ticks: {color: muted}, grid: {color: grid}, title: {display: true, text: y, color: muted}}});

// live cumulative net P&L, aligned on a shared time axis
const times = [...new Set(Object.values(D.live).flatMap(s => s.curve.map(c => c[0])))].sort();
const fmt = t => new Date(t).toLocaleString('en-GB', {weekday: 'short', hour: '2-digit', minute: '2-digit'});
new Chart(document.getElementById('live'), {type: 'line',
  data: {labels: times.map(fmt), datasets: Object.entries(D.live).map(([k, s]) => {
    const m = Object.fromEntries(s.curve); let last = 0;
    return {label: s.name, data: times.map(t => (m[t] !== undefined ? (last = m[t]) : last)), borderColor: COLORS[k], backgroundColor: COLORS[k],
            borderWidth: 2, pointRadius: 0, stepped: true};
  })},
  options: {responsive: true, maintainAspectRatio: false, animation: false, interaction: {mode: 'index', intersect: false},
    plugins: {legend: {labels: {color: muted, boxWidth: 12}}, title: {display: !times.length, text: 'Waiting for first settled trades…', color: muted}},
    scales: axes('Cumulative net £')}});

const tb = document.querySelector('#recent tbody');
if (!D.recent.length) tb.innerHTML = '<tr><td colspan="7" class="note">No trades yet. Flat decisions are logged but not shown.</td></tr>';
for (const r of D.recent) tb.insertAdjacentHTML('beforeend', `<tr><td>${r.strategy}</td><td>${r.period}</td><td>${r.pos}</td>
  <td>${r.mid ?? '…'}</td><td>${r.sip ?? '…'}</td><td class="${cls(r.pnl)}">${r.pnl === null ? 'pending' : gbp(r.pnl)}</td><td class="note">${r.note}</td></tr>`);

const bt = document.querySelector('#bt tbody');
for (const [k, s] of Object.entries(D.backtest)) {
  const f = x => `<span class="${cls(x.net_gbp)}">${gbp(x.net_gbp)}</span> <span class="note">(${x.trades}, ${Math.round(x.win_rate * 100)}%)</span>`;
  bt.insertAdjacentHTML('beforeend', `<tr><td><span class="dot" style="background:${COLORS[k]}"></span>${s.name}</td>
    <td class="note">${Object.entries(s.chosen_params).map(([a, b]) => a + '=' + b).join(', ')}</td>
    <td>${f(s.train)}</td><td>${f(s.test)}</td><td>${s.test.sharpe_daily_ann}</td><td class="neg">${gbp(s.full_year.max_drawdown_gbp)}</td></tr>`);
}
const days = Object.keys(Object.values(D.backtest)[0].equity_curve);
new Chart(document.getElementById('eq'), {type: 'line',
  data: {labels: days, datasets: Object.entries(D.backtest).map(([k, s]) => ({label: s.name, data: days.map(d => s.equity_curve[d]),
         borderColor: COLORS[k], backgroundColor: COLORS[k], borderWidth: 2, pointRadius: 0}))},
  options: {responsive: true, maintainAspectRatio: false, animation: false, interaction: {mode: 'index', intersect: false},
    plugins: {legend: {labels: {color: muted, boxWidth: 12}}}, scales: axes('Cumulative net £ (1 MW)')}});
document.getElementById('eqnote').textContent = `Settings were picked using data before ${D.split}; everything after that date is an out-of-sample test.`;
</script></body></html>"""


if __name__ == "__main__":
    build()
