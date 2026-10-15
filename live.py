"""Forward paper-trading run. Scheduled every 30 minutes for 10 days.

Each run: refresh recent Elexon data, decide each strategy's position for the next tradeable
half-hour (first period whose gate closure hasn't passed), log it, settle earlier trades whose
imbalance price is now published, and rebuild the dashboard. No real orders are placed.
"""
import json
import logging
import sys
import traceback

import pandas as pd

from common import (COST_PER_MWH, DATA, GATE_CLOSURE, ROOT, fetch_mid, fetch_remit,
                    fetch_system_prices, fetch_wind_actual, fetch_windfor)
from strategies import STRATEGIES, features, positions, remit_events

RESULTS = ROOT / "results"
LIVE = DATA / "live"
TRADES = RESULTS / "live_trades.csv"
STATE = RESULTS / "live_state.json"
DAYS = 10
KEEP = pd.Timedelta(days=40)

RESULTS.mkdir(exist_ok=True)
LIVE.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(RESULTS / "live.log"), logging.StreamHandler()])
log = logging.getLogger("live")


def load(name):
    """Local cache if there is one; in the cloud there isn't, so refresh() pulls the last 40 days."""
    for f in (LIVE / f"{name}.pkl", DATA / f"{name}.pkl"):
        if f.exists():
            return pd.read_pickle(f)
    return pd.DataFrame()


def refresh(now):
    """Top up each cached dataset with whatever was published since it was last updated."""
    def upd(name, fetch, time_col, keys, overlap):
        old = load(name)
        since = (old[time_col].max() if len(old) else now - KEEP) - overlap
        new = fetch(since, now + pd.Timedelta(days=1))
        df = pd.concat([old, new]) if len(old) else new
        df = df.drop_duplicates(keys, keep="last").sort_values(time_col)
        df = df[df[time_col] >= now - KEEP]
        df.to_pickle(LIVE / f"{name}.pkl")
        return df
    sp = upd("system_prices", fetch_system_prices, "start", ["start"], pd.Timedelta(days=1))
    mid = upd("mid", fetch_mid, "start", ["start"], pd.Timedelta(days=1))
    wa = upd("wind_actual", fetch_wind_actual, "start", ["start"], pd.Timedelta(days=1))
    wf = upd("windfor", lambda s, e: fetch_windfor(s, min(e, now)), "publish", ["publish", "target"], pd.Timedelta(hours=6))
    rm = upd("remit", lambda s, e: fetch_remit(s, min(e, now)), "publishTime", ["mrid", "revisionNumber"], pd.Timedelta(hours=1))
    return sp, mid, wf, wa, rm


def note(key, row, rm, params, t):
    if key == "weather_fade":
        return f"wind fc {row.fc:,.0f} MW ({row.fc_pct:.0%} pctile), recent fc error {row.err_recent:+,.0f} MW"
    if key == "imbalance_momentum":
        return f"recent NIV {row.niv_sig:+,.0f} MWh ({row.niv_abs_pct:.0%} pctile of size)"
    ev = remit_events(rm, params["min_mw"], params.get("sudden", False))
    ev = ev[(ev.publishTime > t - pd.Timedelta(hours=params["hold_h"] + 2)) & (ev.publishTime <= t)]
    return "; ".join(f"{e.assetId} {e.fuelType} -{e.lost_mw:,.0f} MW" for e in ev.itertuples()) or "no qualifying outage"


def main():
    now = pd.Timestamp.now(tz="UTC")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "start" not in state:
        state["start"] = str(now)
        STATE.write_text(json.dumps(state))
    end = pd.Timestamp(state["start"]) + pd.Timedelta(days=DAYS)
    trading = now < end

    sp, mid, wf, wa, rm = refresh(now)
    target = (now + GATE_CLOSURE).ceil("30min")
    p = features(sp, mid, wf, wa, until=target)
    params = {k: v["chosen_params"] for k, v in json.loads((RESULTS / "backtest.json").read_text())["strategies"].items()}

    trades = pd.read_csv(TRADES, parse_dates=["period_start", "decided_at"]) if TRADES.exists() else pd.DataFrame(
        columns=["strategy", "period_start", "decided_at", "position", "note", "mid", "sip", "pnl_gross", "pnl_net", "status"])

    if trading:
        row_i = p.index[p.start == target][0]
        row = p.loc[row_i]
        new = []
        for key in STRATEGIES:
            done = ((trades.strategy == key) & (pd.to_datetime(trades.period_start, utc=True) == target)).any()
            if done:
                continue
            pos = int(positions(key, p, rm, params[key]).loc[row_i])
            new.append({"strategy": key, "period_start": target, "decided_at": now, "position": pos,
                        "note": note(key, row, rm, params[key], target),
                        "status": "open" if pos else "flat"})
        if new:
            trades = pd.concat([trades, pd.DataFrame(new)], ignore_index=True)
        log.info("decided %s: %s", target, {n["strategy"]: n["position"] for n in new})

    # settle open trades once both prices for their period are published
    trades["period_start"] = pd.to_datetime(trades.period_start, utc=True)
    prices = p.set_index("start")[["mid", "sip"]]
    for i in trades.index[trades.status == "open"]:
        t = trades.at[i, "period_start"]
        if t in prices.index and pd.notna(prices.at[t, "mid"]) and pd.notna(prices.at[t, "sip"]):
            m, s, pos = prices.at[t, "mid"], prices.at[t, "sip"], trades.at[i, "position"]
            gross = pos * (s - m) * 0.5
            trades.loc[i, ["mid", "sip", "pnl_gross", "pnl_net", "status"]] = [
                m, s, round(gross, 2), round(gross - abs(pos) * COST_PER_MWH * 0.5, 2), "settled"]
    trades.sort_values(["period_start", "strategy"]).to_csv(TRADES, index=False)

    import report
    report.build()
    if not trading:
        log.info("10-day window ended %s; settled remaining trades only", end)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("run failed:\n%s", traceback.format_exc())
        sys.exit(1)
