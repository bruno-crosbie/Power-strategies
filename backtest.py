"""Backtest the three strategies on 12 months of history.

Parameters are chosen on the first ~8.5 months (train) and then judged on the untouched last
~3.5 months (test), so the test numbers are not tuned to fit. Writes results/backtest.json.
"""
import json

import numpy as np
import pandas as pd

from common import COST_PER_MWH, DATA, ROOT
from strategies import STRATEGIES, features, positions

SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
MIN_TRAIN_TRADES = 100
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def stats(pos: pd.Series, p: pd.DataFrame) -> dict:
    ok = p.spread.notna()
    pos = pos.where(ok, 0)
    gross = (pos * p.spread.fillna(0) * 0.5)
    net = gross - pos.abs() * COST_PER_MWH * 0.5
    traded = pos != 0
    daily = net.groupby(p.start.dt.date).sum()
    curve = net.cumsum()
    return {
        "trades": int(traded.sum()),
        "longs": int((pos > 0).sum()),
        "shorts": int((pos < 0).sum()),
        "gross_gbp": round(float(gross.sum()), 0),
        "net_gbp": round(float(net.sum()), 0),
        "net_per_trade": round(float(net[traded].mean()), 2) if traded.any() else 0.0,
        "win_rate": round(float((gross[traded] > 0).mean()), 3) if traded.any() else 0.0,
        "max_drawdown_gbp": round(float((curve - curve.cummax()).min()), 0),
        "sharpe_daily_ann": round(float(daily.mean() / daily.std() * np.sqrt(365)), 2) if daily.std() > 0 else 0.0,
    }


def main():
    sp, mid, wf, wa, rm = (pd.read_pickle(DATA / f"{n}.pkl") for n in
                           ("system_prices", "mid", "windfor", "wind_actual", "remit"))
    p = features(sp, mid, wf, wa)
    train, test = p.start < SPLIT, p.start >= SPLIT
    report = {"period": [str(p.start.min()), str(p.start.max())], "split": str(SPLIT),
              "cost_per_mwh": COST_PER_MWH, "strategies": {}}

    for key, s in STRATEGIES.items():
        grid = []
        for params in s["grid"]:
            pos = positions(key, p, rm, params)
            grid.append({"params": params, "train": stats(pos[train], p[train]), "test": stats(pos[test], p[test]),
                         "_pos": pos})
        eligible = [g for g in grid if g["train"]["trades"] >= MIN_TRAIN_TRADES] or grid
        best = max(eligible, key=lambda g: g["train"]["net_gbp"])
        pos = best["_pos"]
        net = (pos * p.spread.fillna(0) * 0.5 - pos.abs() * COST_PER_MWH * 0.5)
        daily_curve = net.groupby(p.start.dt.date).sum().cumsum()
        report["strategies"][key] = {
            "name": s["name"],
            "chosen_params": best["params"],
            "train": best["train"],
            "test": best["test"],
            "full_year": stats(pos, p),
            "grid": [{k: v for k, v in g.items() if k != "_pos"} for g in grid],
            "equity_curve": {str(d): round(float(v), 1) for d, v in daily_curve.items()},
        }
        print(f"\n{s['name']}: chose {best['params']}")
        print(f"  train {best['train']}")
        print(f"  test  {best['test']}")

    (OUT / "backtest.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", OUT / "backtest.json")


if __name__ == "__main__":
    main()
