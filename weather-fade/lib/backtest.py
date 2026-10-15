"""Shared backtest helpers: one way to score trades, one way to choose settings, one way to report.

Every strategy produces a trades table with at least:
    date      when the trade is decided / entered
    position  signed size (+ long, - short, 0 flat)
    pnl       profit or loss in the strategy's currency after costs
and the helpers below turn it into comparable statistics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import RESULTS


def stats(trades: pd.DataFrame, periods_per_year: float) -> dict:
    """Headline numbers for a trades table (rows with position 0 count as 'no trade')."""
    if trades.empty:
        return {"trades": 0, "pnl": 0.0}
    t = trades.sort_values("date")
    active = t[t.position != 0]
    curve = t.pnl.cumsum()
    per = t.pnl
    sharpe = per.mean() / per.std() * np.sqrt(periods_per_year) if per.std() > 0 else 0.0
    tstat = active.pnl.mean() / (active.pnl.std() / np.sqrt(len(active))) if len(active) > 2 and active.pnl.std() > 0 else 0.0
    years = max((t.date.max() - t.date.min()).days / 365.25, 1e-9)
    return {
        "from": str(t.date.min().date()), "to": str(t.date.max().date()), "years": round(years, 1),
        "decisions": int(len(t)), "trades": int(len(active)),
        "longs": int((active.position > 0).sum()), "shorts": int((active.position < 0).sum()),
        "pnl": round(float(t.pnl.sum()), 0), "pnl_per_year": round(float(t.pnl.sum() / years), 0),
        "pnl_per_trade": round(float(active.pnl.mean()), 1) if len(active) else 0.0,
        "hit_rate": round(float((active.pnl > 0).mean()), 3) if len(active) else 0.0,
        "t_stat": round(float(tstat), 2), "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float((curve - curve.cummax()).min()), 0),
    }


def split_stats(trades: pd.DataFrame, split: str, periods_per_year: float) -> dict:
    s = pd.Timestamp(split)
    return {"selection": stats(trades[trades.date < s], periods_per_year),
            "holdout": stats(trades[trades.date >= s], periods_per_year),
            "all": stats(trades, periods_per_year)}


def choose(results: dict, split: str, periods_per_year: float, min_trades: int = 20, key: str = "sharpe"):
    """Pick the setting with the best selection-period score (needs min_trades). Never looks at holdout."""
    table = []
    for name, trades in results.items():
        s = split_stats(trades, split, periods_per_year)
        table.append({"setting": name, **{f"sel_{k}": v for k, v in s["selection"].items() if k in ("trades", "pnl", "sharpe", "hit_rate", "t_stat")},
                      **{f"hold_{k}": v for k, v in s["holdout"].items() if k in ("trades", "pnl", "sharpe", "hit_rate", "t_stat")}})
    ok = [r for r in table if r["sel_trades"] >= min_trades] or table
    best = max(ok, key=lambda r: r[f"sel_{key}"])["setting"]
    return best, table


def equity(trades: pd.DataFrame, freq: str = "W") -> list:
    t = trades.sort_values("date").set_index("date").pnl.cumsum()
    t = t.resample(freq).last().ffill()
    return [[str(d.date()), round(float(v), 0)] for d, v in t.items()]


@dataclass
class Result:
    """Everything the report needs for one strategy."""
    key: str
    name: str
    asset: str
    currency: str
    unit: str
    summary: str
    how_it_works: list
    data: list
    caveats: list
    split: str
    chosen: str = ""
    stats: dict = field(default_factory=dict)
    grid: list = field(default_factory=list)
    curve: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    recent_trades: list = field(default_factory=list)

    def save(self):
        (RESULTS / f"{self.key}.json").write_text(json.dumps(self.__dict__, indent=1, default=str))
        return self


def recent(trades: pd.DataFrame, cols: list, n: int = 15) -> list:
    t = trades[trades.position != 0].sort_values("date").tail(n)
    out = []
    for _, r in t.iterrows():
        out.append({c: (str(r[c].date()) if isinstance(r[c], pd.Timestamp) else
                        round(float(r[c]), 2) if isinstance(r[c], (float, np.floating, int, np.integer)) else str(r[c]))
                    for c in cols})
    return out
