"""The three strategies. Each returns a position per half-hour period: +1 long, -1 short, 0 flat,
using only information published before that period's gate closure (start - 60 min).

Feature lags: a period's SIP/MID/wind outturn is treated as known ~30 min after it ends, so at
gate closure for period t the newest known period is t-4 (ends 90 min before t starts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import GATE_CLOSURE, build_periods, latest_forecast_asof

LAG = 4                     # newest settled period known at gate closure
WINDOW = 48 * 30            # trailing 30 days of half-hours for percentile thresholds


def features(sp, mid, wf, wa, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """until extends the grid to future periods (not yet settled) so live runs can score them."""
    p = build_periods(sp, mid)
    end = max(p.start.max(), until) if until is not None else p.start.max()
    grid = pd.date_range(p.start.min(), end, freq="30min", tz="UTC")
    p = p.set_index("start").reindex(grid).rename_axis("start").reset_index()
    p = p.merge(wa, on="start", how="left")
    p["cutoff"] = p.start - GATE_CLOSURE
    p["fc"] = latest_forecast_asof(wf, p.start, p.cutoff)
    # forecast error for past periods, each judged against the forecast it had at its own gate closure
    p["err"] = p.wind - p.fc
    p["err_recent"] = p.err.shift(LAG).rolling(3, min_periods=1).mean()
    p["fc_pct"] = p.fc.rolling(WINDOW, min_periods=48 * 7).rank(pct=True)
    p["niv_sig"] = p.niv.shift(LAG).rolling(2, min_periods=1).mean()
    p["niv_abs_pct"] = p.niv_sig.abs().rolling(WINDOW, min_periods=48 * 7).rank(pct=True)
    return p


# ------------------------------------------------------------- 1. weather fade

WEATHER_GRID = [{"pct": pct, "confirm": c, "side": side}
                for pct in (0.8, 0.9) for c in (False, True) for side in ("both", "calm_only", "windy_only")]


def weather_fade(p: pd.DataFrame, pct: float = 0.9, confirm: bool = False, side: str = "both") -> pd.Series:
    """Fade the price move a wind forecast causes: unusually windy forecast -> prices pushed low ->
    go long; unusually calm -> prices pushed high -> go short. With confirm=True only fade when
    recent outturn says the forecast has been wrong in that direction (windy forecast but wind
    under-delivering, calm forecast but wind over-delivering). side limits it to one leg."""
    long = (p.fc_pct >= pct) & (side != "calm_only")
    short = (p.fc_pct <= 1 - pct) & (side != "windy_only")
    if confirm:
        long &= p.err_recent < 0
        short &= p.err_recent > 0
    return pd.Series(np.select([long, short], [1, -1], 0), index=p.index)


# ------------------------------------------------------------- 2. REMIT outages

REMIT_GRID = [{"min_mw": mw, "hold_h": h, "sudden": s} for mw in (300, 500, 800) for h in (2, 6) for s in (False, True)]
NON_WEATHER_FUELS = {"Fossil Gas", "Nuclear", "Biomass", "Fossil Hard coal", "Fossil Oil",
                     "Hydro Pumped Storage", "Hydro Water Reservoir", "Other"}


def remit_events(rm: pd.DataFrame, min_mw: float, sudden: bool = False) -> pd.DataFrame:
    """First notice of each fresh, unplanned loss of a large non-weather generator. sudden=True keeps
    only trips: outage already started (within 2h before) when the notice went out."""
    r = rm[(rm.unavailabilityType == "Unplanned") & (rm.eventType == "Production unavailability")
           & rm.fuelType.isin(NON_WEATHER_FUELS)].copy()
    r["lost_mw"] = r.normalCapacity - r.availableCapacity
    r = r.sort_values(["publishTime", "revisionNumber"]).drop_duplicates("mrid", keep="first")
    fresh = r.eventStartTime >= r.publishTime - pd.Timedelta(hours=6)   # outage is news, not old
    if sudden:
        fresh &= (r.eventStartTime >= r.publishTime - pd.Timedelta(hours=2)) & \
                 (r.eventStartTime <= r.publishTime + pd.Timedelta(minutes=30))
    return r[fresh & (r.lost_mw >= min_mw)]


def remit_outage(p: pd.DataFrame, rm: pd.DataFrame, min_mw: float = 500, hold_h: float = 6,
                 sudden: bool = False) -> pd.Series:
    """Go long for the hours after a big unplanned outage is announced: the lost supply should
    leave the system short and push the imbalance price above the traded price."""
    pos = pd.Series(0, index=p.index)
    starts = p.start.values
    for e in remit_events(rm, min_mw, sudden).itertuples():
        first = max(e.publishTime + pd.Timedelta(minutes=60), e.eventStartTime)  # first tradeable moment
        last = min(e.eventEndTime if pd.notna(e.eventEndTime) else first + pd.Timedelta(hours=hold_h),
                   first + pd.Timedelta(hours=hold_h))
        i0, i1 = np.searchsorted(starts, np.datetime64(first.tz_convert(None))), np.searchsorted(starts, np.datetime64(last.tz_convert(None)))
        pos.iloc[i0:i1] = 1
    return pos


# ------------------------------------------------------------- 3. imbalance momentum

MOMENTUM_GRID = [{"pct": pct} for pct in (0.7, 0.8, 0.9)]


def imbalance_momentum(p: pd.DataFrame, pct: float = 0.8) -> pd.Series:
    """The system's imbalance direction persists. When the latest known imbalance volume is
    unusually large, bet the next period settles the same way: short system (NIV > 0) -> long."""
    strong = p.niv_abs_pct >= pct
    return pd.Series(np.where(strong, np.sign(p.niv_sig), 0), index=p.index).fillna(0).astype(int)


STRATEGIES = {
    "weather_fade": {"name": "Weather fade", "fn": weather_fade, "grid": WEATHER_GRID, "needs_remit": False},
    "remit_outage": {"name": "REMIT outage", "fn": remit_outage, "grid": REMIT_GRID, "needs_remit": True},
    "imbalance_momentum": {"name": "Imbalance momentum", "fn": imbalance_momentum, "grid": MOMENTUM_GRID, "needs_remit": False},
}


def positions(key: str, p: pd.DataFrame, rm: pd.DataFrame | None, params: dict) -> pd.Series:
    s = STRATEGIES[key]
    return s["fn"](p, rm, **params) if s["needs_remit"] else s["fn"](p, **params)
