"""Weather forecasts turned into daily numbers, honest about timing.

THE TIMING RULE (the most important thing in this file)
A decision is made at 22:00 UTC on its decision day. For every hour of the next 7 days we may only use
a forecast model run that had been *published* by then. A run that starts at time T is treated as public
at T + 8 hours. Open-Meteo's `previous_dayK` is the run that started K days before the hour, so an hour h
may use lead K = ceil((h - decision + 8h) / 24h). Hours that would need K > 7 (the archive's limit) reuse
the last hour that was available - the forecast is carried forward, nothing from the future leaks in.

The main outputs:
    forecast_features(...)   one row per (decision, j = days ahead, target_day), daily-average features
    analysis_features(...)   the same features from the latest run (lead 0) - a stand-in for "what happened"
    WindModel                wind speeds at 8 GB wind-farm areas -> GB wind capacity factor, per weather model
    SkillModel               how much to trust a forecast j days ahead (the basis of every "fade")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DECISION_HOUR_UTC, HDD_BASE_C, MAX_LEAD_DAYS, PUBLISH_DELAY_HOURS, RAW, TZ

KNOTS = (3.0, 6.0, 9.0, 12.0, 15.0)     # m/s: corners of a piecewise-linear wind power curve
CUTOUT = 25.0                           # turbines shut down above this


# ------------------------------------------------------------------ timing

def hour_plan(decisions: pd.DatetimeIndex) -> pd.DataFrame:
    """Every (decision, j, target_day, hour) and the forecast hour/lead each hour is allowed to use."""
    first = decisions.min() + pd.Timedelta(days=1)
    last = decisions.max() + pd.Timedelta(days=MAX_LEAD_DAYS + 1)
    hours = pd.date_range(pd.Timestamp(first, tz=TZ), pd.Timestamp(last, tz=TZ), freq="h", inclusive="left")
    day_hours = pd.DataFrame({"hour": hours.tz_convert("UTC"), "target_day": pd.to_datetime(hours.date)})
    base = pd.DataFrame({"decision": decisions}).merge(pd.DataFrame({"j": range(1, MAX_LEAD_DAYS + 1)}), how="cross")
    base["target_day"] = base.decision + pd.to_timedelta(base.j, unit="D")
    plan = base.merge(day_hours, on="target_day")
    t = plan.decision.dt.tz_localize("UTC") + pd.Timedelta(hours=DECISION_HOUR_UTC)
    k = np.ceil((plan.hour - t + pd.Timedelta(hours=PUBLISH_DELAY_HOURS)) / pd.Timedelta(days=1)).astype(int)
    horizon_end = (t + pd.Timedelta(days=MAX_LEAD_DAYS) - pd.Timedelta(hours=PUBLISH_DELAY_HOURS)).dt.floor("h")
    plan["src_hour"] = plan.hour.where(k <= MAX_LEAD_DAYS, horizon_end)
    plan["lead"] = np.minimum(k, MAX_LEAD_DAYS)
    return plan


def wind_basis(speed: pd.DataFrame) -> dict:
    """Piecewise-linear pieces of wind speed: lets a linear fit learn the power curve shape."""
    s = speed.clip(lower=0)
    running = s < CUTOUT
    out = {"s": s.where(running, 0)}
    for k in KNOTS:
        out[f"h{k:g}"] = (s - k).clip(lower=0).where(running, 0)
    out["cut"] = (~running).astype(float)
    return out


def _wide(fc: pd.DataFrame) -> pd.DataFrame:
    """Long forecast table -> one row per (time, lead), one column per model|site."""
    fc = fc.assign(col=fc.model + "|" + fc.site)
    return fc.pivot_table(index=["time", "lead"], columns="col", values="value").sort_index()


def _to_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "temp":
        return df
    pieces = wind_basis(df)
    return pd.concat({k: v for k, v in pieces.items()}, axis=1).swaplevel(axis=1)  # columns (model|site, piece)


def forecast_features(name: str, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (forecast rows indexed by decision, j, target_day ; analysis rows indexed by target_day).
    Cached in data/raw/feat_<name>.pkl and rebuilt when the raw forecasts are newer."""
    cache = RAW / f"feat_{name}.pkl"
    src = RAW / f"{name}.pkl"
    if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
        return pd.read_pickle(cache)
    fc = pd.read_pickle(src)
    wide = _wide(fc)
    # forecasts as seen at each decision
    times = wide.index.get_level_values("time")
    first = times.min().tz_convert(None).normalize() + pd.Timedelta(days=MAX_LEAD_DAYS + 1)
    decisions = pd.date_range(first, times.max().tz_convert(None).normalize() - pd.Timedelta(days=1), freq="D")
    plan = hour_plan(decisions)
    vals = wide.reindex(pd.MultiIndex.from_arrays([plan.src_hour, plan.lead]))
    vals.index = pd.MultiIndex.from_frame(plan[["decision", "j", "target_day"]])
    vals = vals.dropna(how="all")
    feats = _to_features(vals, kind).groupby(level=[0, 1, 2]).mean()
    # analysis: lead 0, by local day
    a = wide.xs(0, level="lead")
    a.index = pd.to_datetime(a.index.tz_convert(TZ).date)
    analysis = _to_features(a, kind).groupby(level=0).mean()
    analysis.index.name = "target_day"
    pd.to_pickle((feats, analysis), cache)
    return feats, analysis


def mean_temperature(feats: pd.DataFrame) -> pd.Series:
    return feats.mean(axis=1)


def hdd(temp: pd.Series | np.ndarray):
    return np.clip(HDD_BASE_C - temp, 0, None)


# ------------------------------------------------------------------ actual wind

def wind_capacity_factor() -> pd.Series:
    """Daily GB wind capacity factor from Elexon metered output. Fleet size grows over time, so it is
    estimated as the 99.5th percentile of half-hourly output over the previous year (past data only)."""
    w = pd.read_pickle(RAW / "elexon_wind.pkl").set_index("start").wind_mw
    cap = w.rolling("365D", min_periods=48 * 60).quantile(0.995).shift(1)
    df = pd.DataFrame({"w": w, "cap": cap})
    df["day"] = pd.to_datetime(df.index.tz_convert(TZ).date)
    d = df.groupby("day").agg(w=("w", "mean"), cap=("cap", "last"), n=("w", "size"))
    d = d[d.n >= 40]
    return (d.w / d.cap.ffill()).rename("cf").dropna()


# ------------------------------------------------------------------ models

def ridge(X, y, lam=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    reg = lam * len(X) * np.eye(Z.shape[1]); reg[0, 0] = 0
    beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ beta


def ols(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(X1, y, rcond=None)[0]


class WindModel:
    """Per weather model: learn speed -> GB wind capacity factor from analysis vs Elexon outturn."""

    def __init__(self, feats: pd.DataFrame, analysis: pd.DataFrame, cf: pd.Series):
        flat = lambda df: df.set_axis([f"{a}:{b}" for a, b in df.columns], axis=1).astype(float)
        self.feats, self.analysis, self.cf = flat(feats), flat(analysis), cf
        self.models = sorted({c.split("|")[0] for c in self.feats.columns})

    def _cols(self, m):
        return [c for c in self.feats.columns if c.startswith(m + "|")]

    def fit(self, cutoff: pd.Timestamp):
        self.fns = {}
        for m in self.models:
            cols = self._cols(m)
            X = self.analysis[cols].join(self.cf, how="inner").dropna()
            X = X[X.index < cutoff]
            self.fns[m] = (ridge(X[cols].values, X.cf.values), cols)
        return self

    def predict(self, rows: pd.DataFrame | None = None) -> pd.DataFrame:
        f = self.feats if rows is None else rows
        per = {}
        for m, (fn, cols) in self.fns.items():
            X = f[cols].dropna()
            per[m] = pd.Series(np.clip(fn(X.values), 0, 1), index=X.index)
        P = pd.DataFrame(per).dropna()
        return pd.DataFrame({"F": P.mean(1), "S": P.std(1).fillna(0)}).join(P.add_prefix("F_"))


def add_lagged(fc: pd.DataFrame, col: str = "F") -> pd.DataFrame:
    """Attach yesterday evening's forecast for the same target day (j=7 has none -> uses today's)."""
    prev = fc[col].rename(col + "prev").reset_index()
    prev["decision"] = prev.decision + pd.Timedelta(days=1)
    prev["j"] = prev.j - 1
    out = fc.join(prev.set_index(["decision", "j", "target_day"]), how="left")
    out[col + "prev"] = out[col + "prev"].fillna(out[col])
    return out


class SkillModel:
    """For each j (days ahead): actual = a + b * forecast (+ optional terms), with an error size.
    b below 1 means the forecast should be pulled back toward normal - that pull-back IS the fade."""

    def __init__(self, use_spread=False, use_lag=False, var="F", spread="S"):
        self.use_spread, self.use_lag, self.var, self.spread = use_spread, use_lag, var, spread

    def X(self, g):
        cols = [g[self.var]]
        if self.use_spread:
            cols.append(g[self.var] * g[self.spread])
        if self.use_lag:
            cols.append(g[self.var + "prev"])
        return np.column_stack(cols).astype(float)

    def fit(self, fc: pd.DataFrame, actual: pd.Series, cutoff: pd.Timestamp):
        df = fc.reset_index().merge(actual.rename("actual"), left_on="target_day", right_index=True)
        df = df[df.target_day < cutoff].dropna(subset=["actual", self.var])
        self.params = {}
        for j, g in df.groupby("j"):
            X = self.X(g)
            beta = ols(X, g.actual.values)
            resid = g.actual.values - np.c_[np.ones(len(X)), X] @ beta
            if self.spread in g and g[self.spread].std() > 0:
                s = ols(g[[self.spread]].values, np.abs(resid) * np.sqrt(np.pi / 2))
            else:
                s = np.array([np.abs(resid).mean() * np.sqrt(np.pi / 2), 0.0])
            self.params[j] = {"beta": beta, "sd": s, "slope": float(beta[1]), "n": len(g)}
        return self

    def predict(self, fc: pd.DataFrame) -> pd.DataFrame:
        out = []
        for j, g in fc.groupby(level="j"):
            p = self.params[j]
            mu = np.c_[np.ones(len(g)), self.X(g)] @ p["beta"]
            spread = g[self.spread].values if self.spread in g else np.zeros(len(g))
            sd = np.maximum(p["sd"][0] + p["sd"][1] * spread, 1e-3)
            out.append(pd.DataFrame({"mu": mu, "sd": sd}, index=g.index))
        return pd.concat(out).reindex(fc.index)
