"""UK POWER - WEEK-AHEAD WEATHER FADE

Idea
    Next week's GB power price depends mostly on how windy it will be (wind is cheap; calm weeks need
    expensive gas). Traders price next week off the weather forecast, but forecasts 5-7 days out are only
    partly right. If the market takes them at face value, the price for an extreme (very calm / very windy)
    week is too extreme - so we bet the other way: the FADE.

How it is built (refit every 4 weeks, only on data available at the time)
    1. Wind model    3 weather models' wind speeds at 8 wind-farm areas -> GB wind capacity factor (CF).
    2. Skill model   per day-ahead j: actual CF = a + b * forecast CF.  b < 1 = "trust the forecast less".
    3. Price model   daily price / recent weather-neutral level = f(CF, CF^2, cold, weekday).
    Each target day then gets:
        fair  = price expected after pulling the forecast back by the skill model (our view)
        face  = price if the forecast is believed fully (an over-trusting market)
        naive = recent prices (a market that ignores the weather)

The catch
    Real week-ahead quotes are not free, so we do not know where the market actually trades. We test the
    strategy against markets that trust the forecast 0%, 25%, 50%, 75% and 100%. At 100% the strategy is a
    pure fade; at 0% it is the opposite (using the forecast to beat a market that ignores it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (COST_POWER_GBP_MWH, HDD_BASE_C, MIN_EDGE_POWER_GBP_MWH, POWER_POSITION_MW, RAW,
                    REFIT_EVERY_DAYS, SPLITS, TZ, WALK_FORWARD_START)
from lib.backtest import Result, equity, recent, split_stats
from lib.weather import (SkillModel, WindModel, add_lagged, forecast_features, hdd, wind_capacity_factor)

KEY = "power_week_ahead"
MARKETS = [0.0, 0.25, 0.5, 0.75, 1.0]
PRODUCTS = {   # name: (decision weekday or None for every day, days ahead, naive rule, description)
    "week_ahead": (6, tuple(range(1, 8)), "last7", "Sunday night: next Mon-Sun baseload"),
    "weekend": (0, (5, 6), "same_day", "Monday night: the coming Sat-Sun baseload"),
    "day_plus6": (None, (6,), "same_day", "Every night: the single day 6 days out"),
}


# ------------------------------------------------------------------ data

def daily_inputs() -> pd.DataFrame:
    p = pd.read_pickle(RAW / "elexon_prices.pkl")
    p["day"] = pd.to_datetime(p.start.dt.tz_convert(TZ).dt.date)
    price = p.groupby("day").price.agg(["mean", "size"])
    price = price[price["size"] >= 40]["mean"].rename("price")
    temp = pd.read_pickle(RAW / "gb_temp_actual.pkl").groupby("day").temp.mean()
    df = pd.concat([price, wind_capacity_factor(), hdd(temp).rename("hdd")], axis=1)
    df["dow"] = df.index.dayofweek
    return df


# ------------------------------------------------------------------ price model

class PriceModel:
    """price / level = b0 + b1*CF + b2*CF^2 (+ b3*HDD) + weekday effects.
    level = average of the last 14 days' prices with their own weather effect divided out."""

    def __init__(self, use_temp: bool):
        self.use_temp = use_temp

    def _X(self, cf, cf2, h, dow):
        cols = [cf, cf2] + ([h] if self.use_temp else []) + [(dow == k).astype(float) for k in range(1, 7)]
        return np.column_stack(cols)

    def fit(self, d: pd.DataFrame, cutoff, since="2023-01-01"):
        d = d[(d.index >= since) & (d.index < cutoff)].dropna(subset=["price", "cf", "hdd"])
        level = d.price.rolling(28, min_periods=10).median().shift(1)
        for _ in range(2):   # second pass uses a weather-neutral level
            ok = level.notna().values
            X = self._X(d.cf.values, d.cf.values ** 2, d.hdd.values, d.dow.values)
            self.beta = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[ok]], (d.price / level).values[ok], rcond=None)[0]
            factor = pd.Series(self.ratio(d.cf.values, 0, d.hdd.values, d.dow.values), index=d.index)
            level = (d.price / factor.clip(lower=0.2)).rolling(14, min_periods=7).mean().shift(1)
        return self

    def ratio(self, cf_mu, cf_sd, h, dow):
        """Expected price/level when CF ~ Normal(cf_mu, cf_sd); exact for a quadratic: E[CF^2] = mu^2 + sd^2."""
        cf_mu, cf_sd = np.asarray(cf_mu, float), np.asarray(cf_sd, float)
        X = self._X(cf_mu, cf_mu ** 2 + cf_sd ** 2, np.asarray(h, float), np.asarray(dow))
        return np.c_[np.ones(len(X)), X] @ self.beta

    def level(self, d: pd.DataFrame, t, days: int = 14) -> float:
        w = d[(d.index >= t - pd.Timedelta(days=days)) & (d.index < t)].dropna(subset=["price", "cf"])
        f = self.ratio(w.cf.values, np.zeros(len(w)), w.hdd.fillna(0).values, w.dow.values)
        return float(np.mean(w.price.values / np.clip(f, 0.2, None)))


# ------------------------------------------------------------------ walk-forward predictions

def predictions(use_spread=False, use_lag=True, use_temp=False, cache={}) -> pd.DataFrame:
    """One row per (decision, j, target_day) with fair / face / naive / level and the outcome."""
    k = (use_spread, use_lag, use_temp)
    if k in cache:
        return cache[k]
    d = daily_inputs()
    wf, wa = forecast_features("gb_wind_fc", "wind")
    tf, _ = forecast_features("gb_temp_fc", "temp")
    temp_fc = tf.mean(axis=1)
    wind = WindModel(wf, wa, d.cf.dropna())
    skill = SkillModel(use_spread=use_spread, use_lag=use_lag)
    decisions = sorted(set(wf.index.get_level_values("decision")))
    decisions = [x for x in decisions if x >= pd.Timestamp(WALK_FORWARD_START)]
    rows = []
    for b in range(0, len(decisions), REFIT_EVERY_DAYS):
        block = decisions[b:b + REFIT_EVERY_DAYS]
        t0 = block[0]                       # models only see target days before the first decision
        P = add_lagged(wind.fit(t0).predict())
        skill.fit(P, d.cf, t0)
        price = PriceModel(use_temp).fit(d, t0)
        clim = d.cf[(d.index < t0) & (d.index >= t0 - pd.Timedelta(days=365))]
        sub = P[P.index.get_level_values("decision").isin(block)]
        sk = skill.predict(sub)
        idx = sub.index.to_frame(index=False)
        h = hdd(temp_fc.reindex(sub.index).values)
        h = np.where(np.isnan(h), 0, h)
        dow = idx.target_day.dt.dayofweek.values
        L = idx.decision.map({t: price.level(d, t) for t in block}).values
        naive7 = idx.decision.map({t: d.price[(d.index >= t - pd.Timedelta(days=7)) & (d.index < t)].mean() for t in block}).values
        back = np.where(idx.j < 7, 7, 14)
        naive_day = d.price.reindex(idx.target_day - pd.to_timedelta(back, unit="D")).values
        rows.append(pd.DataFrame({
            "decision": idx.decision, "j": idx.j, "target_day": idx.target_day,
            "F": sub.F.values, "S": sub.S.values, "cf_mu": sk.mu.values, "cf_sd": sk.sd.values,
            "fair": L * price.ratio(np.clip(sk.mu.values, 0, 1), sk.sd.values, h, dow),
            "face": L * price.ratio(sub.F.values, np.zeros(len(sub)), h, dow),
            "level": L * price.ratio(np.full(len(sub), clim.mean()), np.full(len(sub), clim.std()), h, dow),
            "naive7": naive7, "naive_day": naive_day, "L": L,
        }))
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(d[["price", "cf"]], left_on="target_day", right_index=True, how="left")
    cache[k] = out
    return out


def product_table(pred: pd.DataFrame, product: str) -> pd.DataFrame:
    dow, js, rule, _ = PRODUCTS[product]
    s = pred[pred.j.isin(js)]
    if dow is not None:
        s = s[s.decision.dt.dayofweek == dow]
    g = s.groupby("decision").agg(n=("j", "size"), price=("price", "mean"), fair=("fair", "mean"),
                                  face=("face", "mean"), level=("level", "mean"), naive7=("naive7", "first"),
                                  naive_day=("naive_day", "mean"), F=("F", "mean"), cf=("cf", "mean"))
    g["naive"] = g.naive7 if rule == "last7" else g.naive_day
    g["hours"] = 24 * len(js)
    return g[(g.n == len(js)) & g.price.notna() & g.naive.notna()].reset_index()


def trades(tab: pd.DataFrame, market_trust: float) -> pd.DataFrame:
    """Entry price = what a market trusting the forecast `market_trust` would quote. Trade toward fair."""
    entry = market_trust * tab.face + (1 - market_trust) * tab.naive
    edge = tab.fair - entry
    pos = np.sign(edge).where(edge.abs() >= MIN_EDGE_POWER_GBP_MWH, 0) * POWER_POSITION_MW
    mwh = tab.hours
    pnl = pos * (tab.price - entry) * mwh - pos.abs() * COST_POWER_GBP_MWH * mwh
    return pd.DataFrame({"date": tab.decision, "position": pos, "entry": entry, "fair": tab.fair,
                         "outcome": tab.price, "pnl": pnl})


def accuracy(tab: pd.DataFrame) -> dict:
    return {c: {"mae": round(float((tab.price - tab[c]).abs().mean()), 2),
                "rmse": round(float(np.sqrt(((tab.price - tab[c]) ** 2).mean())), 2)} for c in ("fair", "face", "naive", "level")}


def run() -> Result:
    split = SPLITS[KEY]
    s = pd.Timestamp(split)
    variants = {f"spread={a}, lag={b}, temp={c}": dict(use_spread=a, use_lag=b, use_temp=c)
                for a in (False, True) for b in (False, True) for c in (False, True)}
    preds = {}
    for name, kw in variants.items():
        preds[name] = predictions(**kw)
        print(f"    {KEY}: {name}", flush=True)

    res = Result(
        key=KEY, name="UK power: week-ahead weather fade", asset="GB baseload power", currency="£", unit="1 MW",
        summary="Bet that the market over-trusts 5-7 day wind forecasts: price next week from the forecast "
                "pulled back toward normal by how reliable it has been, and trade any gap to the market price.",
        how_it_works=[
            "Every Sunday at 10pm, take the European, US and German weather models' wind forecasts for 8 GB wind-farm areas for each of the next 7 days (only forecasts already published by then).",
            "Turn wind speeds into a GB wind output forecast with a learned power curve (checked against Elexon metered wind).",
            "Measure how reliable forecasts have been at each lead time: 1 day ahead is almost perfect, 7 days ahead needs pulling ~40% back toward normal.",
            "Convert the pulled-back wind forecast into a price with a model of how GB prices react to wind and weekday (the 'fair' price).",
            "Compare with the market price. If the market trusts the raw forecast, fair differs from it whenever the forecast is extreme: that difference is the fade.",
            f"Trade 1 MW if the gap is at least £{MIN_EDGE_POWER_GBP_MWH:g}/MWh; settle against the actual average price of the delivery days; £{COST_POWER_GBP_MWH:g}/MWh cost.",
        ],
        data=["Elexon market index price and metered wind (2022-2026)",
              "Open-Meteo forecast archive: ECMWF, GFS and ICON, 100 m wind at 8 sites, forecasts 1-7 days ahead (Mar 2024 on)",
              "Open-Meteo ERA5 observed temperatures (4 GB cities)"],
        caveats=["Real week-ahead quotes are not free, so the market price is simulated (0-100% trust in the forecast). The real market is somewhere in that range - the results show what each case would earn.",
                 "Decisions are on Sunday night because the free forecast archive stops at 7 days; the real week-ahead contract mostly trades Monday-Friday of the week before.",
                 "About 2.5 years of forecast history: the holdout year is ~55 weeks, so results are noisy.",
                 "Settles against the APX market index, a stand-in for the official week-ahead settlement."],
        split=split)

    products = {}
    for prod in PRODUCTS:
        acc = {}
        for name, p in preds.items():
            tab = product_table(p, prod)
            acc[name] = {"selection": accuracy(tab[tab.decision < s]), "holdout": accuracy(tab[tab.decision >= s])}
        best = min(acc, key=lambda k: acc[k]["selection"]["fair"]["rmse"])
        tab = product_table(preds[best], prod)
        per_year = 52 if PRODUCTS[prod][0] is not None else 365
        scen = {f"{m:.0%}": split_stats(trades(tab, m), split, per_year) for m in MARKETS}
        products[prod] = {"description": PRODUCTS[prod][3], "chosen_model": best, "accuracy": acc[best],
                          "accuracy_all_variants": acc, "scenarios": scen,
                          "curves": {f"market trusts forecast {m:.0%}": equity(trades(tab, m)) for m in (0.0, 0.5, 1.0)},
                          "recent": recent(trades(tab, 1.0), ["date", "position", "entry", "fair", "outcome", "pnl"])}
    main = products["week_ahead"]
    res.chosen = main["chosen_model"]
    res.stats = {p: v["scenarios"] for p, v in products.items()}
    res.curve = main["curves"]
    skill_rows = []
    best_pred = preds[main["chosen_model"]].dropna(subset=["cf"])
    for j, g in best_pred.groupby("j"):
        skill_rows.append({"days_ahead": int(j), "forecast_vs_actual_corr": round(float(g[["F", "cf"]].corr().iloc[0, 1]), 3),
                           "trust_factor": round(float(np.polyfit(g.F, g.cf, 1)[0]), 2),
                           "error_raw": round(float((g.cf - g.F).abs().mean()), 3),
                           "error_pulled_back": round(float((g.cf - g.cf_mu).abs().mean()), 3)})
    res.extra = {"products": {k: {kk: vv for kk, vv in v.items() if kk not in ("curves",)} for k, v in products.items()},
                 "skill": skill_rows, "headline_scenario": "week_ahead"}
    res.recent_trades = main["recent"]
    return res.save()
