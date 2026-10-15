"""NBP GAS - WEEK-AHEAD WEATHER FADE

Idea
    GB gas demand rises when it is cold (heating, in GB and on the Continent, which pulls on NBP via
    the Dutch TTF hub) and when it is calm (gas power stations fill in for missing wind). Traders price
    next week's gas off the weather forecast. Forecasts 5-7 days out are only partly right, so a market
    that takes them at face value over-prices a forecast cold/calm week and under-prices a mild/windy one.
    We price next week with the forecast pulled back toward normal and trade the gap.

How it is built (refit every 4 weeks, only on data available at the time)
    1. Price model (fit on 2018+ actual weather): the % change of a day's gas price versus the last
       7 days' average = a*(change in GB heating degree days) + b*(change in Continental HDD)
                        + c*(change in GB wind capacity factor) + weekend effect.
    2. Skill models, per day-ahead j: actual GB HDD, Continental HDD and wind CF vs their forecasts.
    3. fair = price from pulled-back forecasts; face = price from raw forecasts; naive = last 7 days.

Gas prices also move on news that has nothing to do with weather (LNG cargoes, Norwegian outages,
European storage), so the weather effect is a smaller share of gas price moves than of power moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (COST_GAS_P_THERM, GAS_POSITION_THERMS_PER_DAY, MIN_EDGE_GAS_P_THERM, RAW,
                    REFIT_EVERY_DAYS, SPLITS, WALK_FORWARD_START)
from lib.backtest import Result, equity, recent, split_stats
from lib.weather import SkillModel, WindModel, add_lagged, forecast_features, hdd, wind_capacity_factor

KEY = "nbp_week_ahead"
MARKETS = [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOW = 7
PRODUCTS = {
    "week_ahead": (6, tuple(range(1, 8)), "Sunday night: next Mon-Sun gas"),
    "day_plus6": (None, (6,), "Every night: the single gas day 6 days out"),
}


def daily_inputs() -> pd.DataFrame:
    sap = pd.read_pickle(RAW / "gas_sap.pkl").sap
    gb = pd.read_pickle(RAW / "gb_temp_actual.pkl").groupby("day").temp.mean()
    eu = pd.read_pickle(RAW / "eu_temp_actual.pkl").groupby("day").temp.mean()
    d = pd.concat([sap.rename("sap"), hdd(gb).rename("hdd_gb"), hdd(eu).rename("hdd_eu"),
                   wind_capacity_factor()], axis=1)
    d = d[d.index >= "2018-03-01"]
    d["wkend"] = (d.index.dayofweek >= 5).astype(float)
    return d


def window_means(d: pd.DataFrame) -> pd.DataFrame:
    """Means over the WINDOW full days before each day (what is known at that day's decision)."""
    return d.rolling(WINDOW, min_periods=5).mean().shift(1)


class GasPriceModel:
    def __init__(self, use_eu: bool, use_wind: bool, since: str):
        self.use_eu, self.use_wind, self.since = use_eu, use_wind, since

    def _X(self, dgb, deu, dcf, dwk):
        cols = [dgb] + ([deu] if self.use_eu else []) + ([dcf] if self.use_wind else []) + [dwk]
        return np.column_stack(cols)

    def fit(self, d: pd.DataFrame, cutoff):
        w = window_means(d)
        rows = []
        for j in range(1, 8):
            tgt = d.shift(-j)
            frame = pd.DataFrame({
                "y": np.log(tgt.sap / w.sap),
                "dgb": tgt.hdd_gb - w.hdd_gb, "deu": tgt.hdd_eu - w.hdd_eu,
                "dcf": tgt.cf - w.cf, "dwk": tgt.wkend - w.wkend,
                "target": d.index + pd.Timedelta(days=j)}, index=d.index)
            rows.append(frame)
        f = pd.concat(rows)
        f = f[(f.index >= self.since) & (f.target < cutoff)].replace([np.inf, -np.inf], np.nan).dropna()
        X = self._X(f.dgb, f.deu, f.dcf, f.dwk)
        self.beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], f.y.values, rcond=None)[0]
        return self

    def predict(self, level, dgb, deu, dcf, dwk):
        X = self._X(dgb, deu, dcf, dwk)
        return level * np.exp(np.c_[np.ones(len(X)), X] @ self.beta)


def _fc_frame(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"F": series, "S": 0.0})


def predictions(use_eu=True, use_wind=True, since="2018-03-01", cache={}) -> pd.DataFrame:
    k = (use_eu, use_wind, since)
    if k in cache:
        return cache[k]
    d = daily_inputs()
    w = window_means(d)
    wf, wa = forecast_features("gb_wind_fc", "wind")
    gb_fc = hdd(forecast_features("gb_temp_fc", "temp")[0].mean(axis=1))
    eu_fc = hdd(forecast_features("eu_temp_fc", "temp")[0].mean(axis=1))
    wind = WindModel(wf, wa, d.cf.dropna())
    sk_gb, sk_eu, sk_cf = (SkillModel(use_lag=True) for _ in range(3))
    decisions = sorted(set(wf.index.get_level_values("decision")))
    decisions = [x for x in decisions if x >= pd.Timestamp(WALK_FORWARD_START)]
    out = []
    for b in range(0, len(decisions), REFIT_EVERY_DAYS):
        block = decisions[b:b + REFIT_EVERY_DAYS]
        t0 = block[0]
        model = GasPriceModel(use_eu, use_wind, since).fit(d, t0)
        cf_fc = add_lagged(wind.fit(t0).predict())
        g_fc, e_fc = add_lagged(_fc_frame(gb_fc)), add_lagged(_fc_frame(eu_fc))
        sk_gb.fit(g_fc, d.hdd_gb, t0); sk_eu.fit(e_fc, d.hdd_eu, t0); sk_cf.fit(cf_fc, d.cf, t0)
        idx = cf_fc.index[cf_fc.index.get_level_values("decision").isin(block)]
        idx = idx.intersection(g_fc.index).intersection(e_fc.index)
        fr = idx.to_frame(index=False)
        win = w.reindex(fr.decision + pd.Timedelta(days=1))           # window ending the day before decision+1
        win.index = fr.index
        level = win.sap.values
        raw = {"gb": g_fc.F.reindex(idx).values, "eu": e_fc.F.reindex(idx).values, "cf": cf_fc.F.reindex(idx).values}
        shrunk = {"gb": np.clip(sk_gb.predict(g_fc.reindex(idx)).mu.values, 0, None),
                  "eu": np.clip(sk_eu.predict(e_fc.reindex(idx)).mu.values, 0, None),
                  "cf": np.clip(sk_cf.predict(cf_fc.reindex(idx)).mu.values, 0, 1)}
        wk = (fr.target_day.dt.dayofweek >= 5).astype(float).values - win.wkend.values
        price = lambda v: model.predict(level, v["gb"] - win.hdd_gb.values, v["eu"] - win.hdd_eu.values,
                                        v["cf"] - win.cf.values, wk)
        out.append(pd.DataFrame({"decision": fr.decision, "j": fr.j, "target_day": fr.target_day,
                                 "fair": price(shrunk), "face": price(raw), "naive": level,
                                 "hdd_gb_fc": raw["gb"], "hdd_eu_fc": raw["eu"], "cf_fc": raw["cf"]}))
    res = pd.concat(out, ignore_index=True).merge(d[["sap", "hdd_gb", "hdd_eu", "cf"]], left_on="target_day",
                                                  right_index=True, how="left")
    cache[k] = res
    return res


def product_table(pred: pd.DataFrame, product: str) -> pd.DataFrame:
    dow, js, _ = PRODUCTS[product]
    s = pred[pred.j.isin(js)]
    if dow is not None:
        s = s[s.decision.dt.dayofweek == dow]
    g = s.groupby("decision").agg(n=("j", "size"), price=("sap", "mean"), fair=("fair", "mean"),
                                  face=("face", "mean"), naive=("naive", "first"))
    g["days"] = len(js)
    return g[(g.n == len(js)) & g.price.notna() & g.naive.notna() & g.fair.notna()].reset_index()


def trades(tab: pd.DataFrame, market_trust: float) -> pd.DataFrame:
    entry = market_trust * tab.face + (1 - market_trust) * tab.naive
    edge = tab.fair - entry
    pos = np.sign(edge).where(edge.abs() >= MIN_EDGE_GAS_P_THERM, 0)
    therms = GAS_POSITION_THERMS_PER_DAY * tab.days
    pnl = (pos * (tab.price - entry) - pos.abs() * COST_GAS_P_THERM) * therms / 100   # pence -> £
    return pd.DataFrame({"date": tab.decision, "position": pos, "entry": entry, "fair": tab.fair,
                         "outcome": tab.price, "pnl": pnl})


def accuracy(tab):
    return {c: {"mae": round(float((tab.price - tab[c]).abs().mean()), 2),
                "rmse": round(float(np.sqrt(((tab.price - tab[c]) ** 2).mean())), 2)} for c in ("fair", "face", "naive")}


def run() -> Result:
    split = SPLITS[KEY]
    s = pd.Timestamp(split)
    variants = {f"EU weather={a}, wind={b}, history from {c[:4]}": dict(use_eu=a, use_wind=b, since=c)
                for a in (False, True) for b in (False, True) for c in ("2018-03-01", "2023-04-01")}
    preds = {}
    for name, kw in variants.items():
        preds[name] = predictions(**kw)
        print(f"    {KEY}: {name}", flush=True)
    res = Result(
        key=KEY, name="NBP gas: week-ahead weather fade", asset="NBP gas (GB)", currency="£",
        unit=f"{GAS_POSITION_THERMS_PER_DAY:,} therms/day",
        summary="Same idea as the power fade, for gas: forecast cold or calm weeks push gas prices up; we "
                "price next week from the forecast pulled back toward normal and trade any gap.",
        how_it_works=[
            "Every Sunday at 10pm, take 1-7 day temperature forecasts for 4 GB cities and 4 Continental cities (Berlin, Paris, Amsterdam, Milan) and the GB wind forecast.",
            "Turn them into heating degree days (how far below 15.5°C each day is) and a GB wind capacity factor.",
            "A price model learned on 2018-2026 history says how gas prices move when next week is colder / calmer than the last 7 days.",
            "Pull each forecast back toward normal by how reliable it has been at that lead time, and price next week with it ('fair').",
            "Compare with the market price; trade if the gap is at least "
            f"{MIN_EDGE_GAS_P_THERM:g}p/therm; settle against the actual average System Average Price (SAP) of the delivery days.",
            f"Size: {GAS_POSITION_THERMS_PER_DAY:,} therms per day of delivery (25 standard lots), {COST_GAS_P_THERM:g}p/therm cost.",
        ],
        data=["ONS / National Gas daily System Average Price of gas (2018-2026)",
              "Open-Meteo forecast archive: temperatures (8 cities) and 100 m wind (8 GB sites), Mar 2024 on",
              "Open-Meteo ERA5 observed temperatures and Elexon metered wind for fitting the price model"],
        caveats=["Market price is simulated (0-100% trust in the forecast), as real week-ahead NBP quotes are not free.",
                 "SAP is the National Gas balancing-market average, close to but not the same as NBP day-ahead.",
                 "Gas prices react strongly to non-weather news (LNG, Norway, storage, geopolitics), which this ignores.",
                 "About 2.5 years of forecast history: ~55 holdout weeks."],
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
        products[prod] = {"description": PRODUCTS[prod][2], "chosen_model": best, "accuracy": acc[best],
                          "accuracy_all_variants": acc,
                          "scenarios": {f"{m:.0%}": split_stats(trades(tab, m), split, per_year) for m in MARKETS},
                          "curves": {f"market trusts forecast {m:.0%}": equity(trades(tab, m)) for m in (0.0, 0.5, 1.0)},
                          "recent": recent(trades(tab, 1.0), ["date", "position", "entry", "fair", "outcome", "pnl"])}
    main = products["week_ahead"]
    res.chosen = main["chosen_model"]
    res.stats = {p: v["scenarios"] for p, v in products.items()}
    res.curve = main["curves"]
    d = daily_inputs()
    m = GasPriceModel(True, True, "2018-03-01").fit(d, pd.Timestamp.now())
    res.extra = {"products": {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in products.items()},
                 "price_sensitivity": {"per_extra_GB_HDD_pct": round(float(m.beta[1] * 100), 2),
                                       "per_extra_EU_HDD_pct": round(float(m.beta[2] * 100), 2),
                                       "per_+0.1_wind_CF_pct": round(float(m.beta[3] * 10), 2),
                                       "weekend_pct": round(float(m.beta[4] * 100), 2)},
                 "headline_scenario": "week_ahead"}
    res.recent_trades = main["recent"]
    return res.save()
