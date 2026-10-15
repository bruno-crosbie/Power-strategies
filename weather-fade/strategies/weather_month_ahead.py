"""POWER AND GAS - MONTH-AHEAD AND QUARTER-AHEAD WEATHER FADE

Idea
    Nobody can forecast the weather a month ahead; beyond about two weeks the best guess is "normal for
    the time of year". But spot prices carry the weather of the last few weeks: a calm, cold spell makes
    spot expensive, a windy, mild spell makes it cheap. A forward market that anchors next month's price
    on recent spot is implicitly assuming that weather carries on. The fade bets it won't: price next
    month from the weather-neutral price level and *normal* weather for that month, and trade against a
    price anchored on recent spot.

How it is built (refit at every decision, only on data before it)
    Decision: the 24th of each month, for the next calendar month (quarter-ahead: the 24th of Mar, Jun,
    Sep and Dec, for the next quarter).
    Power: the same price model as the week-ahead fade (price / weather-neutral level = f(wind, cold,
           weekday)), fitted on the previous 3 years. fair = level x average of f over next month's days,
           using each day's normal wind (that month's history) and normal temperature (that date's history).
    Gas:   the week-ahead gas model's sensitivities (how gas moves with GB and Continental cold and GB wind),
           applied to the gap between next month's normal weather and the last 7 days' actual weather.
    Anchor: the average spot price over the last 14 or 28 days - what a spot-anchored market would quote.
    The only setting we choose (on the tuning period, by accuracy) is how many days the level uses (power)
    or how much history fits the gas model; the anchor window and strength are market scenarios.

The catch (again)
    Real month-ahead quotes are not free, so we can't see how much the market anchors on spot. We test
    markets that anchor 50% or 100% on the last 14 or 28 days. The accuracy table (fair vs spot anchor at predicting next month) is the
    most direct evidence either way. P&L is in % on a fixed £100k position so the 2021-22 crisis does not
    swamp everything else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import COST_GAS_P_THERM, COST_POWER_GBP_MWH, RAW, TZ
from lib.backtest import Result, equity, recent, split_stats
from lib.weather import hdd, wind_capacity_factor
from strategies.nbp_week_ahead import GasPriceModel, daily_inputs as gas_inputs, window_means
from strategies.power_week_ahead import PriceModel

KEY = "weather_month_ahead"
DECISION_DAY = 24
SPLIT = "2023-01-01"
NOTIONAL = 100_000
MIN_EDGE_PCT = 0.02
ANCHORS = [0.5, 1.0]                       # how much the market anchors on recent spot...
ANCHOR_WINDOWS = [14, 28]                  # ...and over how many days (both are market scenarios, not our settings)
PRODUCTS = {
    "power_month": ("power", "M", "UK baseload power, next calendar month"),
    "power_quarter": ("power", "Q", "UK baseload power, next quarter"),
    "gas_month": ("gas", "M", "NBP gas (SAP), next calendar month"),
    "gas_quarter": ("gas", "Q", "NBP gas (SAP), next quarter"),
}


# ------------------------------------------------------------------ data and climatology

def power_inputs() -> pd.DataFrame:
    p = pd.read_pickle(RAW / "elexon_prices.pkl")
    p["day"] = pd.to_datetime(p.start.dt.tz_convert(TZ).dt.date)
    price = p.groupby("day").price.agg(["mean", "size"])
    price = price[price["size"] >= 40]["mean"].rename("price")
    temp = pd.read_pickle(RAW / "gb_temp_actual.pkl").groupby("day").temp.mean()
    d = pd.concat([price, wind_capacity_factor(), hdd(temp).rename("hdd")], axis=1)
    d["dow"] = d.index.dayofweek
    return d[d.index >= "2018-03-01"]


def climatology(series: pd.Series, before: pd.Timestamp, days: pd.DatetimeIndex, by: str):
    """Normal value for each target day from history before `before`: by calendar month ('month')
    or by day of year +/- 7 days ('doy'). Returns (mean, sd) arrays."""
    h = series[series.index < before].dropna()
    if by == "month":
        g = h.groupby(h.index.month)
        mu, sd = g.mean(), g.std()
        return mu.reindex(days.month).values, sd.reindex(days.month).values
    doy = h.index.dayofyear.values
    vals = h.values
    mu = np.array([vals[np.minimum(np.abs(doy - d), 366 - np.abs(doy - d)) <= 7].mean() for d in days.dayofyear])
    return mu, np.zeros(len(days))


def periods(freq: str, start="2018-12-01", end=None):
    end = end or pd.Timestamp.now().normalize()
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        if freq == "Q" and m.month not in (1, 4, 7, 10):
            continue
        t = (m - pd.offsets.MonthBegin(1)).replace(day=DECISION_DAY)      # 24th of the month before
        last = m + (pd.offsets.QuarterEnd(0) if freq == "Q" else pd.offsets.MonthEnd(0))
        if last >= end:
            continue
        out.append((t, pd.date_range(m, last, freq="D")))
    return out


# ------------------------------------------------------------------ predictions

def power_rows(freq: str, level_days: int):
    d = power_inputs()
    rows = []
    for t, days in periods(freq):
        model = PriceModel(use_temp=True).fit(d, t, since=str((t - pd.DateOffset(years=3)).date()))
        L = model.level(d, t, level_days)
        cf_mu, cf_sd = climatology(d.cf, t, days, "month")
        h_mu, _ = climatology(d.hdd, t, days, "doy")
        fair = L * float(np.mean(model.ratio(cf_mu, cf_sd, h_mu, days.dayofweek.values)))
        rows.append({"decision": t, "start": days[0], "fair": fair, "unit": "£/MWh",
                     "spot14": d.price[(d.index >= t - pd.Timedelta(days=14)) & (d.index < t)].mean(),
                     "spot28": d.price[(d.index >= t - pd.Timedelta(days=28)) & (d.index < t)].mean(),
                     "outcome": d.price.reindex(days).mean(), "days_with_price": int(d.price.reindex(days).notna().sum()),
                     "n_days": len(days), "cost": COST_POWER_GBP_MWH})
    return pd.DataFrame(rows)


def gas_rows(freq: str, since: str):
    d = gas_inputs()
    w = window_means(d)
    rows = []
    for t, days in periods(freq):
        m = GasPriceModel(use_eu=True, use_wind=True, since=since).fit(d, t)
        win = w.loc[:t].iloc[-1]                                       # last 7 full days before t
        g_mu, _ = climatology(d.hdd_gb, t, days, "doy")
        e_mu, _ = climatology(d.hdd_eu, t, days, "doy")
        c_mu, _ = climatology(d.cf, t, days, "month")
        wk = (days.dayofweek >= 5).astype(float) - win.wkend
        fair = float(np.mean(m.predict(win.sap, g_mu - win.hdd_gb, e_mu - win.hdd_eu, c_mu - win.cf, wk)))
        rows.append({"decision": t, "start": days[0], "fair": fair, "unit": "p/therm",
                     "spot14": d.sap[(d.index >= t - pd.Timedelta(days=14)) & (d.index < t)].mean(),
                     "spot28": d.sap[(d.index >= t - pd.Timedelta(days=28)) & (d.index < t)].mean(),
                     "outcome": d.sap.reindex(days).mean(), "days_with_price": int(d.sap.reindex(days).notna().sum()),
                     "n_days": len(days), "cost": COST_GAS_P_THERM})
    return pd.DataFrame(rows)


def table(product: str, setting: dict) -> pd.DataFrame:
    market, freq, _ = PRODUCTS[product]
    t = power_rows(freq, setting["level_days"]) if market == "power" else gas_rows(freq, setting["since"])
    return t[(t.days_with_price >= 0.9 * t.n_days)].dropna(subset=["fair", "spot14", "spot28", "outcome"]).reset_index(drop=True)


def trades(t: pd.DataFrame, anchoring: float, window: int = 14) -> pd.DataFrame:
    entry = anchoring * t[f"spot{window}"] + (1 - anchoring) * t.fair
    edge = t.fair / entry - 1
    pos = np.sign(edge).where(edge.abs() >= MIN_EDGE_PCT, 0)
    pnl = (pos * (t.outcome / entry - 1) - pos.abs() * t.cost / entry) * NOTIONAL
    return pd.DataFrame({"date": t.decision, "position": pos, "entry": entry, "fair": t.fair,
                         "outcome": t.outcome, "pnl": pnl})


def accuracy(t: pd.DataFrame) -> dict:
    """Typical % miss of each predictor (root-mean-square of log errors)."""
    out = {}
    for c in ("fair", "spot14", "spot28"):
        e = np.log(t.outcome / t[c])
        out[c] = {"rmse_pct": round(float(np.sqrt((e ** 2).mean()) * 100), 1), "bias_pct": round(float(e.mean() * 100), 1),
                  "n": int(len(t))}
    # how often is fair closer to what happened than the 14-day spot anchor?
    out["fair_closer_share"] = round(float((np.abs(np.log(t.outcome / t.fair)) < np.abs(np.log(t.outcome / t.spot14))).mean()), 3)
    return out


def settings_for(product):
    market = PRODUCTS[product][0]
    if market == "power":
        return [{"level_days": l} for l in (14, 28)]
    return [{"since": s} for s in ("2018-03-01", "2023-04-01")]


def label(s):
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in s.items())


def run() -> Result:
    split = SPLIT
    s = pd.Timestamp(split)
    res = Result(
        key=KEY, name="Power & gas: month/quarter-ahead weather fade", asset="UK power and NBP gas", currency="£",
        unit="£100k position per trade",
        summary="Bet that recent unusual weather does not carry into next month: price the next month or "
                "quarter from the weather-neutral level and normal weather, and trade against a price "
                "anchored on recent spot.",
        how_it_works=[
            "On the 24th of each month, look at the last 2-4 weeks of spot prices and the weather behind them.",
            "Strip the weather out to get a weather-neutral price level (power: wind/cold/weekday model; gas: cold in GB and the Continent, and GB wind).",
            "Price next month (or quarter) at that level with normal weather for the time of year: normal wind for that month and normal temperature for each date, from history only.",
            "A market anchored on recent spot quotes something closer to recent spot. If our fair price differs by 2% or more, trade toward fair.",
            "Settle against the actual average spot price over the delivery month or quarter.",
            "Size: £100k of power or gas at the entry price; costs £1/MWh (power) or 0.5p/therm (gas).",
        ],
        data=["Elexon daily average power price 2018-2026", "ONS / National Gas daily SAP 2018-2026",
              "Elexon metered wind and ERA5 observed temperatures (normal weather from history before each decision)"],
        caveats=["Real month/quarter-ahead quotes are not free: the market price is simulated as anchored 50% or 100% on the last 14 or 28 days of spot.",
                 "Monthly decisions mean few data points: ~49 months tuning (Dec 2018-Dec 2022, including the 2021-22 crisis) and ~45 months test; quarters have a quarter as many.",
                 "Fuel-price trends (e.g. gas in 2021-22) move monthly prices far more than weather does, and this strategy has no view on them.",
                 "Settles against spot averages (daily APX index / SAP), a stand-in for the real contracts' settlement."],
        split=split)
    products, curves = {}, {}
    for prod in PRODUCTS:
        tabs = {label(st): table(prod, st) for st in settings_for(prod)}
        best = min(tabs, key=lambda k: accuracy(tabs[k][tabs[k].decision < s])["fair"]["rmse_pct"])
        t = tabs[best]
        per_year = 12 if PRODUCTS[prod][1] == "M" else 4
        scen = {f"{a:.0%} on last {w} days": split_stats(trades(t, a, w), split, per_year) for w in ANCHOR_WINDOWS for a in ANCHORS}
        products[prod] = {"description": PRODUCTS[prod][2], "chosen_model": best,
                          "accuracy": {"selection": accuracy(t[t.decision < s]), "holdout": accuracy(t[t.decision >= s])},
                          "scenarios": scen,
                          "recent": recent(trades(t, 1.0, 14), ["date", "position", "entry", "fair", "outcome", "pnl"], 12)}
        if prod in ("power_month", "gas_month"):
            curves[f"{prod.replace('_', ' ')} (market fully anchored, 14 days)"] = equity(trades(t, 1.0, 14), "ME")
        print(f"    {KEY}: {prod} ({best})", flush=True)
    res.chosen = "; ".join(f"{k}: {v['chosen_model']}" for k, v in products.items())
    res.stats = {p: v["scenarios"] for p, v in products.items()}
    res.curve = curves
    res.extra = {"products": products, "headline_scenario": "power_month"}
    res.recent_trades = products["power_month"]["recent"]
    return res.save()
