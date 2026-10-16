"""Downloaders for every data source. Each function saves a tidy pandas file into data/raw and
returns it. All sources are free and need no account or API key.

  Elexon          UK power prices (MID) and metered wind output        data.elexon.co.uk
  Open-Meteo      archived weather forecasts by lead time              previous-runs-api.open-meteo.com
  ONS             daily GB gas price (National Gas SAP)                ons.gov.uk
  ENTSOG          daily Norwegian pipeline gas flows into Europe/GB    transparency.entsog.eu
  FRED            daily Brent and WTI crude prices                     fred.stlouisfed.org
  EIA             daily WTI futures, 1st and 2nd month (ends 2024)     eia.gov
"""
from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from config import (FORECAST_ARCHIVE_START, FORECAST_MODELS, KWH_PER_THERM, MAX_LEAD_DAYS, PARTS, RAW)

session = requests.Session()


def get(url, params=None, retries=5, timeout=120, as_json=True):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json() if as_json else r
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} {params}")


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_pickle(RAW / f"{name}.pkl")
    return df


def load(name: str) -> pd.DataFrame:
    return pd.read_pickle(RAW / f"{name}.pkl")


# ===================================================================== Elexon (UK power)
ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"


def _weeks(start, end):
    t = pd.Timestamp(start)
    while t < end:
        yield t, min(t + pd.Timedelta(days=7), end)
        t += pd.Timedelta(days=7)


def elexon_prices(start="2018-01-01") -> pd.DataFrame:
    """Half-hourly APX market index price (£/MWh)."""
    end = pd.Timestamp.now(tz="UTC")
    def one(w):
        return get(f"{ELEXON}/datasets/MID", {"from": w[0].strftime("%Y-%m-%dT%H:%MZ"),
                                             "to": w[1].strftime("%Y-%m-%dT%H:%MZ"), "dataProviders": "APXMIDP"})["data"]
    with ThreadPoolExecutor(8) as ex:
        rows = sum(ex.map(one, list(_weeks(pd.Timestamp(start, tz="UTC"), end))), [])
    df = pd.DataFrame(rows)
    df = df[(df.dataProvider == "APXMIDP") & (df.volume > 0)]
    df = pd.DataFrame({"start": pd.to_datetime(df.startTime, utc=True), "price": df.price})
    return save(df.drop_duplicates("start").sort_values("start"), "elexon_prices")


def elexon_wind(start="2018-01-01") -> pd.DataFrame:
    """Half-hourly metered GB wind output (MW)."""
    end = pd.Timestamp.now(tz="UTC")
    def one(w):
        return get(f"{ELEXON}/datasets/FUELHH", {"settlementDateFrom": str(w[0].date()),
                                                "settlementDateTo": str((w[0] + pd.Timedelta(days=6)).date()),
                                                "fuelType": "WIND"})["data"]
    with ThreadPoolExecutor(8) as ex:
        rows = sum(ex.map(one, list(_weeks(pd.Timestamp(start, tz="UTC"), end))), [])
    df = pd.DataFrame(rows)
    df = pd.DataFrame({"start": pd.to_datetime(df.startTime, utc=True), "wind_mw": df.generation})
    return save(df.drop_duplicates("start").sort_values("start"), "elexon_wind")


# ===================================================================== Open-Meteo (forecast archive)

def _open_meteo(lat, lon, var, model, start, end) -> pd.DataFrame:
    """Hourly values of `var` as forecast 0..7 days before each hour, long format."""
    hourly = ",".join([var] + [f"{var}_previous_day{k}" for k in range(1, MAX_LEAD_DAYS + 1)])
    params = {"latitude": lat, "longitude": lon, "hourly": hourly, "models": model, "start_date": start,
              "end_date": end, "timezone": "GMT", "wind_speed_unit": "ms"}
    for attempt in range(6):
        try:
            h = get("https://previous-runs-api.open-meteo.com/v1/forecast", params, retries=2, timeout=180)["hourly"]
            break
        except (RuntimeError, KeyError):
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"open-meteo failed {lat},{lon} {var} {model} {start}")
    df = pd.DataFrame(h).rename(columns={var: f"{var}_previous_day0"})
    df["time"] = pd.to_datetime(df.time, utc=True)
    long = df.melt(id_vars="time", var_name="lead", value_name="value")
    long["lead"] = long.lead.str.extract(r"previous_day(\d)").astype(int)
    return long


def forecast_archive(sites: dict, var: str, name: str, models=FORECAST_MODELS) -> pd.DataFrame:
    """Download in 90-day chunks; completed chunks are cached in data/parts so reruns only fetch new days."""
    end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    edges = list(pd.date_range(FORECAST_ARCHIVE_START, end, freq="90D").strftime("%Y-%m-%d")) + [end]
    frames = []
    for model in models:
        for site, (lat, lon) in sites.items():
            for a, b in zip(edges[:-1], edges[1:]):
                last_chunk = b == end
                b_incl = end if last_chunk else (pd.Timestamp(b) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                f = PARTS / f"{name}_{model}_{site}_{a}.pkl"
                if f.exists() and not last_chunk:
                    df = pd.read_pickle(f)
                else:
                    df = _open_meteo(lat, lon, var, model, a, b_incl)
                    df["site"], df["model"] = site, model
                    df.to_pickle(f)
                frames.append(df)
            print(f"    {name}: {model} {site}", flush=True)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    return save(out.drop_duplicates(["time", "lead", "site", "model"]), name)


def actual_temperatures(cities: dict, name: str, start="2018-01-01") -> pd.DataFrame:
    """Observed daily mean temperature (ERA5 reanalysis) per city, topped up for the last few days
    (ERA5 runs ~5 days behind) with the weather models' own recent analysis."""
    end = (pd.Timestamp.now() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    frames = []
    for city, (lat, lon) in cities.items():
        d = get("https://archive-api.open-meteo.com/v1/archive",
                {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                 "daily": "temperature_2m_mean", "timezone": "Europe/London"})["daily"]
        obs = pd.DataFrame({"day": pd.to_datetime(d["time"]), "city": city, "temp": d["temperature_2m_mean"]}).dropna()
        r = get("https://api.open-meteo.com/v1/forecast",
                {"latitude": lat, "longitude": lon, "past_days": 14, "forecast_days": 1,
                 "daily": "temperature_2m_mean", "timezone": "Europe/London"})["daily"]
        rec = pd.DataFrame({"day": pd.to_datetime(r["time"]), "city": city, "temp": r["temperature_2m_mean"]}).dropna()
        rec = rec[(rec.day > obs.day.max()) & (rec.day < pd.Timestamp.now().normalize())]
        frames.append(pd.concat([obs, rec]))
    return save(pd.concat(frames).dropna(), name)


# ===================================================================== ONS / National Gas (GB gas price)

def gas_sap() -> pd.DataFrame:
    """Daily System Average Price of gas, converted to pence/therm. Published weekly by the ONS."""
    page = get("https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas",
               as_json=False).text
    link = re.search(r'href="(/file\?uri=[^"]+\.xlsx)"', page).group(1)
    xlsx = get("https://www.ons.gov.uk" + link, as_json=False).content
    raw = pd.read_excel(io.BytesIO(xlsx), sheet_name="1.Daily SAP Gas", header=None)
    hdr = raw.index[raw[0].astype(str).str.strip().eq("Date")][0]
    df = raw.iloc[hdr + 1:, :2]
    df.columns = ["day", "p_kwh"]
    df["day"] = pd.to_datetime(df.day, errors="coerce")
    df["p_kwh"] = pd.to_numeric(df.p_kwh, errors="coerce")
    df = df.dropna()
    df["sap"] = df.p_kwh * KWH_PER_THERM              # p/kWh -> p/therm
    return save(df.set_index("day")[["sap"]], "gas_sap")


# ===================================================================== ENTSOG (Norwegian gas flows)
NORWAY_POINTS = {   # pointKey: (label, receiving country)
    "ITP-00022": ("St Fergus", "GB"), "ITP-00091": ("Easington", "GB"),
    "ITP-00106": ("Zeebrugge", "BE"), "ITP-00045": ("Dunkerque", "FR"),
    "ITP-00525": ("Dornum", "DE"), "ITP-00630": ("North Sea Entry", "DK"),
}


def norway_flows(start="2021-11-01") -> pd.DataFrame:
    """Daily physical flow (GWh/day) of Norwegian gas arriving at each entry point."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    def one(win):
        d = get("https://transparency.entsog.eu/api/v1/operationaldatas",
                {"from": str(win[0].date()), "to": str(win[1].date()), "indicator": "Physical Flow",
                 "periodType": "day", "timezone": "UTC", "limit": 10000, "pointKey": ",".join(NORWAY_POINTS)},
                timeout=180)
        return d.get("operationaldatas", [])
    wins, t = [], pd.Timestamp(start, tz="UTC")
    while t < end:
        wins.append((t, min(t + pd.Timedelta(days=60), end)))
        t += pd.Timedelta(days=61)
    with ThreadPoolExecutor(4) as ex:
        rows = sum(ex.map(one, wins), [])
    df = pd.DataFrame(rows)
    df = df[df.directionKey == "entry"]
    df["day"] = pd.to_datetime(df.periodFrom.str[:10])
    df["gwh"] = pd.to_numeric(df.value, errors="coerce") / 1e6
    df["country"] = df.pointKey.map(lambda k: NORWAY_POINTS[k][1])
    df = df.groupby(["day", "pointKey", "country"], as_index=False).gwh.sum()
    return save(df, "norway_flows")


# ===================================================================== FRED / EIA (oil)

def fred(series: str, name: str) -> pd.DataFrame:
    txt = get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series}, as_json=False).text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["day", "price"]
    df["day"] = pd.to_datetime(df.day)
    df["price"] = pd.to_numeric(df.price, errors="coerce")
    return save(df.dropna().set_index("day"), name)


def eia_futures(code: str, name: str) -> pd.DataFrame:
    """EIA weekly-row tables ('Week Of' + Mon..Fri columns) flattened to a daily series."""
    html = get(f"https://www.eia.gov/dnav/pet/hist/{code}D.htm", as_json=False).text
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    t = max(tables, key=len).dropna(how="all")
    t = t[t.iloc[:, 0].astype(str).str.match(r"\d{4} ")]
    rows = []
    for _, r in t.iterrows():
        year, rest = str(r.iloc[0]).split(" ", 1)
        monday = pd.to_datetime(f"{year} {rest.split(' to ')[0].replace('- ', '-').strip()}", format="%Y %b-%d", errors="coerce")
        if pd.isna(monday):
            continue
        for k in range(5):
            v = pd.to_numeric(r.iloc[1 + k], errors="coerce")
            if pd.notna(v):
                rows.append((monday + pd.Timedelta(days=k), float(v)))
    df = pd.DataFrame(rows, columns=["day", "price"]).set_index("day").sort_index()
    return save(df[~df.index.duplicated()], name)


def download_all(only: set | None = None):
    from config import EU_CITIES, GB_CITIES, GB_WIND_SITES
    jobs = {
        "elexon": lambda: (elexon_prices(), elexon_wind()),
        "gb_wind_fc": lambda: forecast_archive(GB_WIND_SITES, "wind_speed_100m", "gb_wind_fc"),
        "gb_temp_fc": lambda: forecast_archive(GB_CITIES, "temperature_2m", "gb_temp_fc", models=FORECAST_MODELS[:1]),
        "eu_temp_fc": lambda: forecast_archive(EU_CITIES, "temperature_2m", "eu_temp_fc", models=FORECAST_MODELS[:1]),
        "gas": gas_sap,
        "temps": lambda: (actual_temperatures(GB_CITIES, "gb_temp_actual"), actual_temperatures(EU_CITIES, "eu_temp_actual")),
        "norway": norway_flows,
        "oil": lambda: (fred("DCOILBRENTEU", "brent"), fred("DCOILWTICO", "wti"),
                        eia_futures("RCLC1", "wti_c1"), eia_futures("RCLC2", "wti_c2")),
    }
    for name, job in jobs.items():
        if only and name not in only:
            continue
        t = time.time()
        print(f"  downloading {name} ...", flush=True)
        job()
        print(f"  {name} done in {time.time() - t:.0f}s", flush=True)
