"""Shared data access for the UK power paper-trading strategies.

All data comes from the free Elexon Insights API (no key needed).
Trade model: a 1 MW position in one half-hour settlement period, entered at the
APX market index price (MID) and settled at the system imbalance price (SIP).
P&L = position * (SIP - MID) * 0.5 MWh.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

GATE_CLOSURE = pd.Timedelta(minutes=60)   # last moment we may trade a period, before its start
PRICE_LAG = pd.Timedelta(minutes=30)      # assumed delay before a period's SIP/MID is published
COST_PER_MWH = 2.0                         # assumed round-trip trading cost for the "net" figures

_session = requests.Session()


def get(path: str, params: dict | None = None, retries: int = 4):
    for attempt in range(retries):
        try:
            r = _session.get(f"{BASE}/{path}", params=params, timeout=60)
            if r.status_code == 200:
                j = r.json()
                return j["data"] if isinstance(j, dict) else j
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} tries: {path} {params}")


def iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%MZ")


def chunks(start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta):
    t = start
    while t < end:
        yield t, min(t + step, end)
        t += step


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- fetchers

def fetch_system_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.tz_convert("Europe/London").date() - timedelta(days=1),
                         end.tz_convert("Europe/London").date(), freq="D")
    rows = sum(pmap(lambda d: get(f"balancing/settlement/system-prices/{d.date()}"), days), [])
    if not rows:
        return pd.DataFrame(columns=["start", "sip", "niv"])
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(columns={"systemSellPrice": "sip", "netImbalanceVolume": "niv"})
    df = df[["start", "sip", "niv"]].drop_duplicates("start", keep="last")
    return df[(df.start >= start) & (df.start < end)].sort_values("start")


def fetch_mid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    def one(win):
        return get("datasets/MID", {"from": iso(win[0]), "to": iso(win[1]), "dataProviders": "APXMIDP"})
    rows = sum(pmap(one, list(chunks(start, end, pd.Timedelta(days=7)))), [])
    if not rows:
        return pd.DataFrame(columns=["start", "mid", "mid_vol"])
    df = pd.DataFrame(rows)
    df = df[(df.dataProvider == "APXMIDP") & (df.volume > 0)]
    df["start"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(columns={"price": "mid", "volume": "mid_vol"})
    return df[["start", "mid", "mid_vol"]].drop_duplicates("start", keep="last").sort_values("start")


def fetch_windfor(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Every wind forecast published in [start, end): publish time, target hour, MW."""
    def one(win):
        return get("datasets/WINDFOR", {"publishDateTimeFrom": iso(win[0]), "publishDateTimeTo": iso(win[1])})
    rows = sum(pmap(one, list(chunks(start, end, pd.Timedelta(days=1)))), [])
    if not rows:
        return pd.DataFrame(columns=["publish", "target", "fc"])
    df = pd.DataFrame(rows)
    df["publish"] = pd.to_datetime(df["publishTime"], utc=True)
    df["target"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(columns={"generation": "fc"})
    return df[["publish", "target", "fc"]].drop_duplicates(["publish", "target"]).sort_values(["publish", "target"])


def fetch_wind_actual(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Metered half-hourly wind generation (FUELHH), published ~30 min after each period ends."""
    days = pd.date_range(start.tz_convert("Europe/London").date() - timedelta(days=1),
                         end.tz_convert("Europe/London").date(), freq="7D")
    def one(d):
        return get("datasets/FUELHH", {"settlementDateFrom": str(d.date()),
                                        "settlementDateTo": str((d + pd.Timedelta(days=6)).date()), "fuelType": "WIND"})
    rows = sum(pmap(one, days), [])
    if not rows:
        return pd.DataFrame(columns=["start", "wind"])
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(columns={"generation": "wind"})[["start", "wind"]]
    return df.drop_duplicates("start", keep="last").sort_values("start")


def fetch_remit(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    def one(win):
        return get("datasets/REMIT", {"publishDateTimeFrom": iso(win[0]), "publishDateTimeTo": iso(win[1])})
    rows = sum(pmap(one, list(chunks(start, end, pd.Timedelta(days=1)))), [])
    cols = ["mrid", "revisionNumber", "publishTime", "unavailabilityType", "eventType", "fuelType",
            "assetId", "participantId", "normalCapacity", "availableCapacity", "eventStartTime",
            "eventEndTime", "cause", "eventStatus"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).reindex(columns=cols)
    for c in ("publishTime", "eventStartTime", "eventEndTime"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df.drop_duplicates(["mrid", "revisionNumber"]).sort_values("publishTime")


# ---------------------------------------------------------------- frames

def build_periods(sp: pd.DataFrame, mid: pd.DataFrame) -> pd.DataFrame:
    """Half-hour frame with SIP, MID and the tradeable spread."""
    df = sp.merge(mid, on="start", how="outer").sort_values("start").reset_index(drop=True)
    df["spread"] = df["sip"] - df["mid"]          # what a 1 MW long earns per MWh
    return df


def latest_forecast_asof(wf: pd.DataFrame, targets: pd.Series, cutoff: pd.Series) -> pd.Series:
    """For each period, the latest wind forecast published at or before its cutoff (hourly forecast,
    so the period maps to the hour it starts in)."""
    q = pd.DataFrame({"target": targets.dt.floor("h"), "cutoff": cutoff, "i": range(len(targets))})
    q = q.sort_values("cutoff")
    w = wf.sort_values("publish")
    out = pd.merge_asof(q, w.rename(columns={"publish": "cutoff"}), on="cutoff", by="target", direction="backward")
    return out.sort_values("i")["fc"].reset_index(drop=True)


def forecast_asof_lag(wf: pd.DataFrame, targets: pd.Series, cutoff: pd.Series, lag: pd.Timedelta) -> pd.Series:
    return latest_forecast_asof(wf, targets, cutoff - lag)
