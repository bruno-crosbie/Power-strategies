"""Every setting in one place. Change things here, not inside the strategy files."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"          # downloaded data, one file per source
PARTS = ROOT / "data" / "parts"      # chunk cache for big forecast downloads (lets a download resume)
RESULTS = ROOT / "results"
for p in (RAW, PARTS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

TZ = "Europe/London"

# ---------------------------------------------------------------- weather forecasts (Open-Meteo)
FORECAST_ARCHIVE_START = "2024-03-15"        # first day the archive keeps forecasts by lead time
FORECAST_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global"]   # European, US, German global models
MAX_LEAD_DAYS = 7                            # the archive keeps forecasts made up to 7 days before
PUBLISH_DELAY_HOURS = 8                      # a model run is assumed public 8h after it starts
DECISION_HOUR_UTC = 22                       # all week-ahead decisions are made at 22:00 UTC

# Main GB wind-farm areas (rough centres of the big offshore clusters and Scottish onshore)
GB_WIND_SITES = {
    "dogger_hornsea": (54.2, 1.9),
    "east_anglia_thames": (52.0, 2.0),
    "lincs_triton": (53.4, 0.9),
    "moray_firth": (58.2, -2.8),
    "forth_tay": (56.5, -1.9),
    "irish_sea": (54.0, -3.6),
    "s_scotland_onshore": (55.5, -3.8),
    "highlands_onshore": (57.4, -4.6),
}
GB_CITIES = {"london": (51.5, -0.12), "birmingham": (52.48, -1.9), "manchester": (53.48, -2.24), "glasgow": (55.86, -4.25)}
EU_CITIES = {"berlin": (52.52, 13.4), "paris": (48.86, 2.35), "amsterdam": (52.37, 4.9), "milan": (45.46, 9.19)}
HDD_BASE_C = 15.5                            # heating degree days: how far below 15.5°C the day is

# ---------------------------------------------------------------- testing protocol
# Settings are chosen on the "selection" period only; the "holdout" period is never used for tuning.
SPLITS = {
    "power_week_ahead": "2025-09-01",
    "nbp_week_ahead": "2025-09-01",
    "norway_supply": "2024-07-01",
    "brent_trend": "2015-01-01",
    "oil_carry": "2011-01-01",
    "brent_wti_spread": "2015-01-01",
}
WALK_FORWARD_START = "2024-09-01"            # forecast strategies: first decision after a 6-month warm-up
REFIT_EVERY_DAYS = 28

# ---------------------------------------------------------------- costs (round trip, per unit traded)
COST_POWER_GBP_MWH = 1.0                     # week-ahead power bid/offer
COST_GAS_P_THERM = 0.5                       # NBP bid/offer
COST_OIL_USD_BBL = 0.05                      # Brent/WTI futures, including slippage
MIN_EDGE_POWER_GBP_MWH = 2.0
MIN_EDGE_GAS_P_THERM = 1.5

# ---------------------------------------------------------------- units
KWH_PER_THERM = 29.3071
OIL_CONTRACT_BBL = 1000                      # one Brent or WTI futures contract
GAS_POSITION_THERMS_PER_DAY = 25_000         # one NBP "lot" is 1,000 therms/day; we paper-trade 25 lots
POWER_POSITION_MW = 1

NOW = pd.Timestamp.now(tz="UTC")
