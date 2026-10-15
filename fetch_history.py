"""Download 12 months of Elexon history for backtesting into ./data."""
import sys
import time

import pandas as pd

from common import DATA, fetch_mid, fetch_remit, fetch_system_prices, fetch_wind_actual, fetch_windfor

END = pd.Timestamp.now(tz="UTC").floor("D")
START = END - pd.Timedelta(days=365)

jobs = {
    "system_prices": fetch_system_prices,
    "mid": fetch_mid,
    "windfor": lambda s, e: fetch_windfor(s - pd.Timedelta(days=3), e),  # forecasts are published ahead
    "wind_actual": fetch_wind_actual,
    "remit": fetch_remit,
}

for name, fn in jobs.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    t = time.time()
    df = fn(START, END)
    df.to_pickle(DATA / f"{name}.pkl")
    print(f"{name}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)
