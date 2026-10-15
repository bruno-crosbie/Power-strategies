"""Download data, run the three weather-fade backtests, build results/report.html.

    py run_weather.py                 # full run (first time: ~10 min of downloads + ~7 min wind-feature cache)
    py run_weather.py --no-download   # reuse data already in data/raw
"""
import importlib
import sys
import time
import traceback

from lib.sources import download_all
from report import build_report

STRATEGIES = ["power_week_ahead", "nbp_week_ahead", "weather_month_ahead"]
SOURCES = {"elexon", "gb_wind_fc", "gb_temp_fc", "eu_temp_fc", "gas", "temps"}

if __name__ == "__main__":
    if "--no-download" not in sys.argv:
        print("1/3 downloading data")
        download_all(SOURCES)
    print("2/3 backtesting")
    for name in STRATEGIES:
        t = time.time()
        try:
            importlib.import_module(f"strategies.{name}").run()
            print(f"  {name}: done in {time.time() - t:.0f}s", flush=True)
        except Exception:
            print(f"  {name}: FAILED
{traceback.format_exc()}", flush=True)
    print("3/3 building report")
    print("report:", build_report())
