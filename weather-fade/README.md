# Weather fade research: UK power and NBP gas

Does the forward market lean too hard on the weather? These backtests test three versions of a
"weather fade" for GB power and NBP gas, using only free data and strict no-peeking rules.
Paper research only: no real money, not financial advice.

**Full write-up with charts:** https://bruno-crosbie.github.io/Power-strategies/weather-fade.html
(also in [`results/report.html`](results/report.html)).

## The three strategies

| Strategy | The bet | Result in the untouched test period |
|---|---|---|
| [`power_week_ahead.py`](strategies/power_week_ahead.py) | The market over-trusts 5-7 day wind forecasts, so fade extreme calm or windy weeks | **No tradeable fade.** Forecasts are reliable to ~5 days; the pull-back is too small after costs |
| [`nbp_week_ahead.py`](strategies/nbp_week_ahead.py) | Same for gas, with GB and Continental temperatures plus GB wind | **No signal.** Weather explains little of weekly gas moves |
| [`weather_month_ahead.py`](strategies/weather_month_ahead.py) | Recent unusual weather won't carry into next month or quarter, so fade a price anchored on recent spot | **Promising.** Beat a spot-anchored price on all 4 products (power/gas × month/quarter) |

## Live: month-ahead fade against real quotes

[`live_month_ahead.py`](live_month_ahead.py) runs on GitHub Actions (daily, and straight after you edit
[`live/quotes.json`](live/quotes.json)). It prices next month and next quarter for UK power and NBP gas,
and when you enter the market price you see for a contract, it paper-trades toward our fair price if the
gap is 2% or more, then settles against the actual average spot price after delivery.
Page: https://bruno-crosbie.github.io/Power-strategies/month-ahead.html

## How the week-ahead fade works

1. Every Sunday at 22:00 UTC, take the ECMWF, GFS and ICON forecasts for the next 7 days (wind at 8 GB
   wind-farm areas; temperature in 4 GB and 4 Continental cities), **only runs already published by then**.
2. Learn a power curve: wind speeds → GB wind capacity factor (checked against Elexon metered wind).
3. Measure forecast skill at each lead time. The "trust factor" is ~1.0 up to 4 days ahead, 0.87 at 6 days
   and 0.69 at 7 days. That pull-back toward normal is what a fade would earn on.
4. Price next week: a weather-neutral price level × how prices react to wind, cold and weekday.
5. **fair** uses the pulled-back forecast; **face** uses the raw forecast (an over-trusting market);
   **naive** uses recent prices (a market ignoring the weather).
6. Real week-ahead quotes aren't free, so the market is simulated as trusting the forecast 0-100%,
   and results are shown for each case.

The month/quarter version prices the next period from today's weather-neutral level and **normal**
weather for the time of year, and trades against a market anchored 50-100% on the last 14 or 28 days of spot.

## Run it

```bash
pip install pandas numpy requests openpyxl lxml
python run_weather.py                 # downloads ~10 min, first wind-feature build ~7 min, then ~1-2 min
python run_weather.py --no-download   # rerun on cached data
```

## Testing rules

- **No peeking:** a weather model run is treated as public 8 hours after it starts, and models are refit
  every 4 weeks on past data only.
- **Tune early, test late:** settings are chosen on data before the split date in `config.py`; everything
  after it is an untouched test.
- **Costs:** £1/MWh for power, 0.5p/therm for gas; a minimum edge is required before trading.

## Data (free, no keys)

Elexon BMRS (power prices, metered wind) · Open-Meteo previous-runs archive (forecasts by lead time,
from March 2024) and ERA5 (observed temperatures) · ONS / National Gas daily System Average Price of gas.
