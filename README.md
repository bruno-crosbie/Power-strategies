# UK power paper trading

Three GB power trading strategies, backtested on 12 months of Elexon data and paper-traded live for 10 days.
**No real money. Not financial advice.**

**Live dashboard:** https://bruno-crosbie.github.io/Power-strategies/

## How a trade works
Each trade is 1 MW for one half-hour settlement period, entered at the Elexon market index price (MID) and settled at the
system imbalance price (SIP): P&L = position × (SIP − MID) × 0.5 MWh, less an assumed £2/MWh cost. Decisions use only data
published before the period's gate closure (60 min before it starts).

## Strategies
| Strategy | Idea | Data |
|---|---|---|
| Weather fade | Fade the price premium on unusually calm (low wind forecast) periods | Elexon WINDFOR, FUELHH |
| REMIT outage | Go long after a sudden unplanned trip of a ≥800 MW non-weather unit | Elexon REMIT |
| Imbalance momentum | The system's imbalance direction persists; follow unusually large recent NIV | Elexon system prices |

## Files
- `common.py` – Elexon API fetchers and the trade model
- `strategies.py` – the three strategies and their parameter grids
- `fetch_history.py`, `backtest.py` – download a year of data, pick settings on Oct–Jun, test on Jul–Oct → `results/backtest.json`
- `live.py` – one live step: decide the next tradeable half-hour, settle old trades, rebuild the dashboard
- `report.py` – builds `results/dashboard.html` and `docs/index.html`
- `.github/workflows/paper-trade.yml` – runs `live.py` every 30 minutes on GitHub Actions and commits the results
