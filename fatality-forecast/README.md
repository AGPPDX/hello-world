# US Traffic Fatality Forecast, 2025–2029

Forecasts annual US motor-vehicle traffic deaths for the next five years (2025–2029), nationally and for each of the 50 states and DC, from 30 years (1995–2024) of NHTSA data.

**Published report:** https://claude.ai/artifact/Xsf5HPFRNqwHbMLaa6cTBi (private; share from the page's Share menu)

## Result

| Year | Forecast | 80% range |
|---|---|---|
| 2025 | 39,629 | 37,069 – 42,366 |
| 2026 | 39,994 | 35,407 – 45,175 |
| 2027 | 40,137 | 34,202 – 47,101 |
| 2028 | 40,068 | 33,490 – 47,938 |
| 2029 | 40,181 | 33,387 – 48,357 |

## State forecasts

All 50 states and DC, in `data/state_forecasts.csv` (one row per state and year, with 80% and 95% ranges and deaths per 100,000 people). State forecasts are scaled so they add up to the national forecast each year.

- Six models per state: naive (last year), 3-year average, damped trend, ARIMA, rate per person × population, and one gradient-boosting model trained on all states together.
- The blend of the top three (ARIMA, 3-year average, damped trend), scaled to the national total, averaged 6% error one year out and 15% five years out in backtests. That is about the same as the naive baseline overall; it beat the baseline in 29 of 51 states.
- The pooled machine-learning model did not beat the per-state models.
- Small states (DC, RI, ND, VT, WY) swing the most year to year, so their ranges are widest.
- State miles driven are not used: FHWA publishes them only as yearly spreadsheets whose layout changes. Population is used instead.

## Data

| File | What it is | Source |
|---|---|---|
| `data/fars_annual.json` | Fatal crashes and deaths per year, 1995–2024 | NHTSA FARS national annual files (`static.nhtsa.gov`) |
| `data/raw/TRFVOLUSM227NFWA.csv` | Vehicle miles traveled, monthly | FHWA, via FRED |
| `data/raw/POPTHM.csv` | US population, monthly | Census/BEA, via FRED |
| `data/raw/UNRATE.csv` | Unemployment rate, monthly | BLS, via FRED |
| `data/results.json` | Model scores, forecasts, backtest detail | Output of `src/model.py` |
| `data/fars_state_annual.csv` | Fatal crashes and deaths by state and year, 1995–2024 | NHTSA FARS (same files) |
| `data/raw/states/{ST}POP.csv` | State population, annual | Census, via FRED |
| `data/raw/states/{ST}UR.csv` | State unemployment rate, monthly | BLS, via FRED |
| `data/state_results.json` | State model scores, forecasts, per-state backtest error | Output of `src/state_model.py` |
| `data/state_forecasts.csv` | State forecasts 2025–2029 with ranges | Output of `src/state_model.py` |

The NHTSA CrashAPI blocks automated requests from cloud servers, so FARS bulk files are used instead. They contain the same records.

## Models

Eight models are scored with a rolling backtest: train on 1995 up to each year from 2005 to 2019, predict the next 1–5 years, and measure average percent error (MAPE).

Naive (last year) · linear drift · damped-trend exponential smoothing · auto ARIMA · ARIMAX (miles driven + unemployment) · rate-per-mile × miles driven · ridge regression · gradient boosting.

The forecast is an inverse-error-weighted blend of the three best non-baseline models. Ranges come from the blend's own backtest errors.

## Rerun it

Requires Python 3.10+ and `curl`. From this folder:

```bash
pip install -r requirements.txt
python src/fetch_fars.py /tmp/fars data/fars_annual.json data/fars_state_annual.csv   # ~1 GB download, a few minutes
python src/fetch_fred.py data/raw
python src/fetch_fred_states.py data/raw/states
python src/model.py data          # national; writes data/results.json
python src/state_model.py data    # states (needs results.json); ~2 minutes
python src/build_report.py data/results.json data/state_results.json report/template.html report/traffic-fatality-outlook.html
```

`model.py` and `state_model.py` print statsmodels convergence warnings during the backtest. They are expected; run with `python -W ignore` to hide them.

**When NHTSA publishes the 2025 FARS file**, change `range(1995, 2025)` in `fetch_fars.py`, the year bounds in `model.py` (`1995:2024`, `2025:2029`, backtest origins) and `Y0`, `Y1`, `ORIGINS` in `state_model.py` forward by one year.
