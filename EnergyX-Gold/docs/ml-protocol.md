# Method for the forecasting, anomaly and clustering questions

The question list names models but not their settings. This document fixes the settings once, so every answer is computed the same way and can be repeated. Each question's own choices (cohort, period, thresholds) are in `data/gold60_questions.yaml`. [reviews/gold60-review.md](../reviews/gold60-review.md) lists the choices that need confirming.

The code is in `energyx_gold/analytics/`: `series.py`, `forecasting.py`, `anomaly.py` and `clustering.py`.

## Data

- **Readings:** valid hourly readings from the snapshot's `electricity` view, in site-local standard time. An hour with no valid reading is missing.
- **Cohorts:**
  - A building type is BDG2's primary space usage: Education, Office, Healthcare, Parking, Entertainment/public assembly, Public services, and so on.
  - "US" and "UK" are the site's country. The anonymous sites (Bobcat, Eagle, Gator, Hog, Swan) are in neither.
  - A cohort includes only buildings with at least one valid reading somewhere in 2016 to 2017.
  - A building with no readings in a question's period, or none to test on, gets no result. It is left out of averages, and the evidence names it.
- **Model inputs:** gaps of up to 24 hours are filled by linear interpolation; longer gaps stay missing. Errors are always measured against real readings, never filled ones.
- **Period:** a trained model uses only readings inside the question's stated period, including for lags and windows. So the first days of a period have no lags. Tiny Time Mixers is the exception: its 512-hour context may start before the period.
- **Weather:** the site's observed hourly weather at the hour being forecast, which assumes a perfect weather forecast. Gaps of up to 24 hours are filled linearly. Where a question says so, remaining gaps are filled with the training median.

## Forecasting

- **Setup:** day-ahead. At 00:00 each day, the model forecasts that day's 24 hours.
- **Split:** a period named in a question is split by whole days. The first floor(80%) of days train, and the rest test. A zero-shot model (Tiny Time Mixers) needs no training, so every day of the period is forecast.

| Model | Settings |
|---|---|
| Seasonal naive | The reading 168 hours earlier |
| LightGBM | Features hour, day of week, weekend flag, and lags of 24, 48 and 168 hours. Since no lag is under 24 hours, it never needs its own forecasts. 300 trees, learning rate 0.05, 31 leaves, `random_state` 0, deterministic, one thread. Trained on real readings from the training days |
| KNN | The 168 hours before midnight predict the next 24. A window starts every day, so windows overlap. Windows use the filled series, for targets too. 5 neighbours; inputs min-max scaled on the training windows. Windows with a missing hour are skipped |
| ARIMA | SARIMAX (2,0,1)x(1,0,1,24) with a constant, fitted once on the training hours. Each test day is forecast from the readings up to its midnight, with the parameters fixed |
| Tiny Time Mixers | `ibm-granite/granite-timeseries-ttm-r2`, branch `512-48-ft-r2.1`, pinned to commit `b972f0c`. Zero-shot on CPU, with the hourly frequency token. The 512 hours before midnight, standardised by their own mean and standard deviation, predict the day's 24 hours, taken from the first 24 of the model's 48 outputs |

**Error metrics**, pooled over a building's forecast hours:

| Metric | Formula |
|---|---|
| MAE | mean \|forecast - reading\| (kWh) |
| MSE | mean squared error |
| RMSE | square root of MSE |
| MAPE | mean \|forecast - reading\| / reading × 100, over hours whose reading isn't 0 |

For a cohort, the metric is the plain average of the buildings' values.

## Anomalies

| Method | Settings |
|---|---|
| Modified z-score | 0.6745 × (x - median) / MAD (Iglewicz and Hoaglin). If the MAD is 0, it uses 1.253314 × the mean absolute deviation. Threshold 3.5 unless the question gives one |
| Isolation Forest | Features value, hour of day (0 to 23) and day of week (Monday 0); contamination "auto"; `random_state` 0, as in EnergyX's `detect_anomalies` tool |
| Change points | ruptures `Pelt`, model "l2", minimum segment as the question gives, jump 24, penalty 2 x variance x log(number of points) with the variance of the whole signal |
| LOF | 20 neighbours. The score is `-negative_outlier_factor_` |

## Clustering and similarity

- **Profile:** a building's mean valid reading for each local hour of the day (24 values), or for each hour of the week (168 values), over 2016 to 2017.
- **Scaling:** each profile is min-max scaled to 0 to 1. A flat profile becomes all zeros.
- **K-Means:** 10 initialisations, `random_state` 0.
- **Agglomerative clustering:** Ward linkage on Euclidean distance, scored by the Euclidean silhouette.
- **DTW clustering:** tslearn's `TimeSeriesKMeans` with DTW distance, 10 initialisations, `random_state` 0, at most 50 iterations, scored by tslearn's silhouette with `metric="dtw"`.
- **Choosing k:** when a question doesn't fix k, k runs from 2 to 10 and the highest silhouette score wins. Scores are compared at 12 decimals and a tie goes to the smaller k. Several questions are decided by gaps under 0.0001, so `reviews/gold60-review.md` flags them.

## Repeatability

- Every random seed comes from `context.seed` (0).
- DuckDB, LightGBM, BLAS and PyTorch each run on one thread.
- Two runs give identical output (`validate_gold`).
