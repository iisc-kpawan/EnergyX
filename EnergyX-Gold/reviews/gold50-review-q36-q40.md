# Review notes: Q36 to Q40

Category 8 (forecasting and predictive analytics). Every module matches its independent check on BDG2's own CSV files: statsforecast is called through its StatsForecast frame interface there, and LightGBM and Tiny Time Mixers are called directly. Two runs give the same output hash. Times are site-local standard time. Errors are measured only against real readings; model inputs use the filled series (linear, gaps up to 24 hours) where the definition says so.

## Q36: seasonal naive, Lamb_education_Norris, 8 May 2016 00:00 to 11:00

- Answer 0.225 kWh (unrounded 0.225), over 12 scored hours.
- The definition doesn't name the building; the question does. The module uses Lamb_education_Norris.
- The forecast is the real reading 168 hours earlier. No gap filling is pinned. 1 May 2016 has no missing hour, so filling would change nothing.
- Scoring all 24 hours of 8 May instead of 12 gives 0.229167 (the Q37 seasonal naive MAE).

## Q37: seasonal naive against AutoARIMA, Lamb_education_Norris, 8 May 2016

- Answer: seasonal naive 0.229167 kWh, AutoARIMA 0.544093 kWh; the lower is Seasonal Naive.
- AutoARIMA chose ARIMA(1,1,0)(1,0,0)[24] (ar1 0.466, sar1 0.571, AICc 739.50), fitted to the 168 history hours. The history has no gap, so filling changes nothing.
- AutoARIMA with `stepwise=False` (full search) chooses ARIMA(0,1,4)(1,0,0)[24] and gives 0.394735. The winner stays Seasonal Naive.
- statsforecast's AutoARIMA is deterministic here; the model and the frame interface agree to 1e-6.

## Q38: AutoARIMA daily forecasts, London education buildings, 8 to 14 May 2016

- Answer: median MAE 465.24678 kWh over 22 buildings. All 22 London education buildings (all at Robin) have 30 complete history days and 7 complete target days. The mean MAE is 636.711084.
- 22 is even, so the median is the mean of the 11th and 12th MAEs (443.347357 and 487.146203). Each is 4.7% from the median, so either one alone would fall outside the 2% tolerance.
- Robin_education_Della has a flat meter: every hour reads 0.7492 kWh, so every day is 17.9808 kWh and its MAE is 0. Leaving it out gives 21 buildings and a median of 487.146203 (4.7% higher). Flagged: a reviewer may want to drop constant series from the cohort.
- Chosen models range from ARIMA(0,0,0) with mean to ARIMA(1,0,0)(1,1,0)[7]; each building's model is in the evidence.
- Days are site-local calendar days and a day with a missing hour doesn't count. No day in the window is incomplete, so the rule doesn't bind.

## Q39: AutoARIMA against LightGBM, Peacock_office_Major, 1 March 2016

- Answer: AutoARIMA 8.563384 kWh, LightGBM 10.527901 kWh; the lower is AutoARIMA.
- AutoARIMA chose ARIMA(1,0,2)(0,1,1)[24].
- The history has 6 missing hours (23 February 08:00 to 10:00 and 12:00 to 13:00, 26 February 09:00), all filled. LightGBM trains on the 144 hours from 24 February to 29 February, with the filled series for both target and lag-24. Training only on real readings (143 rows, dropping 26 February 09:00) gives 10.346314. The winner stays AutoARIMA.
- LightGBM also runs with `n_jobs=1` and `verbose=-1`; neither changes the model. With the ml-protocol's general LightGBM settings (300 trees, 31 leaves, min_child_samples 20) and the same features, LightGBM gives 10.091854. The winner stays AutoARIMA.

## Q40: seasonal naive against Tiny Time Mixers, Dublin offices, 8 January 2016

- Answer: seasonal naive median MAE 3.092708 kWh, TTM 8.515341 kWh, over 9 buildings; the lower is Seasonal Naive.
- All 9 Wolf office buildings qualify. Wolf_office_Bobbie has one filled hour (7 January 13:00).
- Checkpoint: `get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=168, prediction_length=24)` selects the branch `90-30-ft-r2.1`, pinned to commit `6e5cb8ee51e0634a45637490f5db43148b2fa6be`. No TTM r2 checkpoint has a 168-hour context. This one reads 90 hours and outputs 30, cut to 24 by `prediction_filter_length`. Given all 168 hours, the model keeps the last 90 itself (checked: identical outputs), so TTM sees 4 January 06:00 to 7 January 23:00 only. The module passes those 90 hours explicitly and checks that get_model still selects this branch.
- The context is standardised by its own mean and population standard deviation, as in docs/ml-protocol.md. The checkpoint also scales internally, so dropping the external step gives 8.515325 (the same to 5 significant digits).
- The ml-protocol checkpoint (512-48-ft-r2.1) can't run here without padding: the data start on 1 January 2016. Zero-padding the 168 hours to 512 with an observed mask gives a TTM median of 6.194654. The winner stays Seasonal Naive.
- Seasonal naive for 8 January (a Friday) repeats 1 January, New Year's Day. It still beats TTM in all 9 buildings.

## Definitions to consider changing

- Q36: name the building in the definition, as Q37 and Q39 do.
- Q38: decide whether a flat meter (Robin_education_Della) belongs in the cohort. It moves the median by 4.7%, beyond the 2% tolerance.
- Q40: the definition says TTM "runs zero-shot on the 168 history hours". The selected checkpoint uses only the last 90. Either say so, or pin a checkpoint and padding rule that use all 168.
