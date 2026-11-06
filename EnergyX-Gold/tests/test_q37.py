"""Independent check of Q37: seasonal naive against AutoARIMA, Lamb_education_Norris, 8 May 2016.

Recomputed with pandas over BDG2's electricity_cleaned.csv, calling statsforecast through its
StatsForecast frame interface: no snapshot, no DuckDB, no energyx_gold.analytics.
"""

from __future__ import annotations

import pandas as pd
import pytest

TASK_ID = "Q37"
BUILDING = "Lamb_education_Norris"
TARGET = pd.Timestamp("2016-05-08 00:00")
HISTORY = 168
HORIZON = 24


def fill(values: pd.Series, limit: int = 24) -> pd.Series:
    gap = values.isna()
    run = (gap != gap.shift()).cumsum()
    length = gap.groupby(run).transform("sum")
    return values.interpolate(limit_area="inside").where(~(gap & (length > limit)))


@pytest.fixture(scope="module")
def want(raw) -> dict:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    values = raw.electricity_cleaned[BUILDING].astype(float)
    history_index = pd.date_range(TARGET - pd.Timedelta(hours=HISTORY), periods=HISTORY, freq="h")
    target_index = pd.date_range(TARGET, periods=HORIZON, freq="h")
    history = fill(values.reindex(history_index))
    assert history.notna().all()
    actual = values.reindex(target_index)

    frame = pd.DataFrame({"unique_id": BUILDING, "ds": history_index, "y": history.to_numpy()})
    sf = StatsForecast(models=[AutoARIMA(season_length=24)], freq="h", n_jobs=1)
    arima = sf.forecast(df=frame, h=HORIZON)["AutoARIMA"].to_numpy()
    naive = history.to_numpy()[:HORIZON]  # 1 May, same hours
    ok = actual.notna().to_numpy()
    maes = {
        "seasonal_naive_mae": float(abs(naive[ok] - actual.to_numpy()[ok]).mean()),
        "autoarima_mae": float(abs(arima[ok] - actual.to_numpy()[ok]).mean()),
    }
    return maes


def test_maes_match(gold, want):
    answer = gold(TASK_ID)["result"]["answer"]
    for key, value in want.items():
        assert answer[key] == pytest.approx(value, rel=1e-6, abs=5e-7)


def test_winner_matches(gold, want):
    better = "Seasonal Naive" if want["seasonal_naive_mae"] < want["autoarima_mae"] else "AutoARIMA"
    assert gold(TASK_ID)["result"]["answer"]["lower_mae_method"] == better
