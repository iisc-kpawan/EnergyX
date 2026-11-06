"""Independent check of Q40: seasonal naive against Tiny Time Mixers, Dublin offices, 8 January 2016.

Recomputed with pandas over BDG2's electricity_cleaned.csv and BDG2_metadata.csv, calling
tsfm_public and torch directly: no snapshot, no DuckDB, no energyx_gold.analytics.

Set HF_HUB_OFFLINE=1 once the model is in the local Hugging Face cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TASK_ID = "Q40"
TARGET = pd.Timestamp("2016-01-08 00:00")
HISTORY = 168
HORIZON = 24
TTM_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
TTM_BRANCH = "90-30-ft-r2.1"
TTM_REVISION = "6e5cb8ee51e0634a45637490f5db43148b2fa6be"
FREQ_TOKEN = 7  # hourly


def fill(values: pd.Series, limit: int = 24) -> pd.Series:
    gap = values.isna()
    run = (gap != gap.shift()).cumsum()
    length = gap.groupby(run).transform("sum")
    return values.interpolate(limit_area="inside").where(~(gap & (length > limit)))


@pytest.fixture(scope="module")
def want(raw) -> dict:
    import torch
    from tsfm_public.toolkit.get_model import get_model

    assert get_model(TTM_MODEL, context_length=HISTORY, prediction_length=HORIZON, return_model_key=True) == TTM_BRANCH
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = get_model(TTM_MODEL, context_length=HISTORY, prediction_length=HORIZON,
                      model_revision=TTM_REVISION).eval()
    context = model.config.context_length

    meta = raw.metadata
    readings = raw.electricity_cleaned
    named = meta.loc[(meta["City"] == "Dublin") & (meta["primaryspaceusage"] == "Office"), "building_id"]
    cohort = sorted(b for b in named if b in readings.columns and readings[b].notna().any())

    history_index = pd.date_range(TARGET - pd.Timedelta(hours=HISTORY), periods=HISTORY, freq="h")
    target_index = pd.date_range(TARGET, periods=HORIZON, freq="h")
    per_building = {}
    for building in cohort:
        history = fill(readings[building].astype(float).reindex(history_index)).to_numpy()
        actual = readings[building].astype(float).reindex(target_index).to_numpy()
        if np.isnan(history).any() or np.isnan(actual).any():
            continue
        naive = history[:HORIZON]  # 168 hours before each target hour
        window = history[-context:]
        mean, sd = window.mean(), window.std() or 1.0
        x = torch.tensor((window - mean) / sd, dtype=torch.float32).reshape(1, context, 1)
        with torch.no_grad():
            out = model(past_values=x, freq_token=torch.tensor([FREQ_TOKEN])).prediction_outputs
        ttm = out[0, :HORIZON, 0].numpy().astype(float) * sd + mean
        per_building[building] = {"seasonal_naive": float(np.abs(naive - actual).mean()),
                                  "ttm": float(np.abs(ttm - actual).mean())}
    return {"context": int(context), "per_building": per_building,
            **{f"{m}_median_mae": float(np.median([r[m] for r in per_building.values()]))
               for m in ("seasonal_naive", "ttm")}}


def test_medians_match(gold, want):
    answer = gold(TASK_ID)["result"]["answer"]
    for key in ("seasonal_naive_median_mae", "ttm_median_mae"):
        assert answer[key] == pytest.approx(want[key], rel=1e-6, abs=5e-7)
    assert answer["buildings"] == len(want["per_building"])
    better = "Seasonal Naive" if want["seasonal_naive_median_mae"] < want["ttm_median_mae"] else "TTM"
    assert answer["lower_median_mae_model"] == better


def test_every_building_matches(gold, want):
    evidence = gold(TASK_ID)["evidence"]
    assert evidence["definition"]["ttm"]["context_hours_used"] == want["context"] == 90
    assert sorted(evidence["per_building"]) == sorted(want["per_building"])
    for building, record in evidence["per_building"].items():
        for m in ("seasonal_naive", "ttm"):
            assert record[f"{m}_mae"] == pytest.approx(want["per_building"][building][m], rel=1e-6, abs=5e-7)
