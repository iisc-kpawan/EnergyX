"""Q19: the five Berkeley buildings with the highest summer-afternoon mean in 2017, from BDG2's CSVs."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q19"
START, END = pd.Timestamp("2017-06-01"), pd.Timestamp("2017-09-01")
HOURS = [14, 15, 16, 17]
WINDOW_HOURS = 368
TOP_N = 5


def ranking(raw) -> pd.Series:
    meta = raw.metadata
    cohort = sorted(meta.loc[meta["City"] == "Berkeley", "building_id"])
    assert set(meta.loc[meta["City"] == "Berkeley", "site_id"]) == {"Bear"}
    frame = raw.electricity_cleaned
    rows = (frame.index >= START) & (frame.index < END) & frame.index.hour.isin(HOURS)
    window = frame.loc[rows, cohort]
    assert len(window) == WINDOW_HOURS
    valid = window.notna().sum()
    kept = valid.index[valid >= 0.9 * WINDOW_HOURS]
    return window[kept].mean().sort_index().sort_values(ascending=False, kind="stable")


def test_matches_raw_bdg2_file(gold, raw):
    means = ranking(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == sorted(means.index[:TOP_N])
    assert output["evidence"]["buildings_meeting_coverage"] == len(means)
    for row in output["evidence"]["ranking"]:
        assert row["mean_kwh"] == pytest.approx(float(means[row["building_id"]]), rel=1e-6, abs=1e-6)


def test_the_cut_is_clear(raw):
    means = ranking(raw)
    assert means.iloc[TOP_N - 1] > 1.05 * means.iloc[TOP_N]
