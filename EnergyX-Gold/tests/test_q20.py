"""Q20: London education buildings with the largest weekday-weekend profile gap in 2017, from BDG2's CSVs."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q20"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
YEAR_HOURS = 8760
TOP_N = 5


def scale_rows(frame: pd.DataFrame) -> pd.DataFrame:
    low, high = frame.min(axis=1), frame.max(axis=1)
    return frame.sub(low, axis=0).div(high - low, axis=0)


def gaps(raw) -> pd.Series:
    meta = raw.metadata
    cohort = sorted(meta.loc[(meta["City"] == "London") & (meta["primaryspaceusage"] == "Education"), "building_id"])
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), cohort]
    valid = year.notna().sum()
    year = year[valid.index[valid >= 0.9 * YEAR_HOURS]]
    is_weekday = year.index.dayofweek < 5
    weekday = year[is_weekday].groupby(year.index[is_weekday].hour).mean().T
    weekend = year[~is_weekday].groupby(year.index[~is_weekday].hour).mean().T
    gap = (scale_rows(weekday) - scale_rows(weekend)).abs().mean(axis=1)
    return gap.sort_index().sort_values(ascending=False, kind="stable")


def test_matches_raw_bdg2_file(gold, raw):
    gap = gaps(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == sorted(gap.index[:TOP_N])
    assert output["evidence"]["buildings_ranked"] == len(gap)
    for row in output["evidence"]["ranking"]:
        assert row["mean_abs_gap"] == pytest.approx(float(gap[row["building_id"]]), rel=1e-6, abs=1e-6)


def test_the_cut_is_clear(raw):
    gap = gaps(raw)
    assert gap.iloc[TOP_N - 1] > gap.iloc[TOP_N]
