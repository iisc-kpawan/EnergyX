"""Q13: the five US office buildings with the highest 2017 peak-to-average ratio, from BDG2's CSVs."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q13"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
YEAR_HOURS = 8760
TOP_N = 5
# BDG2's metadata gives either "US" or a US state as the country.
US = {"US", "Arizona", "Florida", "Texas", "New York", "California", "New Jersey"}


def ranking(raw) -> tuple[pd.Series, int]:
    meta = raw.metadata
    assert set(meta["Country"]) <= US | {"UK", "Ireland", "Canada", "Anonymous"}
    cohort = sorted(meta.loc[meta["Country"].isin(US) & (meta["primaryspaceusage"] == "Office"), "building_id"])
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), [b for b in cohort if b in frame.columns]]
    assert len(year) == YEAR_HOURS
    valid = year.notna().sum()
    kept = valid.index[valid >= 0.9 * YEAR_HOURS]
    ratios = year[kept].max() / year[kept].mean()
    return ratios.sort_index().sort_values(ascending=False, kind="stable"), len(cohort)


def test_matches_raw_bdg2_file(gold, raw):
    ratios, cohort = ranking(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == sorted(ratios.index[:TOP_N])
    assert output["result"]["count"] == TOP_N
    assert output["evidence"]["buildings_in_cohort"] == cohort
    assert output["evidence"]["buildings_meeting_coverage"] == len(ratios)
    for row in output["evidence"]["ranking"]:
        assert row["peak_to_average"] == pytest.approx(float(ratios[row["building_id"]]), rel=1e-6, abs=1e-6)


def test_the_cut_is_clear(raw):
    ratios, _ = ranking(raw)
    assert ratios.iloc[TOP_N - 1] > ratios.iloc[TOP_N]
