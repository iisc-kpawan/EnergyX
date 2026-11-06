"""Q16: Cockatoo_office_Alton's peak hour of day in 2017, from BDG2's cleaned CSV."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q16"
BUILDING = "Cockatoo_office_Alton"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")


def profile(raw) -> pd.Series:
    series = raw.electricity_cleaned[BUILDING]
    year = series[(series.index >= START) & (series.index < END)].dropna()
    return year.groupby(year.index.hour).mean()


def test_matches_raw_bdg2_file(gold, raw):
    means = profile(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == int(means.idxmax())
    assert output["evidence"]["profile_mean_kwh_by_hour"] == pytest.approx(means.tolist(), rel=1e-6)


def test_the_peak_is_clear(raw):
    means = profile(raw).sort_values(ascending=False)
    assert means.iloc[0] > means.iloc[1]
