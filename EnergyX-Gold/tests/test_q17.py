"""Q17: Bear_education_Alfredo's highest-total month in 2017, from BDG2's cleaned CSV."""

from __future__ import annotations

import calendar

import pandas as pd
import pytest

TASK = "Q17"
BUILDING = "Bear_education_Alfredo"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")


def monthly(raw) -> pd.DataFrame:
    series = raw.electricity_cleaned[BUILDING]
    year = series[(series.index >= START) & (series.index < END)].dropna()
    return year.groupby(year.index.month).agg(["sum", "count"])


def test_matches_raw_bdg2_file(gold, raw):
    months = monthly(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == calendar.month_name[int(months["sum"].idxmax())]
    reported = {row["month"]: row for row in output["evidence"]["months"]}
    for month, row in months.iterrows():
        assert reported[calendar.month_name[month]]["total_kwh"] == pytest.approx(float(row["sum"]), rel=1e-6)
        assert reported[calendar.month_name[month]]["valid_hours"] == int(row["count"])


def test_the_peak_is_clear(raw):
    totals = monthly(raw)["sum"].sort_values(ascending=False)
    assert totals.iloc[0] > 1.05 * totals.iloc[1]
