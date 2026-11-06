"""Q12: Fox_office_Alice's highest daily total in 2017, from BDG2's cleaned CSV."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q12"
BUILDING = "Fox_office_Alice"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")


def daily(raw) -> pd.DataFrame:
    series = raw.electricity_cleaned[BUILDING]
    year = series[(series.index >= START) & (series.index < END)].dropna()
    days = year.groupby(year.index.normalize()).agg(["sum", "count"])
    return days.sort_values("sum", ascending=False, kind="stable")


def test_matches_raw_bdg2_file(gold, raw):
    days = daily(raw)
    answer = gold(TASK)["result"]["answer"]
    evidence = gold(TASK)["evidence"]
    assert answer["date"] == days.index[0].strftime("%Y-%m-%d")
    assert answer["total_kwh"] == pytest.approx(float(days["sum"].iloc[0]), rel=1e-6)
    assert evidence["peak_day_valid_hours"] == int(days["count"].iloc[0])
    assert evidence["runner_up"]["date"] == days.index[1].strftime("%Y-%m-%d")
    assert evidence["runner_up"]["total_kwh"] == pytest.approx(float(days["sum"].iloc[1]), rel=1e-6)


def test_the_peak_is_clear(raw):
    days = daily(raw)
    assert days["sum"].iloc[0] > days["sum"].iloc[1]
