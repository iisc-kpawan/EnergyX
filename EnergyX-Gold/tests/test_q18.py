"""Q18: Cockatoo_office_Alton's weekday and weekend profiles in 2017, from BDG2's cleaned CSV."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q18"
BUILDING = "Cockatoo_office_Alton"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")


def profiles(raw) -> tuple[pd.Series, pd.Series]:
    series = raw.electricity_cleaned[BUILDING]
    year = series[(series.index >= START) & (series.index < END)].dropna()
    weekday = year[year.index.dayofweek < 5]
    weekend = year[year.index.dayofweek >= 5]
    return weekday.groupby(weekday.index.hour).mean(), weekend.groupby(weekend.index.hour).mean()


def test_matches_raw_bdg2_file(gold, raw):
    weekday, weekend = profiles(raw)
    gap = weekday - weekend
    answer = gold(TASK)["result"]["answer"]
    assert answer["weekday_mean_kwh"] == pytest.approx(float(weekday.mean()), rel=1e-6)
    assert answer["weekend_mean_kwh"] == pytest.approx(float(weekend.mean()), rel=1e-6)
    assert answer["weekday_to_weekend_ratio"] == pytest.approx(float(weekday.mean() / weekend.mean()), rel=1e-6)
    assert answer["weekday_peak_hour"] == int(weekday.idxmax())
    assert answer["weekend_peak_hour"] == int(weekend.idxmax())
    assert answer["largest_gap_hour"] == int(gap.abs().idxmax())
    assert answer["largest_gap_kwh"] == pytest.approx(float(gap[gap.abs().idxmax()]), rel=1e-6)
    evidence = gold(TASK)["evidence"]
    assert evidence["weekday_profile_mean_kwh_by_hour"] == pytest.approx(weekday.tolist(), rel=1e-6)
    assert evidence["weekend_profile_mean_kwh_by_hour"] == pytest.approx(weekend.tolist(), rel=1e-6)
