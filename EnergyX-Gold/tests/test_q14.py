"""Q14: London building types with a month over 1.2 times their annual average daily use, from BDG2's CSVs."""

from __future__ import annotations

import calendar

import pandas as pd
import pytest

TASK = "Q14"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
YEAR_HOURS = 8760


def monthly_ratios(raw) -> pd.DataFrame:
    """One row per building type, one column per month: month's average daily kWh / year's."""
    meta = raw.metadata
    london = meta[meta["City"] == "London"].set_index("building_id")
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), london.index]
    valid = year.notna().sum()
    kept = valid.index[valid >= 0.9 * YEAR_HOURS]
    by_type = year[kept].T.groupby(london.loc[kept, "primaryspaceusage"]).sum().T
    monthly = by_type.groupby(by_type.index.month).sum()
    days = pd.Series({m: calendar.monthrange(2017, m)[1] for m in range(1, 13)})
    return (monthly.div(days, axis=0) / (by_type.sum() / 365)).T


def test_matches_raw_bdg2_file(gold, raw):
    ratios = monthly_ratios(raw)
    output = gold(TASK)
    expected = sorted(ratios.index[(ratios > 1.2).any(axis=1)])
    assert output["result"]["answer"] == expected
    types = output["evidence"]["types"]
    assert sorted(types) == sorted(ratios.index)
    for building_type, row in ratios.iterrows():
        reported = types[building_type]["monthly_ratio"]
        for month, value in row.items():
            assert reported[calendar.month_name[month]] == pytest.approx(float(value), rel=1e-6, abs=1e-6)


def test_no_type_is_near_the_threshold(raw):
    highest = monthly_ratios(raw).max(axis=1)
    assert ((highest - 1.2).abs() > 0.05).all()
