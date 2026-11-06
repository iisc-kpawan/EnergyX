"""Q11: Rat_office_Adele's total in July 2016, from BDG2's cleaned CSV."""

from __future__ import annotations

import pandas as pd
import pytest

TASK = "Q11"
BUILDING = "Rat_office_Adele"
START, END = pd.Timestamp("2016-07-01"), pd.Timestamp("2016-08-01")


def test_matches_raw_bdg2_file(gold, raw):
    series = raw.electricity_cleaned[BUILDING]
    july = series[(series.index >= START) & (series.index < END)]
    output = gold(TASK)
    assert output["result"]["answer"] == pytest.approx(float(july.sum()), rel=1e-6)
    assert output["result"]["unit"] == "kWh"
    assert output["evidence"]["valid_hours"] == int(july.notna().sum())
    assert output["evidence"]["window_hours"] == 744
