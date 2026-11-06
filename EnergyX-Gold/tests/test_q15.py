"""Q15: 2017 totals of the largest 10% and smallest 50% of buildings by floor area, from BDG2's CSVs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TASK = "Q15"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")


def test_matches_raw_bdg2_file(gold, raw):
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END)]
    totals = year.sum()[year.notna().any()]
    area = raw.metadata.set_index("building_id").loc[totals.index, "sqm"]
    assert area.notna().all() and (area > 0).all()
    largest_cut, smallest_cut = np.percentile(area, 90), np.percentile(area, 50)
    largest, smallest = totals[area >= largest_cut], totals[area <= smallest_cut]
    answer = gold(TASK)["result"]["answer"]
    assert answer["largest_10pct_buildings"] == len(largest)
    assert answer["smallest_50pct_buildings"] == len(smallest)
    assert answer["largest_10pct_total_kwh"] == pytest.approx(float(largest.sum()), rel=1e-6)
    assert answer["smallest_50pct_total_kwh"] == pytest.approx(float(smallest.sum()), rel=1e-6)
    assert answer["ratio_largest_to_smallest"] == pytest.approx(float(largest.sum() / smallest.sum()), rel=1e-6)
    assert gold(TASK)["evidence"]["buildings_with_2017_readings"] == len(totals)
