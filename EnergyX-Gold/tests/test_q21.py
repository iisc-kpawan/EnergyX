"""Independent check of Q21: Fox_office_Alice's 2017 electricity EUI, from BDG2's own files."""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q21"
BUILDING = "Fox_office_Alice"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def test_eui(raw, gold):
    readings = raw.electricity_cleaned
    readings = readings.loc[(readings.index >= YEAR_START) & (readings.index < YEAR_END), BUILDING]
    area = float(raw.metadata.set_index("building_id").loc[BUILDING, "sqm"])
    assert area > 0
    expected = float(readings.sum()) / area

    output = gold(TASK_ID)
    assert output["result"]["answer"] == pytest.approx(expected, rel=RELATIVE, abs=ABSOLUTE)
    assert output["evidence"]["valid_hours"] == int(readings.notna().sum())
    assert output["evidence"]["floor_area_m2"] == pytest.approx(area, rel=RELATIVE)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
