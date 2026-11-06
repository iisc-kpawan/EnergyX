"""Independent check of Q27: Pearson correlation of Fox_office_Alice's 2017 readings with Fox air temperature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q27"
BUILDING = "Fox_office_Alice"
SITE = "Fox"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def test_correlation(raw, gold):
    readings = raw.electricity_cleaned[BUILDING]
    weather = raw.weather[raw.weather["site_id"] == SITE].set_index("timestamp")["airTemperature"]
    assert not weather.index.duplicated().any()
    frame = pd.DataFrame({"value": readings, "temp": weather}).dropna()
    frame = frame[(frame.index >= YEAR_START) & (frame.index < YEAR_END)]
    expected = float(np.corrcoef(frame["value"], frame["temp"])[0, 1])

    output = gold(TASK_ID)
    assert output["result"]["answer"] == pytest.approx(expected, rel=RELATIVE, abs=ABSOLUTE)
    assert output["evidence"]["paired_hours"] == len(frame)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
