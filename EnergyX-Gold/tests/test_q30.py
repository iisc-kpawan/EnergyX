"""Independent check of Q30: office buildings with |Spearman correlation| > 0.75 against site dew point in 2017.

Recomputed from BDG2's own files with pandas and numpy. Spearman is the Pearson correlation of
average ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q30"
VARIABLE = "dewTemperature"
METHOD = "spearman"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
MIN_VALID_HOURS = 0.90 * 8760
THRESHOLD = 0.75
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def correlations(raw) -> pd.Series:
    readings = raw.electricity_cleaned
    year = readings[(readings.index >= YEAR_START) & (readings.index < YEAR_END)]
    meta = raw.metadata.set_index("building_id")
    offices = [b for b in year.columns if b in meta.index and meta.loc[b, "primaryspaceusage"] == "Office"]
    valid = year[offices].notna().sum()
    cohort = sorted(valid.index[valid >= MIN_VALID_HOURS])
    weather = raw.weather
    weather = weather[(weather["timestamp"] >= YEAR_START) & (weather["timestamp"] < YEAR_END)]
    assert not weather.duplicated(["site_id", "timestamp"]).any()
    site_values = weather.pivot(index="timestamp", columns="site_id", values=VARIABLE)
    scores = {}
    for building in cohort:
        frame = pd.DataFrame({"x": year[building], "y": site_values[meta.loc[building, "site_id"]]}).dropna()
        if METHOD == "spearman":
            frame = frame.rank(method="average")
        scores[building] = float(np.corrcoef(frame["x"], frame["y"])[0, 1])
    return pd.Series(scores)


def test_members(raw, gold):
    scores = correlations(raw)
    expected = sorted(scores.index[scores.abs() > THRESHOLD])
    output = gold(TASK_ID)
    assert output["result"]["answer"] == expected
    assert output["result"]["count"] == len(expected)
    assert output["evidence"]["cohort_size"] == len(scores)
    reported = {e["building_id"]: e["correlation"] for e in output["evidence"]["correlations"]}
    assert set(reported) == set(scores.index)
    for building, value in scores.items():
        assert reported[building] == pytest.approx(value, rel=RELATIVE, abs=ABSOLUTE), building


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
