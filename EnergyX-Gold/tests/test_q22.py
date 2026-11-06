"""Independent check of Q22: the five office buildings with the highest 2017 EUI, from BDG2's own files."""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q22"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
MIN_VALID_HOURS = 0.90 * 8760
TOP_N = 5
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def office_eui(raw) -> pd.Series:
    readings = raw.electricity_cleaned
    year = readings[(readings.index >= YEAR_START) & (readings.index < YEAR_END)]
    meta = raw.metadata.set_index("building_id")
    offices = meta[(meta["primaryspaceusage"] == "Office") & meta["sqm"].notna() & (meta["sqm"] > 0)]
    buildings = [b for b in year.columns if b in offices.index]
    year = year[buildings]
    keep = year.notna().sum() >= MIN_VALID_HOURS
    eui = year.sum()[keep] / offices.loc[keep[keep].index, "sqm"]
    return eui.sort_index()


def test_top_five(raw, gold):
    eui = office_eui(raw)
    ordered = eui.sort_values(ascending=False, kind="mergesort")
    output = gold(TASK_ID)
    assert output["result"]["answer"] == sorted(ordered.index[:TOP_N])
    assert output["result"]["count"] == TOP_N
    assert output["evidence"]["cohort_size"] == len(eui)
    assert ordered.iloc[TOP_N - 1] > ordered.iloc[TOP_N]
    for entry in output["evidence"]["ranking"]:
        assert entry["building_id"] == ordered.index[entry["rank"] - 1]
        assert entry["eui"] == pytest.approx(ordered.iloc[entry["rank"] - 1], rel=RELATIVE, abs=ABSOLUTE)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
