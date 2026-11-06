"""Independent check of Q23: median 2017 EUI of office and education buildings, from BDG2's own files."""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q23"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
MIN_VALID_HOURS = 0.90 * 8760
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def eui_by_label(raw) -> pd.DataFrame:
    readings = raw.electricity_cleaned
    year = readings[(readings.index >= YEAR_START) & (readings.index < YEAR_END)]
    meta = raw.metadata.set_index("building_id")
    meta = meta[meta["sqm"].notna() & (meta["sqm"] > 0)]
    buildings = [b for b in year.columns if b in meta.index]
    year = year[buildings]
    frame = pd.DataFrame({"kwh": year.sum(), "hours": year.notna().sum(),
                          "sqm": meta.loc[buildings, "sqm"], "label": meta.loc[buildings, "primaryspaceusage"]})
    frame = frame[frame["hours"] >= MIN_VALID_HOURS]
    frame["eui"] = frame["kwh"] / frame["sqm"]
    return frame


def test_medians(raw, gold):
    frame = eui_by_label(raw)
    office = frame.loc[frame["label"] == "Office", "eui"]
    education = frame.loc[frame["label"] == "Education", "eui"]
    expected = {
        "office_median_eui": office.median(),
        "education_median_eui": education.median(),
        "difference": office.median() - education.median(),
        "ratio": office.median() / education.median(),
    }
    answer = gold(TASK_ID)["result"]["answer"]
    for key, value in expected.items():
        assert answer[key] == pytest.approx(value, rel=RELATIVE, abs=ABSOLUTE), key
    assert answer["office_buildings"] == len(office)
    assert answer["education_buildings"] == len(education)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
