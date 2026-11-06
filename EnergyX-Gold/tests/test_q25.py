"""Independent check of Q25: median 2017 EUI of newer and older US education buildings, from BDG2's own files.

BDG2's Country column mixes countries and US states, so it is mapped explicitly here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q25"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
MIN_VALID_HOURS = 0.90 * 8760
NEWER_FROM_YEAR = 2007
US_VALUES = {"US", "Arizona", "California", "Florida", "New Jersey", "New York", "Texas"}
NON_US_VALUES = {"UK", "Ireland", "Canada", "Anonymous"}
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def us_education(raw) -> pd.DataFrame:
    meta = raw.metadata.set_index("building_id")
    assert set(meta["Country"].dropna()) <= US_VALUES | NON_US_VALUES
    meta = meta[meta["Country"].isin(US_VALUES) & (meta["primaryspaceusage"] == "Education")
                & meta["sqm"].notna() & (meta["sqm"] > 0)]
    readings = raw.electricity_cleaned
    year = readings[(readings.index >= YEAR_START) & (readings.index < YEAR_END)]
    buildings = [b for b in year.columns if b in meta.index]
    year = year[buildings]
    frame = pd.DataFrame({"eui": year.sum() / meta.loc[buildings, "sqm"], "hours": year.notna().sum(),
                          "yearbuilt": meta.loc[buildings, "yearbuilt"]})
    return frame[frame["hours"] >= MIN_VALID_HOURS]


def test_medians(raw, gold):
    frame = us_education(raw)
    dated = frame[frame["yearbuilt"].notna()]
    newer = dated.loc[dated["yearbuilt"] >= NEWER_FROM_YEAR, "eui"]
    older = dated.loc[dated["yearbuilt"] < NEWER_FROM_YEAR, "eui"]
    answer = gold(TASK_ID)["result"]["answer"]
    assert answer["newer_median_eui"] == pytest.approx(newer.median(), rel=RELATIVE, abs=ABSOLUTE)
    assert answer["older_median_eui"] == pytest.approx(older.median(), rel=RELATIVE, abs=ABSOLUTE)
    assert answer["difference"] == pytest.approx(newer.median() - older.median(), rel=RELATIVE, abs=ABSOLUTE)
    assert answer["ratio"] == pytest.approx(newer.median() / older.median(), rel=RELATIVE, abs=ABSOLUTE)
    assert answer["newer_buildings"] == len(newer)
    assert answer["older_buildings"] == len(older)
    assert answer["higher_median_group"] == ("newer" if newer.median() > older.median() else "older")
    assert gold(TASK_ID)["evidence"]["left_out_no_year_built"] == int(frame["yearbuilt"].isna().sum())


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
