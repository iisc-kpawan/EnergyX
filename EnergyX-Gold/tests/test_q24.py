"""Independent check of Q24: buildings above their primary-use label's 90th percentile of 2017 EUI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q24"
YEAR_START = pd.Timestamp("2017-01-01")
YEAR_END = pd.Timestamp("2018-01-01")
MIN_VALID_HOURS = 0.90 * 8760
MIN_LABEL_SIZE = 10


def labelled_eui(raw) -> pd.DataFrame:
    readings = raw.electricity_cleaned
    year = readings[(readings.index >= YEAR_START) & (readings.index < YEAR_END)]
    meta = raw.metadata.set_index("building_id")
    meta = meta[meta["sqm"].notna() & (meta["sqm"] > 0) & meta["primaryspaceusage"].notna()]
    buildings = [b for b in year.columns if b in meta.index]
    year = year[buildings]
    frame = pd.DataFrame({"eui": year.sum() / meta.loc[buildings, "sqm"], "hours": year.notna().sum(),
                          "label": meta.loc[buildings, "primaryspaceusage"]})
    return frame[frame["hours"] >= MIN_VALID_HOURS]


def test_members(raw, gold):
    frame = labelled_eui(raw)
    expected, left_out = [], []
    for label, group in frame.groupby("label"):
        if len(group) < MIN_LABEL_SIZE:
            left_out.append(label)
            continue
        threshold = np.percentile(group["eui"].to_numpy(), 90)  # numpy's default linear method
        expected.extend(group.index[group["eui"] > threshold])

    output = gold(TASK_ID)
    assert output["result"]["answer"] == sorted(expected)
    assert output["result"]["count"] == len(expected)
    assert output["evidence"]["labels_left_out"] == sorted(left_out)
    assert output["evidence"]["cohort_size"] == len(frame)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
