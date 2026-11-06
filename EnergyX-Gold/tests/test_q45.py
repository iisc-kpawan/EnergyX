"""Q45: Isolation Forest anomalies at Berkeley's (Bear) education buildings in May 2016.

Recomputed with pandas from electricity_cleaned.csv and BDG2_metadata.csv, calling scikit-learn
directly with the same settings. Berkeley is the metadata's City "Berkeley", checked against the
snapshot's sites table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q45"


def test_berkeley_is_bear(raw, context):
    meta = raw.metadata
    assert sorted(set(meta.loc[meta["City"] == "Berkeley", "site_id"])) == ["Bear"]
    assert context.sql("SELECT site_id FROM sites WHERE city = 'Berkeley'")["site_id"].tolist() == ["Bear"]


@pytest.fixture(scope="module")
def counts(raw) -> dict[str, int]:
    meta = raw.metadata
    cohort = meta.loc[(meta["City"] == "Berkeley") & (meta["primaryspaceusage"] == "Education"), "building_id"]
    frame = raw.electricity_cleaned
    may = frame.loc[(frame.index >= pd.Timestamp("2016-05-01")) & (frame.index < pd.Timestamp("2016-06-01"))]
    out = {}
    for building in sorted(cohort):
        if building not in may.columns:
            continue
        values = may[building].dropna().astype(float)
        if values.empty:
            continue
        x = np.column_stack([values.to_numpy(float), values.index.hour.to_numpy(float),
                             values.index.dayofweek.to_numpy(float)])
        labels = IsolationForest(contamination="auto", random_state=0).fit_predict(x)
        out[building] = int((labels == -1).sum())
    return out


def test_matches_raw_bdg2_files(gold, counts):
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    output = gold(TASK)
    assert output["result"]["answer"] == sorted(name for name, _ in ranked[:5])
    assert {b: v["anomalies"] for b, v in output["evidence"]["per_building"].items()} == counts


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
