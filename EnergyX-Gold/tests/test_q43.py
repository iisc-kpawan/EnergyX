"""Q43: IQR outliers of London office buildings in 2017.

Recomputed with pandas from electricity_cleaned.csv and BDG2_metadata.csv. London is the metadata's
City "London" (Mouse, Robin and Shrew), checked against the snapshot's sites table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q43"


def test_london_sites_agree_with_the_sites_table(raw, context):
    meta = raw.metadata
    raw_sites = sorted(set(meta.loc[meta["City"] == "London", "site_id"]))
    sites = context.sql("SELECT site_id FROM sites WHERE city = 'London' ORDER BY 1")["site_id"].tolist()
    assert raw_sites == sites == ["Mouse", "Robin", "Shrew"]


def expected(raw) -> tuple[dict, int]:
    meta = raw.metadata
    offices = meta.loc[(meta["City"] == "London") & (meta["primaryspaceusage"] == "Office"), "building_id"]
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= pd.Timestamp("2017-01-01")) & (frame.index < pd.Timestamp("2018-01-01"))]
    per_building, days = {}, set()
    for building in sorted(offices):
        if building not in year.columns:
            continue
        values = year[building].dropna()
        if values.empty:
            continue
        q1, q3 = np.percentile(values.to_numpy(float), [25, 75])
        iqr = q3 - q1
        outside = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
        per_building[building] = int(len(outside))
        days |= set(outside.index.date)
    return per_building, len(days)


def test_matches_raw_bdg2_files(gold, raw):
    per_building, days = expected(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == {"outlier_readings": sum(per_building.values()), "days_with_outlier": days}
    recorded = output["evidence"]["per_building"]
    assert {b: v["outliers"] for b, v in recorded.items()} == per_building


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
