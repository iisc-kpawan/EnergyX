"""Q33: the city with the highest median 2017 office EUI.

Recomputed from BDG2's electricity_cleaned.csv and BDG2_metadata.csv with pandas. BDG2's City
column writes "Ottawa, Ontario" and "Anonymous"; the test maps those to the site metadata's
"Ottawa" and no city, and checks the mapping against the snapshot's sites table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q33"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
HOURS_2017 = 8760
CITY_LABELS = {"Ottawa, Ontario": "Ottawa", "Anonymous": None}


def city(label):
    return CITY_LABELS.get(label, label)


def test_city_mapping_agrees_with_the_sites_table(raw, context):
    mapped = raw.metadata.groupby("site_id")["City"].first().map(city)
    sites = context.sql("SELECT site_id, city FROM sites").set_index("site_id")["city"]
    assert mapped.to_dict() == sites.where(sites.notna(), None).to_dict()


def city_medians(raw) -> tuple[pd.Series, pd.Series]:
    meta = raw.metadata.set_index("building_id")
    offices = meta.index[meta["primaryspaceusage"] == "Office"]
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), [b for b in offices if b in frame.columns]]
    valid = year.notna().sum()
    kept = valid[valid >= 0.9 * HOURS_2017].index
    table = pd.DataFrame({"kwh": year[kept].sum(), "sqm": meta.loc[kept, "sqm"]})
    table = table[table["sqm"] > 0]
    table["eui"] = table["kwh"] / table["sqm"]
    table["city"] = meta.loc[table.index, "City"].map(city)
    table = table[table["city"].notna()]
    size = table.groupby("city").size()
    medians = table.groupby("city")["eui"].median()
    return medians[size >= 5], size[size >= 5]


def test_matches_raw_bdg2_files(gold, raw):
    medians, size = city_medians(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == medians.idxmax()
    recorded = output["evidence"]["median_eui_by_city"]
    assert set(recorded) == set(medians.index)
    for name in medians.index:
        assert recorded[name]["offices"] == int(size[name])
        assert recorded[name]["median_eui_kwh_per_m2"] == pytest.approx(medians[name], rel=1e-6, abs=1e-6)


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
