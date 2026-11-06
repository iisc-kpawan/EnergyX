"""Q04: office buildings eligible for weather normalization in 2016, from BDG2's own files."""

from __future__ import annotations

from _coverage_q04_q05 import coverage

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q04"
YEAR, YEAR_HOURS = 2016, 8784


def test_matches_raw_bdg2_files(gold, raw):
    meta = raw.metadata.set_index("building_id")
    offices = meta[(meta["primaryspaceusage"] == "Office") & (meta["sqm"] > 0)].index
    cov = coverage(raw, YEAR)
    cov = cov[cov.index.isin(offices)]
    elec_ok = cov["elec_hours"] >= 0.95 * YEAR_HOURS
    joint_ok = cov["elec_hours_with_temp"] >= 0.95 * cov["elec_hours"]
    site_ok = cov["site_temp_hours"] >= 0.95 * YEAR_HOURS
    output = gold(TASK)
    assert set(output["result"]["answer"]) == set(cov[elec_ok & joint_ok].index)
    assert output["evidence"]["alternative_site_temperature_over_year_hours"]["count"] == int((elec_ok & site_ok).sum())


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
