"""Q05: buildings with one year of 80% electricity and 80% site temperature coverage, from BDG2's own files."""

from __future__ import annotations

from _coverage_q04_q05 import coverage

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q05"
YEAR_HOURS = {2016: 8784, 2017: 8760}


def test_matches_raw_bdg2_files(gold, raw):
    meta = raw.metadata.set_index("building_id")
    with_area = set(meta[meta["sqm"] > 0].index)
    members = set()
    by_year = {}
    for year, hours in YEAR_HOURS.items():
        cov = coverage(raw, year)
        ok = (cov["elec_hours"] >= 0.8 * hours) & (cov["site_temp_hours"] >= 0.8 * hours)
        by_year[year] = set(cov[ok].index) & with_area
        members |= by_year[year]
    output = gold(TASK)
    assert set(output["result"]["answer"]) == members
    assert output["evidence"]["members_qualifying_in_2016"] == len(by_year[2016])
    assert output["evidence"]["members_qualifying_in_2017"] == len(by_year[2017])


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
