"""Q01: buildings in London, checked against BDG2's metadata file."""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q01"
CITY = "London"


def test_matches_raw_bdg2_metadata(gold, raw):
    london = raw.metadata[raw.metadata["City"] == CITY]
    output = gold(TASK)
    assert output["result"]["answer"] == len(london)
    assert output["evidence"]["buildings_by_site"] == london["site_id"].value_counts().sort_index().to_dict()


def test_london_sites_are_mouse_robin_shrew(raw):
    assert set(raw.metadata.loc[raw.metadata["City"] == CITY, "site_id"]) == {"Mouse", "Robin", "Shrew"}


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
