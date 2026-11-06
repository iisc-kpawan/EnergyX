"""Q31: buildings per country.

Recounted from BDG2's own BDG2_metadata.csv with pandas. Its Country column holds US state names
for most US sites, "US" for Washington DC and "Anonymous" for the five anonymous sites, so the test
maps those labels explicitly and checks the mapping against the snapshot's sites table.
"""

from __future__ import annotations

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q31"
NO_COUNTRY = "no country recorded"
US_LABELS = {"US", "California", "Texas", "New York", "Arizona", "Florida", "New Jersey"}


def canonical(label: str) -> str:
    if label in US_LABELS:
        return "US"
    return NO_COUNTRY if label == "Anonymous" else label


def test_mapping_agrees_with_the_sites_table(raw, context):
    mapped = raw.metadata.groupby("site_id")["Country"].first().map(canonical)
    sites = context.sql("SELECT site_id, country_canonical FROM sites").set_index("site_id")["country_canonical"]
    assert mapped.to_dict() == sites.fillna(NO_COUNTRY).to_dict()


def test_matches_raw_bdg2_file(gold, raw):
    expected = raw.metadata["Country"].map(canonical).value_counts().to_dict()
    answer = gold(TASK)["result"]["answer"]
    assert answer == {key: int(value) for key, value in expected.items()}
    assert sum(answer.values()) == len(raw.metadata) == 1636


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
