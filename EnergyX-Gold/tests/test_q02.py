"""Q02: building-use labels at US sites, checked against BDG2's metadata file."""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q02"
# The metadata's Country column holds the state for most US sites and "US" for Washington DC.
US_VALUES = {"US", "Arizona", "California", "Florida", "New Jersey", "New York", "Texas"}
OTHER_VALUES = {"UK", "Ireland", "Canada", "Anonymous"}


def test_country_values_are_all_mapped(raw):
    assert set(raw.metadata["Country"]) == US_VALUES | OTHER_VALUES


def test_matches_raw_bdg2_metadata(gold, raw):
    us = raw.metadata[raw.metadata["Country"].isin(US_VALUES)]
    labels = us["primaryspaceusage"].dropna()
    output = gold(TASK)
    assert output["result"]["answer"] == sorted(set(labels))
    assert output["result"]["count"] == labels.nunique()
    assert output["evidence"]["buildings_at_us_sites"] == len(us)
    assert output["evidence"]["buildings_by_type"] == labels.value_counts().to_dict()


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
