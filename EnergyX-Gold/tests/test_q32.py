"""Q32: the site with the most Education buildings, recounted from BDG2_metadata.csv with pandas."""

from __future__ import annotations

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q32"


def counts(raw):
    meta = raw.metadata
    education = meta[meta["primaryspaceusage"] == "Education"]
    return education.groupby("site_id").size()


def test_matches_raw_bdg2_file(gold, raw):
    by_site = counts(raw)
    ranked = sorted(by_site.items(), key=lambda item: (-item[1], item[0]))
    output = gold(TASK)
    assert output["result"]["answer"] == ranked[0][0]
    assert output["evidence"]["education_buildings_by_site"] == {k: int(v) for k, v in by_site.items()}


def test_winner_is_not_tied(raw):
    values = sorted(counts(raw).values, reverse=True)
    assert values[0] > values[1]


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
