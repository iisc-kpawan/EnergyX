"""Q08: education buildings with an inner gap longer than 168 hours, from BDG2's own files."""

from __future__ import annotations

from _runs_q08_q10 import inner_gaps, runs

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q08"
MAX_GAP_HOURS = 168


def education(raw):
    meta = raw.metadata
    buildings = set(meta.loc[meta["primaryspaceusage"] == "Education", "building_id"])
    return [b for b in raw.electricity_cleaned.columns if b in buildings]


def test_matches_raw_bdg2_files(gold, raw):
    frame = raw.electricity_cleaned
    expected = {}
    for b in education(raw):
        gaps = inner_gaps(frame[b].notna().to_numpy())
        if len(gaps) and gaps.max() > MAX_GAP_HOURS:
            expected[b] = int(gaps.max())
    output = gold(TASK)
    assert set(output["result"]["answer"]) == set(expected)
    assert output["evidence"]["longest_gap_hours_by_member"] == expected


def test_edge_runs_alternative(gold, raw):
    frame = raw.electricity_cleaned
    with_edges = set()
    for b in education(raw):
        missing = frame[b].isna().to_numpy()
        if missing.any() and runs(missing).max() > MAX_GAP_HOURS:
            with_edges.add(b)
    output = gold(TASK)
    added = with_edges - set(output["result"]["answer"])
    assert sorted(added) == output["evidence"]["added_if_edge_runs_counted"]


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
