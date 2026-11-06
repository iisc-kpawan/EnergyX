"""Q07: missing-hour percentage for Peacock_office_Annie, from BDG2's cleaned file."""

from __future__ import annotations

import pytest

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q07"
BUILDING = "Peacock_office_Annie"


def test_matches_raw_bdg2_file(gold, raw):
    series = raw.electricity_cleaned[BUILDING]
    valid = series.dropna()
    first, last = valid.index.min(), valid.index.max()
    hours = int((last - first).total_seconds() // 3600) + 1
    missing = hours - len(valid)
    output = gold(TASK)
    assert output["result"]["answer"] == pytest.approx(round(100 * missing / hours, 6), rel=1e-6, abs=5e-7)
    assert output["evidence"]["missing_hours"] == missing
    assert output["evidence"]["observed_hours"] == hours


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
