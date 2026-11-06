"""Q09: buildings with more than 72 consecutive zero hours, from BDG2's raw and cleaned files."""

from __future__ import annotations

import numpy as np

from _runs_q08_q10 import runs

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q09"
MAX_ZERO_RUN_HOURS = 72


def longest(mask: np.ndarray) -> int:
    lengths = runs(mask)
    return int(lengths.max()) if len(lengths) else 0


def test_hourly_index_is_complete(raw):
    index = raw.electricity.index
    assert len(index) == 17544 and index.is_monotonic_increasing and not index.duplicated().any()


def test_masked_hours_are_the_raw_zeros(raw):
    zero = raw.electricity.eq(0)
    assert (zero == (raw.electricity_cleaned.isna() & raw.electricity.notna())).all().all()


def test_matches_raw_bdg2_file(gold, raw):
    frame = raw.electricity
    clock, skipping = set(), set()
    for b in frame.columns:
        values = frame[b].to_numpy()
        if longest(values == 0) > MAX_ZERO_RUN_HOURS:  # NaN != 0, so a missing hour ends a run
            clock.add(b)
        observed = values[~np.isnan(values)]
        if longest(observed == 0) > MAX_ZERO_RUN_HOURS:
            skipping.add(b)
    output = gold(TASK)
    assert set(output["result"]["answer"]) == clock
    assert output["evidence"]["skipping_missing_hours_rule"]["count"] == len(skipping)


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
