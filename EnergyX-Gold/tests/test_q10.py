"""Q10: hours filled for Peacock_office_Annie when gaps of up to six hours are interpolated."""

from __future__ import annotations

import numpy as np

from _runs_q08_q10 import inner_gaps

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q10"
BUILDING = "Peacock_office_Annie"
MAX_FILL = 6


def test_matches_raw_bdg2_file(gold, raw):
    gaps = inner_gaps(raw.electricity_cleaned[BUILDING].notna().to_numpy())
    output = gold(TASK)
    assert output["result"]["answer"] == int(gaps[gaps <= MAX_FILL].sum())
    assert output["evidence"]["partial_fill_reading_first_six_hours_of_every_gap"] == int(np.minimum(gaps, MAX_FILL).sum())


def test_interpolation_fills_the_same_hours(gold, raw):
    series = raw.electricity_cleaned[BUILDING]
    valid = series.dropna()
    series = series.loc[valid.index.min():valid.index.max()]
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    length = missing.groupby(run_id).transform("size")
    filled = series.interpolate(limit_area="inside").where(~missing | (length <= MAX_FILL))
    assert int((missing & filled.notna()).sum()) == gold(TASK)["result"]["answer"]


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
