"""Q03: buildings with a 12-calendar-month window at least 95% valid, from BDG2's cleaned file."""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q03"
MONTHS = pd.period_range("2016-01", "2017-12", freq="M")
WINDOW = 12
STARTS = 13


def monthly(raw):
    frame = raw.electricity_cleaned
    frame = frame[(frame.index >= "2016-01-01") & (frame.index < "2018-01-01")]
    valid = frame.notna().groupby(frame.index.to_period("M")).sum().reindex(MONTHS, fill_value=0)
    hours = pd.Series([p.days_in_month * 24 for p in MONTHS], index=MONTHS)
    return valid, hours


def test_matches_raw_bdg2_file(gold, raw):
    valid, hours = monthly(raw)
    members, strict = set(), set()
    for k in range(STARTS):
        window = MONTHS[k:k + WINDOW]
        share = valid.loc[window].sum() / hours.loc[window].sum()
        members |= set(share[share >= 0.95 - 1e-12].index)
        every = (valid.loc[window].div(hours.loc[window], axis=0) >= 0.95 - 1e-12).all()
        strict |= set(every[every].index)
    output = gold(TASK)
    assert set(output["result"]["answer"]) == members
    assert output["evidence"]["strict_reading_every_month_95pct"]["count"] == len(strict)


def test_window_hours(gold):
    hours = gold(TASK)["evidence"]["definition"]["window_hours"]
    assert hours["2016-01"] == 8784 and hours["2016-03"] == 8760 and len(hours) == STARTS


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
