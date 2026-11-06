"""Q42: Fox_office_Alice's 2017 readings above mean + 2 sd (ddof 1), recomputed with pandas from electricity_cleaned.csv."""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q42"
BUILDING = "Fox_office_Alice"
SIGMA = 2.0


def readings(raw) -> pd.Series:
    column = raw.electricity_cleaned[BUILDING]
    inside = (column.index >= pd.Timestamp("2017-01-01")) & (column.index < pd.Timestamp("2018-01-01"))
    return column[inside].dropna().astype(float)


def test_matches_raw_bdg2_file(gold, raw):
    values = readings(raw)
    threshold = values.mean() + SIGMA * values.std(ddof=1)
    output = gold(TASK)
    assert output["result"]["answer"] == int((values > threshold).sum())
    evidence = output["evidence"]
    assert evidence["valid_hours"] == len(values)
    assert evidence["mean_kwh"] == pytest.approx(values.mean(), rel=1e-6, abs=1e-6)
    assert evidence["sd_kwh"] == pytest.approx(values.std(ddof=1), rel=1e-6, abs=1e-6)
    assert evidence["threshold_kwh"] == pytest.approx(threshold, rel=1e-6, abs=1e-6)


def test_no_reading_sits_at_the_threshold(raw):
    values = readings(raw)
    threshold = values.mean() + SIGMA * values.std(ddof=1)
    assert (values - threshold).abs().min() > 1e-3


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
