"""Q44: day-over-day increases above 20% for Cockatoo_office_Alton, recomputed with pandas from electricity_cleaned.csv."""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q44"
BUILDING = "Cockatoo_office_Alton"


def increase_dates(raw) -> list[str]:
    column = raw.electricity_cleaned[BUILDING]
    column = column[(column.index >= pd.Timestamp("2016-01-01")) & (column.index < pd.Timestamp("2018-01-01"))]
    daily = column.resample("D").agg(["sum", "count"])
    total = daily["sum"].where(daily["count"] == 24)
    total = total.reindex(pd.date_range("2016-01-01", "2017-12-31", freq="D"))
    previous = total.shift(1)
    hits = total.index[(total > 1.2 * previous).to_numpy()]
    return [d.strftime("%Y-%m-%d") for d in hits]


def test_matches_raw_bdg2_file(gold, raw):
    dates = increase_dates(raw)
    output = gold(TASK)
    assert output["result"]["answer"] == len(dates)
    assert output["evidence"]["increase_dates"] == dates


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
