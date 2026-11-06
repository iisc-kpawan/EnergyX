"""Independent check of Q26: mean July 2017 air temperature over the London sites, from BDG2's own files."""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK_ID = "Q26"
MONTH_START = pd.Timestamp("2017-07-01")
MONTH_END = pd.Timestamp("2017-08-01")
RELATIVE = 1e-6
ABSOLUTE = 5e-7


def test_mean(raw, gold):
    meta = raw.metadata
    sites = sorted(meta.loc[meta["City"] == "London", "site_id"].unique())
    assert sites == ["Mouse", "Robin", "Shrew"]
    weather = raw.weather
    july = weather[weather["site_id"].isin(sites) & (weather["timestamp"] >= MONTH_START)
                   & (weather["timestamp"] < MONTH_END)]
    assert not july.duplicated(["site_id", "timestamp"]).any()
    temps = july["airTemperature"].dropna()

    output = gold(TASK_ID)
    assert output["result"]["answer"] == pytest.approx(temps.mean(), rel=RELATIVE, abs=ABSOLUTE)
    assert output["evidence"]["valid_site_hours"] == len(temps)
    for site, group in july.groupby("site_id"):
        assert output["evidence"]["per_site"][site]["mean_c"] == pytest.approx(
            group["airTemperature"].mean(), rel=RELATIVE, abs=ABSOLUTE)


def test_deterministic(context):
    task = get_task(TASK_ID)
    assert output_hash(run_task(task, context)) == output_hash(run_task(task, context))
