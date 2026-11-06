"""Q35: the most similar pair of cities in different countries by 2017 office profile.

Recomputed from BDG2's electricity_cleaned.csv and BDG2_metadata.csv with pandas. City and country
labels are mapped to the site metadata's spelling ("Ottawa, Ontario" is Ottawa, US state names are
US, "Anonymous" is neither).
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q35"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
HOURS_2017 = 8760
US_LABELS = {"US", "California", "Texas", "New York", "Arizona", "Florida", "New Jersey"}
CITY_LABELS = {"Ottawa, Ontario": "Ottawa", "Anonymous": None}


def country(label):
    if label in US_LABELS:
        return "US"
    return None if label == "Anonymous" else label


def correlations(raw) -> dict[str, float]:
    meta = raw.metadata.set_index("building_id")
    offices = meta.index[meta["primaryspaceusage"] == "Office"]
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), [b for b in offices if b in frame.columns]]
    valid = year.notna().sum()
    year = year[valid[valid >= 0.9 * HOURS_2017].index]
    weekday = year[year.index.dayofweek <= 4]
    profile = weekday.groupby(weekday.index.hour).mean().T
    low, high = profile.min(axis=1), profile.max(axis=1)
    span = high - low
    scaled = profile.sub(low, axis=0).div(span.where(span > 0), axis=0).fillna(0.0)
    scaled["city"] = meta.loc[scaled.index, "City"].map(lambda c: CITY_LABELS.get(c, c))
    scaled["country"] = meta.loc[scaled.index, "Country"].map(country)
    scaled = scaled[scaled["city"].notna()]
    size = scaled.groupby("city").size()
    scaled = scaled[scaled["city"].isin(size[size >= 3].index)]
    profiles = scaled.groupby("city")[list(range(24))].mean()
    nation = scaled.groupby("city")["country"].first()
    out = {}
    for a, b in itertools.combinations(sorted(profiles.index), 2):
        if nation[a] != nation[b]:
            out[f"{a}|{b}"] = float(pd.Series(profiles.loc[a].values).corr(pd.Series(profiles.loc[b].values)))
    return out


def test_matches_raw_bdg2_files(gold, raw):
    expected = correlations(raw)
    output = gold(TASK)
    best = sorted(expected.items(), key=lambda item: (-item[1], item[0]))[0][0]
    assert output["result"]["answer"] == best
    recorded = output["evidence"]["correlation_by_pair"]
    assert set(recorded) == set(expected)
    for pair, value in expected.items():
        assert recorded[pair] == pytest.approx(value, rel=1e-6, abs=1e-6)


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
