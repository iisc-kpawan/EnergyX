"""Q34: normalized 2017 weekday office profiles by country.

Recomputed from BDG2's electricity_cleaned.csv and BDG2_metadata.csv with pandas. BDG2 writes US
state names in its Country column; the test maps them to US and "Anonymous" to no country, and
checks the mapping against the snapshot's sites table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q34"
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")
HOURS_2017 = 8760
NIGHT = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
DAY = list(range(7, 20))
US_LABELS = {"US", "California", "Texas", "New York", "Arizona", "Florida", "New Jersey"}


def country(label):
    if label in US_LABELS:
        return "US"
    return None if label == "Anonymous" else label


def test_mapping_agrees_with_the_sites_table(raw, context):
    mapped = raw.metadata.groupby("site_id")["Country"].first().map(country)
    sites = context.sql("SELECT site_id, country_canonical FROM sites").set_index("site_id")["country_canonical"]
    assert mapped.to_dict() == sites.where(sites.notna(), None).to_dict()


def scaled_profiles(raw) -> pd.DataFrame:
    meta = raw.metadata.set_index("building_id")
    offices = meta.index[meta["primaryspaceusage"] == "Office"]
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= START) & (frame.index < END), [b for b in offices if b in frame.columns]]
    valid = year.notna().sum()
    year = year[valid[valid >= 0.9 * HOURS_2017].index]
    weekday = year[year.index.dayofweek <= 4]
    profile = weekday.groupby(weekday.index.hour).mean().T
    assert profile.notna().all(axis=1).all()
    low, high = profile.min(axis=1), profile.max(axis=1)
    span = (high - low)
    scaled = profile.sub(low, axis=0).div(span.where(span > 0), axis=0).fillna(0.0)
    scaled["country"] = meta.loc[scaled.index, "Country"].map(country)
    return scaled


def test_matches_raw_bdg2_files(gold, raw):
    scaled = scaled_profiles(raw)
    named = scaled[scaled["country"].notna()]
    profiles = named.groupby("country")[list(range(24))].mean()
    counts = named.groupby("country").size()
    output = gold(TASK)
    answer = output["result"]["answer"]
    assert len(answer) == 3 * len(profiles)
    for name in profiles.index:
        profile = profiles.loc[name]
        assert answer[f"{name}_peak_hour"] == int(profile.idxmax())
        assert answer[f"{name}_night_to_day_ratio"] == pytest.approx(
            profile[NIGHT].mean() / profile[DAY].mean(), rel=1e-6, abs=1e-6)
        assert answer[f"{name}_office_buildings"] == int(counts[name])
        assert output["evidence"]["scaled_profile_by_country"][name] == pytest.approx(list(profile), rel=1e-6, abs=1e-6)


def test_deterministic(gold, context):
    assert output_hash(run_task(get_task(TASK), context)) == output_hash(gold(TASK))
