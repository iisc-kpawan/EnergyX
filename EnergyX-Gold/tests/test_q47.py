"""Q47: the five office buildings whose normalized 2017 hourly profile is closest to Eagle_office_Lamont's."""

from __future__ import annotations

import numpy as np
import pytest

TASK = "Q47"
TARGET = "Eagle_office_Lamont"


def test_matches_raw_bdg2_file(gold, raw):
    meta = raw.metadata
    frame = raw.electricity_cleaned
    year = frame.loc[(frame.index >= "2017-01-01") & (frame.index < "2018-01-01")]
    assert len(year) == 8760
    offices = [b for b in meta.loc[meta["primaryspaceusage"] == "Office", "building_id"] if b in year.columns]
    valid = year[offices].notna().sum()
    candidates = sorted(b for b in offices if b != TARGET and valid[b] >= 0.9 * 8760)

    means = year[[TARGET, *candidates]].groupby(year.index.hour).mean().T
    assert not means.isna().any().any()
    low, high = means.min(axis=1), means.max(axis=1)
    scaled = means.sub(low, axis=0).div(high - low, axis=0).fillna(0.0)

    target = scaled.loc[TARGET].to_numpy()
    distance = {b: float(np.linalg.norm(scaled.loc[b].to_numpy() - target)) for b in candidates}
    ranking = sorted(distance, key=lambda b: (distance[b], b))

    output = gold(TASK)
    assert output["result"]["answer"] == sorted(ranking[:5])
    assert output["result"]["count"] == 5
    evidence = output["evidence"]
    assert evidence["candidates"] == len(candidates)
    for row, building in zip(evidence["euclidean_ranking"], ranking):
        assert row["building_id"] == building
        assert row["distance"] == pytest.approx(distance[building], rel=1e-6, abs=1e-6)
