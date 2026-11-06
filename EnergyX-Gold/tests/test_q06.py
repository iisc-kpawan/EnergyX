"""Q06: duplicated timestamps for Bear_education_Alfredo, from BDG2's electricity file."""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.config import load_config
from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import output_hash, run_task

TASK = "Q06"
BUILDING = "Bear_education_Alfredo"


def test_matches_raw_bdg2_file(gold, raw):
    # Read the CSV's timestamp strings directly, so duplicates can't be lost in parsing or caching.
    source = load_config().bdg2_source_dir / "electricity.csv"
    frame = pd.read_csv(source, usecols=["timestamp", BUILDING], dtype={"timestamp": str})
    present = frame[frame[BUILDING].notna()]
    duplicated = present["timestamp"].value_counts()
    output = gold(TASK)
    assert output["result"]["answer"] == int((duplicated > 1).sum())
    assert output["evidence"]["rows"] == len(present)


def test_answer_is_zero(gold):
    assert gold(TASK)["result"]["answer"] == 0


def test_deterministic(gold):
    assert output_hash(run_task(get_task(TASK), GoldContext())) == output_hash(gold(TASK))
