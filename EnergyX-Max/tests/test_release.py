"""Checks on the release files in outputs/, once the pipeline has produced them."""

from __future__ import annotations

import pytest

from energyx_max.common import CATEGORIES, LEVELS, ROOT, read_jsonl
from energyx_max.config import REPO_ROOT

ROWS = read_jsonl(ROOT / "EnergyX-Max.jsonl")
pytestmark = pytest.mark.skipif(not ROWS, reason="outputs/EnergyX-Max.jsonl not built yet")


def test_ids_and_texts_are_unique():
    assert len({r["question_id"] for r in ROWS}) == len(ROWS)
    assert len({r["question"] for r in ROWS}) == len(ROWS)


def test_fields_are_valid():
    for row in ROWS:
        assert row["category"] in CATEGORIES.values()
        assert row["difficulty_level"] in LEVELS
        assert (REPO_ROOT / row["program"]).exists()
        assert row["validation"] == "automatic checks only; not manually validated"


def test_no_gold_question_is_included():
    gold = {r["question"].lower().strip() for r in read_jsonl(REPO_ROOT / "data/gold/EnergyX-Gold-questions.jsonl")}
    assert not gold & {r["question"].lower().strip() for r in ROWS}
