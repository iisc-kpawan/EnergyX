"""Checks for the parts of the pipeline that don't need an LLM."""

from __future__ import annotations

import random

import pytest

from energyx_max import bdg2, execute, slots
from energyx_max.common import CATEGORIES
from energyx_max.seeds import deduplicate, load


def test_sampled_slots_are_valid():
    rng = random.Random(0)
    meta = bdg2.metadata()
    for _ in range(20):
        fill = slots.sample(["city", "building_type", "year", "top_n"], rng)
        assert fill is not None
        ids = bdg2.buildings(city=fill["city"]["value"], building_type=fill["building_type"]["value"])
        assert len(bdg2.qualifying(ids, fill["year"]["value"])) >= slots.MIN_COHORT
    fill = slots.sample(["building_id", "month"], rng)
    assert fill["building_id"]["value"] in meta.index
    assert fill["month"]["text"][0].isupper()


def test_render_uses_slot_text():
    fill = {"building_type": {"value": "Office", "text": "office"}, "year": {"value": 2017, "text": "2017"}}
    assert slots.render("Which {building_type} buildings used most in {year}?", fill) == \
        "Which office buildings used most in 2017?"


@pytest.fixture(scope="module")
def workers():
    with execute.pool(1) as pool:
        yield pool


def run(workers, code, **slot_values):
    return workers.apply(execute.run, (code, slot_values))


def test_a_program_runs_on_the_data(workers):
    code = "def answer(data, building_id, year):\n    r = data.readings(building_id, *data.year_range(year))\n" \
           "    return {'answer': float(r.sum().iloc[0]), 'unit': 'kWh'}\n"
    result = run(workers, code, building_id="Fox_office_Alice", year=2017)
    assert result["ok"] and result["answer"] > 0


def test_unusable_answers_are_rejected(workers):
    assert not run(workers, "def answer(data):\n    return {'answer': float('nan')}\n")["ok"]
    assert not run(workers, "def answer(data):\n    return {'answer': []}\n")["ok"]
    assert not run(workers, "def answer(data):\n    raise ValueError('no data')\n")["ok"]


def test_seeds_map_to_the_ten_categories():
    kept = deduplicate(load())
    assert len(kept) > 400
    assert {s["category"] for s in kept} <= set(CATEGORIES) | {None}
