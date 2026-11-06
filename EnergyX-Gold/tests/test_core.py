"""The shared pieces: schemas, registry, hashing and data access."""

from __future__ import annotations

import pytest

from energyx_gold.core import registry
from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task, load_tasks
from energyx_gold.core.runner import output_hash
from energyx_gold.core.schemas import check_output


def output(result: dict, **extra) -> dict:
    return {"task_id": "Q03", "status": "success", "result": result, "evidence": {}, **extra}


@pytest.fixture(scope="module")
def set_task():
    return get_task("Q03")


def test_manifest_and_modules_match():
    tasks = load_tasks()
    assert registry.check(tasks, tasks) == []


def test_set_result_must_be_sorted_and_counted(set_task):
    assert check_output(set_task, output({"answer": ["a", "b"], "count": 2})) == []
    assert check_output(set_task, output({"answer": ["b", "a"], "count": 2}))
    assert check_output(set_task, output({"answer": ["a", "b"], "count": 3}))


def test_result_must_fit_the_answer_type(set_task):
    assert check_output(set_task, output({"answer": 628}))


def test_alternates_are_checked(set_task):
    good = {"reading": "a 365-day stretch with no gap over 24 hours", "result": {"answer": ["a"], "count": 1}}
    bad = {"reading": "unsorted", "result": {"answer": ["b", "a"], "count": 2}}
    base = {"answer": ["a", "b"], "count": 2}
    assert check_output(set_task, output(base, alternates=[good])) == []
    assert check_output(set_task, output(base, alternates=[bad]))


def test_refusal_needs_evidence_of_absence(set_task):
    refusal = set_task.model_copy(update={"answer_type": "refusal"})
    result = {"refused": True, "predicate": "temporal_resolution = PT15M"}
    absent = {"absence_check": {"query": "SELECT 1 FROM electricity_streams WHERE resolution = 'PT15M'", "rows": 0}}
    assert check_output(refusal, {**output(result), "evidence": absent}) == []
    assert check_output(refusal, output(result))


def test_hash_ignores_provenance_and_float_noise():
    a = output({"answer": 0.1 + 0.2, "unit": "kWh"}, provenance={"code_commit": "x"})
    b = output({"answer": 0.3, "unit": "kWh"}, provenance={"code_commit": "y"})
    assert output_hash(a) == output_hash(b)


def test_data_access_and_graph():
    context = GoldContext()
    london = context.get_metadata(city="London")
    assert len(london) > 0 and set(london["city"]) == {"London"}
    readings = context.get_measurements(london["building_id"].head(1).tolist())
    assert len(readings) > 0
    if not context.config.graph_dir.exists():
        pytest.skip("no knowledge graph in this snapshot")
    buildings = context.query_kg("SELECT (COUNT(?b) AS ?n) WHERE { ?b a ekg:Building }")
    assert buildings.loc[0, "n"] == context.sql("SELECT COUNT(*) AS n FROM buildings").loc[0, "n"]
