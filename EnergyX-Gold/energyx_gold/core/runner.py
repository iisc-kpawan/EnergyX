"""Run task modules and write outputs/gold_results.jsonl."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import provenance, registry
from .config import REPO_ROOT
from .context import GoldContext
from .loader import get_task
from .schemas import Task, check_output

RESULTS = REPO_ROOT / "outputs" / "gold_results.jsonl"
HASH_DIGITS = 10  # significant digits kept for floats when hashing


class TaskError(RuntimeError):
    pass


def run_task(task: Task, context: GoldContext) -> dict:
    """Run the task's main(context), check its output, and add provenance."""
    main = registry.load_main(task)
    context.queries_run.clear()
    output = main(context)
    problems = check_output(task, output)
    if problems:
        raise TaskError(f"{task.task_id}: " + "; ".join(problems))
    output["provenance"] = provenance.collect(context.config)
    return output


def canonical(output: dict) -> str:
    return json.dumps(output, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _round_floats(value):
    if isinstance(value, float):
        return float(f"{value:.{HASH_DIGITS}g}")
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def output_hash(output: dict) -> str:
    """SHA-256 of the output without provenance, with floats rounded, for the determinism check."""
    body = {k: v for k, v in output.items() if k != "provenance"}
    return hashlib.sha256(canonical(_round_floats(body)).encode()).hexdigest()


def write_results(outputs: list[dict], path: Path | None = None) -> Path:
    """Merge outputs into gold_results.jsonl, one line per task, sorted by task_id."""
    path = path or RESULTS
    rows = {}
    if path.exists():
        rows = {json.loads(line)["task_id"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}
    for output in outputs:
        rows[output["task_id"]] = json.loads(canonical(output))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(rows[k]) + "\n" for k in sorted(rows)))
    return path


def run_standalone(task_id: str) -> None:
    """What `python -m energyx_gold.tasks.<category>.<TASK>` does: run one task and print its output."""
    output = run_task(get_task(task_id), GoldContext())
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
