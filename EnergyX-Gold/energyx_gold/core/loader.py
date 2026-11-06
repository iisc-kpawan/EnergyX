"""Read and write the task manifest, data/task_manifest.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from .config import REPO_ROOT
from .schemas import Task

MANIFEST = REPO_ROOT / "data" / "task_manifest.jsonl"


def load_tasks(path: Path | None = None) -> list[Task]:
    """Every task in the manifest. A line that doesn't match the Task model raises."""
    path = path or MANIFEST
    if not path.exists():
        return []
    return [Task.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def get_task(task_id: str, path: Path | None = None) -> Task:
    for task in load_tasks(path):
        if task.task_id == task_id:
            return task
    raise KeyError(f"{task_id} is not in the task manifest")


def write_tasks(tasks: list[Task], path: Path | None = None) -> None:
    path = path or MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(tasks, key=lambda t: t.task_id)
    path.write_text("".join(json.dumps(t.model_dump(mode="json"), ensure_ascii=False) + "\n" for t in ordered))
