"""Match manifest tasks to their reference modules."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from .schemas import Task

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
CATEGORY_MODULES = {
    1: "discovery", 2: "quality", 3: "statistics", 4: "temporal", 5: "performance", 6: "weather",
    7: "spatial", 8: "forecasting", 9: "anomaly", 10: "clustering",
}


def module_name(task_id: str, category: int) -> str:
    return f"energyx_gold.tasks.category_{category:02d}_{CATEGORY_MODULES[category]}.{task_id.replace('-', '_')}"


def load_main(task: Task) -> Callable[..., dict]:
    module = importlib.import_module(task.reference_module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError(f"{task.reference_module} has no main(context)")
    return main


def task_files() -> dict[str, Path]:
    """Task modules on disk, by task ID (Q01.py is Q01). Files starting with _ are helpers."""
    return {path.stem.replace("_", "-"): path
            for path in sorted(TASKS_DIR.glob("category_*/*.py")) if not path.stem.startswith("_")}


def check(tasks: list[Task], all_tasks: list[Task] | None = None) -> list[str]:
    """Problems linking the manifest and the task modules, for `tasks`.

    Pass the whole manifest as `all_tasks` to also report modules with no manifest line.
    A manifest task with no module yet isn't a problem.
    """
    problems = []
    files = task_files()
    if all_tasks is not None:
        for task_id in sorted(set(files) - {task.task_id for task in all_tasks}):
            problems.append(f"{files[task_id].name} has no manifest line")
    for task in tasks:
        expected = module_name(task.task_id, task.category)
        if task.reference_module != expected:
            problems.append(f"{task.task_id}: reference_module is {task.reference_module}, expected {expected}")
        if task.task_id not in files:
            continue
        try:
            load_main(task)
        except Exception as err:  # an import error is exactly what this check reports
            problems.append(f"{task.task_id}: {type(err).__name__}: {err}")
    return problems
