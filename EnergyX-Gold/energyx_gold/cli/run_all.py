"""Run every task that has a module and write outputs/gold_results.jsonl.

    python -m energyx_gold.cli.run_all
"""

from __future__ import annotations

from ..core import registry
from ..core.loader import load_tasks
from . import run_task


def main() -> None:
    implemented = registry.task_files()
    task_ids = [task.task_id for task in load_tasks() if task.task_id in implemented]
    run_task.main(task_ids)


if __name__ == "__main__":
    main()
