"""Run one or more gold tasks and write their results.

    python -m energyx_gold.cli.run_task Q01 [Q02 ...]
"""

from __future__ import annotations

import argparse
import time

from ..core.context import GoldContext
from ..core.loader import get_task
from ..core.runner import run_task, write_results


def summarise(answer_type: str, result: dict) -> str:
    if answer_type == "set":
        head = ", ".join(result["answer"][:3])
        return f"{result['count']} members ({head}{', ...' if result['count'] > 3 else ''})"
    if answer_type == "refusal":
        return f"not answerable: {result['predicate']}"
    if answer_type == "record":
        return ", ".join(f"{k}={v}" for k, v in result["answer"].items() if not isinstance(v, (dict, list)))[:160]
    if answer_type == "method":
        return f"method ({len(result['method'].split())} words)"
    return str(result.get("answer"))


def main(task_ids: list[str] | None = None) -> None:
    if task_ids is None:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("task_ids", nargs="+")
        task_ids = parser.parse_args().task_ids
    context = GoldContext()
    outputs = []
    for task_id in task_ids:
        task = get_task(task_id)
        start = time.perf_counter()
        output = run_task(task, context)
        outputs.append(output)
        print(f"{task_id}: {summarise(task.answer_type, output['result'])} [{time.perf_counter() - start:.1f} s]")
    print(f"wrote {write_results(outputs)}")


if __name__ == "__main__":
    main()
