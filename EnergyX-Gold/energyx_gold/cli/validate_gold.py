"""Check the gold tasks and write outputs/validation_report.json.

    python -m energyx_gold.cli.validate_gold [Q01 ...]

Checks:
- every task module has a manifest line, and every reference_module imports a main(context);
- each task runs, its output is well formed, and two runs give the same output;
- the data its required_data names exists in the snapshot, and its queries use no other views;
- how its answer compares with the question set's draft answer, when it has one.

Exits with status 1 on a missing module, a run failure, a malformed or changing output, or
missing data. Undeclared views and a difference from the draft are reported without failing.
With task IDs, only those tasks' entries in the report are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

from ..core import registry
from ..core.config import REPO_ROOT
from ..core.context import GoldContext
from ..core.loader import load_tasks
from ..core.runner import output_hash, run_task
from ..core.schemas import Task
from ..data_access.sql import REQUIRED_DATA_VIEWS, VIEWS

REPORT = REPO_ROOT / "outputs" / "validation_report.json"


def check_data(task: Task, output: dict, views_with_rows: dict[str, bool]) -> dict:
    declared = {view for label in task.required_data for view in REQUIRED_DATA_VIEWS[label]}
    queries = " ".join(q for q in output["evidence"].get("queries", []) if not q.startswith("SPARQL:"))
    used = {name for name in re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", queries, flags=re.IGNORECASE) if name in VIEWS}
    return {
        "empty_or_missing_views": sorted(v for v in declared if not views_with_rows[v]),
        "undeclared_views": sorted(used - declared),
    }


def compare_with_draft(task: Task, output: dict) -> dict:
    draft, result, kind = task.draft, output["result"], task.answer_type
    if kind == "numeric" and draft.get("truth_value") is not None:
        tolerance = task.evaluation.tolerance or 0.0
        gold, expected = float(result["answer"]), float(draft["truth_value"])
        close = abs(gold - expected) <= tolerance * max(abs(expected), 1e-12) if tolerance else gold == expected
        return {"draft": expected, "gold": gold, "matches": close}
    if kind == "set" and draft.get("truth_set_size") is not None:
        members = result["answer"]
        comparison = {"draft_size": int(draft["truth_set_size"]), "gold_size": len(members),
                      "matches": len(members) == int(draft["truth_set_size"])}
        if draft.get("truth_members_sha256"):
            # The spreadsheet doesn't say how it encoded the member list, so try the likely encodings.
            encodings = {"newline_joined": "\n".join(members), "comma_joined": ",".join(members),
                         "json_list": json.dumps(members)}
            prefix = draft["truth_members_sha256"]
            comparison["members_sha256_matches"] = [
                name for name, text in encodings.items() if hashlib.sha256(text.encode()).hexdigest().startswith(prefix)
            ] or False
        return comparison
    if kind == "scalar_text" and draft.get("truth_answer") is not None:
        return {"draft": draft["truth_answer"], "gold": result["answer"], "matches": result["answer"] == draft["truth_answer"]}
    if kind == "refusal" and draft.get("refusal_predicate"):
        return {"draft": draft["refusal_predicate"], "gold": result["predicate"],
                "matches": result["predicate"] == draft["refusal_predicate"]}
    return {"matches": None, "note": "no comparable draft answer"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_ids", nargs="*", help="default: every task with a module")
    args = parser.parse_args()

    all_tasks = load_tasks()
    implemented = registry.task_files()
    tasks = [t for t in all_tasks if t.task_id in implemented and (not args.task_ids or t.task_id in args.task_ids)]
    # Given task IDs, check only those tasks' modules; otherwise check the whole manifest.
    registry_problems = registry.check(tasks, None if args.task_ids else all_tasks)
    for problem in registry_problems:
        print(f"registry: {problem}")

    views_with_rows = {}
    if tasks:
        context = GoldContext()
        views_with_rows = {view: context.view_has_rows(view) for view in VIEWS}
    entries, failed = [], bool(registry_problems)
    for task in tasks:
        entry = {"task_id": task.task_id, "status": task.status}
        try:
            first = run_task(task, GoldContext())
            second = run_task(task, GoldContext())
        except Exception as err:  # a task that can't run is a finding, not a crash
            entry.update(runs=False, error=f"{type(err).__name__}: {err}")
            failed = True
        else:
            entry.update(
                runs=True,
                well_formed=True,
                deterministic=output_hash(first) == output_hash(second),
                output_sha256=output_hash(first),
                data=check_data(task, first, views_with_rows),
                draft_comparison=compare_with_draft(task, first),
            )
            failed = failed or not entry["deterministic"] or bool(entry["data"]["empty_or_missing_views"])
        entries.append(entry)
        print(json.dumps(entry))

    if args.task_ids and REPORT.exists():
        # A partial run updates its tasks' entries and keeps the rest of the last report.
        previous = json.loads(REPORT.read_text())
        kept = {e["task_id"]: e for e in previous.get("tasks", [])}
        kept.update({e["task_id"]: e for e in entries})
        entries = [kept[k] for k in sorted(kept)]
        registry_problems = previous.get("registry_problems", []) if not registry_problems else registry_problems
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"registry_problems": registry_problems, "tasks": entries}, indent=2) + "\n")
    print(f"wrote {REPORT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
