"""Write the gold answers as tables for checking agent responses against them.

    uv run python scripts/export_results.py

Reads data/task_manifest.jsonl and outputs/gold_results.jsonl and writes, in the final list's order
(S.No):
- outputs/final60_results.csv: one row per question, with the full answer as JSON;
- outputs/final60_results.md: the same as a readable table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energyx_gold.core.config import REPO_ROOT  # noqa: E402
from energyx_gold.core.loader import load_tasks  # noqa: E402
from energyx_gold.core.runner import RESULTS  # noqa: E402

CSV_PATH = REPO_ROOT / "outputs" / "gold50_results.csv"
MD_PATH = REPO_ROOT / "outputs" / "gold50_results.md"
SET_PREVIEW = 10


def _number(value) -> str:
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def readable(task, output: dict | None) -> str:
    """A short form of the answer for the Markdown table."""
    if output is None:
        return "(not computed)"
    result = output["result"]
    kind = task.answer_type
    if kind == "numeric":
        unit = result.get("unit") or ""
        return f"{_number(result['answer'])} {unit}".strip()
    if kind == "set":
        members = result["answer"]
        shown = ", ".join(members[:SET_PREVIEW])
        more = f" … and {len(members) - SET_PREVIEW} more" if len(members) > SET_PREVIEW else ""
        return f"{result['count']}: {shown}{more}" if members else "0 (none)"
    if kind == "scalar_text":
        return str(result["answer"])
    if kind == "record":
        units = result.get("units", {})
        parts = []
        for key, value in result["answer"].items():
            unit = units.get(key, "")
            parts.append(f"{key}: {_number(value)}{' ' + unit if unit else ''}")
        return "; ".join(parts)
    if kind == "refusal":
        return f"Not answerable ({result['predicate']})"
    return "Method (no single value); see the method text"


def main() -> None:
    tasks = sorted(load_tasks(), key=lambda t: (t.serial or 10**6, t.task_id))
    outputs = {}
    if RESULTS.exists():
        outputs = {json.loads(l)["task_id"]: json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()}

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s_no", "question_id", "level", "question", "answer_type", "answer", "answer_json",
                         "answer_instructions", "status", "status_note"])
        for t in tasks:
            output = outputs.get(t.task_id)
            payload = output["result"] if output else None
            writer.writerow([t.serial, t.task_id, t.difficulty, t.question, t.answer_type, readable(t, output),
                             json.dumps(payload, ensure_ascii=False) if payload else "", t.answer_format.instructions,
                             t.status, t.status_note or ""])

    lines = [
        "# EnergyX-GOLD: gold answers for the 60 questions",
        "",
        "One row per question, in the question list's order. `outputs/gold50_results.csv` has the same rows with the "
        "full answer as JSON, and `outputs/gold_results.jsonl` has the evidence. Questions marked "
        "`gold_candidate` rest on a choice listed in `reviews/final60-review.md`.",
        "",
        "| S.No | ID | Level | Question | Gold answer | Status |",
        "|---:|---|---|---|---|---|",
    ]
    for t in tasks:
        question = t.question.replace("|", "\\|")
        answer = readable(t, outputs.get(t.task_id)).replace("|", "\\|")
        lines.append(f"| {t.serial} | {t.task_id} | {t.difficulty} | {question} | {answer} | {t.status} |")
    MD_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {CSV_PATH}\nwrote {MD_PATH}")


if __name__ == "__main__":
    main()
