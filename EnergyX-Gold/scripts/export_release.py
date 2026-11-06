"""Write the EnergyX-Gold release files into outputs/.

    uv run python scripts/export_release.py

Reads data/task_manifest.jsonl and outputs/gold_results.jsonl and writes, in question order:
- outputs/EnergyX-Gold.jsonl: one object per question with its metadata, definition, answer format,
  grading rule, reference answer and the path of its reference implementation;
- outputs/EnergyX-Gold.csv and outputs/EnergyX-Gold.xlsx: the same as a flat table, the reference answer
  as JSON text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energyx_gold.core.config import REPO_ROOT  # noqa: E402
from energyx_gold.core.loader import load_tasks  # noqa: E402
from energyx_gold.core.runner import RESULTS  # noqa: E402

OUT = REPO_ROOT / "outputs"
EXCEL_CELL = 32000  # Excel's cell limit is 32,767 characters
GRADING = {
    "numeric": "numeric: within the relative tolerance (0 means exact)",
    "set": "set: the same members; compared on the key named in answer_format",
    "scalar_text": "label: exact match",
    "record": "record: each field compared; numbers within the relative tolerance, labels exactly",
}


def rows() -> list[dict]:
    results = {json.loads(line)["task_id"]: json.loads(line) for line in RESULTS.read_text().splitlines() if line}
    out = []
    for task in sorted(load_tasks(), key=lambda t: t.serial or 0):
        result = results[task.task_id]
        module = task.reference_module.replace(".", "/") + ".py"
        out.append({
            "question_id": task.task_id,
            "question": task.question,
            "category": task.category_name,
            "secondary_categories": task.secondary_categories,
            "sub_task": task.sub_task,
            "difficulty_level": task.difficulty,
            "intended_user_roles": task.user_roles,
            "expected_output_format": task.expected_output_format,
            "required_data": task.required_data,
            "temporal_scope": task.temporal_scope,
            "knowledge_source": task.knowledge_source,
            "source_reference": task.source_reference,
            "seed_origin": task.seed_origin.get("origin"),
            "closest_seed_id": task.seed_origin.get("closest_seed_id"),
            "closest_seed_question": task.seed_origin.get("closest_seed_question"),
            "definition": task.definition,
            "answer_instructions": task.answer_format.instructions,
            "answer_type": task.answer_type,
            "output_type": task.output_type,
            "unit": task.answer_format.unit,
            "grading_rule": GRADING[task.answer_type],
            "relative_tolerance": task.evaluation.tolerance,
            "reference_answer": result["result"],
            "reference_evidence": result["evidence"],
            "reference_implementation": module,
            "independent_check": f"tests/test_{task.task_id.lower()}.py",
            "execution_stack": task.execution_stack,
            "data_version": task.data_version,
        })
    return out


def main() -> None:
    data = rows()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EnergyX-Gold.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
    flat = pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                          for k, v in row.items() if k != "reference_evidence"} for row in data])
    flat.to_csv(OUT / "EnergyX-Gold.csv", index=False)
    sheet = flat.copy()
    too_long = sheet["reference_answer"].str.len() > EXCEL_CELL
    sheet.loc[too_long, "reference_answer"] = [
        json.dumps({"count": row["reference_answer"].get("count"), "members": "see EnergyX-Gold.jsonl"})
        for row, long in zip(data, too_long) if long]
    with pd.ExcelWriter(OUT / "EnergyX-Gold.xlsx") as writer:
        sheet.to_excel(writer, sheet_name="EnergyX-Gold", index=False)
    print(f"{len(data)} questions written to {OUT}")


if __name__ == "__main__":
    main()
