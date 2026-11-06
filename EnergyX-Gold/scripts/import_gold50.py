"""Make data/task_manifest.jsonl the 50 EnergyX-Gold questions (Q01 to Q50).

    uv run python scripts/import_gold50.py

Reads the question list (config key questions_xlsx: S.No, Question ID, Level, Category, Question, with
three title rows above the header) and data/gold50_questions.yaml, which pins each question's
definition and metadata. For each question it also records the closest seed question, from the
EnergyX-Seed survey sheets and the earlier 60-question draft in data/seeds/.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energyx_gold.core.config import REPO_ROOT, load_config  # noqa: E402
from energyx_gold.core.loader import write_tasks  # noqa: E402
from energyx_gold.core.registry import module_name  # noqa: E402
from energyx_gold.core.schemas import NOT_ANSWERABLE, Task  # noqa: E402

SETTINGS = REPO_ROOT / "data" / "gold50_questions.yaml"
SEEDS = REPO_ROOT / "data" / "seeds"
REFUSAL_SENTENCE = f"If the data can't answer the question, reply \"{NOT_ANSWERABLE}\" and name the missing data."
EVALUATION = {"numeric": "numeric", "set": "set_match", "scalar_text": "scalar_text", "record": "record",
              "refusal": "refusal", "method": "rubric"}
# At or above this similarity a Gold question counts as derived from the seed; below, as newly written.
DERIVED = 0.6


def seed_pool() -> list[tuple[str, str, str]]:
    """(source, seed_id, question) for every seed question on disk."""
    pool = []
    team = pd.read_excel(SEEDS / "team_seed_questions.xlsx", sheet_name="gold")
    pool += [("EnergyX-Seed (team sheet)", r.question_id, str(r.question)) for r in team.itertuples()]
    advanced = pd.read_excel(SEEDS / "advanced_seed_questions.xlsx")
    pool += [("EnergyX-Seed (advanced sheet)", f"ADV-{i + 1:02d}", str(q)) for i, q in enumerate(advanced["Question"])]
    draft = pd.read_excel(SEEDS / "gold60_draft_questions.xlsx")
    pool += [("EnergyX-Gold draft of 60", r[2], str(r[3])) for r in draft.itertuples()]
    return pool


def closest_seed(question: str, pool: list[tuple[str, str, str]]) -> dict:
    def score(entry):
        return difflib.SequenceMatcher(None, question.lower(), entry[2].lower()).ratio()
    source, seed_id, text = max(pool, key=score)
    similarity = round(score((source, seed_id, text)), 3)
    return {"origin": "derived from seed" if similarity >= DERIVED else "newly written",
            "closest_seed_source": source, "closest_seed_id": seed_id,
            "closest_seed_question": text, "similarity": similarity}


def main() -> None:
    config = load_config()
    listing = pd.read_excel(config.questions_xlsx, sheet_name=0, header=3).dropna(subset=["Question ID"])
    settings = yaml.safe_load(SETTINGS.read_text())
    pool = seed_pool()

    tasks = []
    for row in listing.itertuples(index=False):
        serial, task_id, level, category_name, question = (int(row[0]), str(row[1]).strip(), str(row[2]).strip(),
                                                           str(row[3]).strip(), str(row[4]).strip())
        spec = settings[task_id]
        tasks.append(Task.model_validate({
            "task_id": task_id, "serial": serial, "question": question,
            "category": spec["category"], "category_name": category_name, "difficulty": level,
            "secondary_categories": spec["secondary_categories"], "sub_task": spec["sub_task"],
            "user_roles": spec["user_roles"], "knowledge_source": spec["knowledge_source"],
            "source_reference": spec["source_reference"], "seed_origin": closest_seed(question, pool),
            "expected_output_format": spec["expected_output_format"],
            "required_data": spec["required_data"], "temporal_scope": str(spec["temporal_scope"]),
            "answer_type": spec["answer_type"], "output_type": spec["output_type"],
            "execution_stack": spec["execution_stack"],
            "answer_format": {
                "instructions": f"{spec['instructions'].strip()} {REFUSAL_SENTENCE}",
                "unit": spec.get("unit"), "decimals": spec.get("decimals"),
                "order": "any" if spec["answer_type"] == "set" else None, "key": spec.get("key"),
            },
            "evaluation": {"method": EVALUATION[spec["answer_type"]], "field": spec.get("key"),
                           "tolerance": spec["tolerance"]},
            "definition": spec["definition"].strip(), "data_version": config.data_version,
            "reference_module": module_name(task_id, spec["category"]),
            "status": "gold_candidate", "status_note": "Reference module not yet verified.", "draft": {},
        }))

    write_tasks(tasks)
    print(f"{len(tasks)} tasks written")


if __name__ == "__main__":
    main()
