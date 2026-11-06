"""Stage 5: write the EnergyX-Max release files into outputs/.

    uv run python -m energyx_max.build

Writes, in outputs/:
- EnergyX-Max.jsonl and EnergyX-Max.csv: one row per question, with the seed-question attributes,
  the template, the slot values, the answer computed by the template's program, and provenance;
- templates.jsonl: every generated template with the Judge Agent's verdict and the program status;
- summary.md: counts by category and level, and by pipeline stage.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from .common import CATEGORIES, LEVELS, MODELS, ROOT, WORK, read_jsonl, write_jsonl


def main() -> None:
    seeds = {s["seed_id"]: s for s in read_jsonl(WORK / "seeds.jsonl")}
    judged = {t["template_id"]: t for t in read_jsonl(WORK / "templates_judged.jsonl")}
    programs = {p["template_id"]: p for p in read_jsonl(WORK / "programs.jsonl")}
    instances = read_jsonl(WORK / "instances.jsonl")
    instances.sort(key=lambda r: (judged[r["template_id"]]["category"], judged[r["template_id"]]["level"],
                                  r["template_id"], r["question"]))

    rows = []
    for n, instance in enumerate(instances, 1):
        template = judged[instance["template_id"]]
        seed = seeds.get(template["seed_id"], {})
        rows.append({
            "question_id": f"MAX-{n:05d}",
            "question": instance["question"],
            "category": CATEGORIES[template["category"]],
            "difficulty_level": template["level"],
            "sub_task": template.get("sub_task"),
            "intended_user_role": template.get("user_role"),
            "expected_output_format": template.get("expected_output_format"),
            "required_data": template.get("required_data"),
            "knowledge_source": seed.get("knowledge_source") or "EnergyX-Seed question",
            "source_reference": seed.get("source_reference"),
            "assumptions": template.get("assumptions"),
            "answer_type": template["answer_type"],
            "answer": instance["answer"],
            "unit": instance.get("unit"),
            "template_id": template["template_id"],
            "template": template["template"],
            "slots": instance["slots"],
            "seed_id": template["seed_id"],
            "seed_source": template.get("seed_source"),
            "program": f"programs/{template['template_id']}.py",
            "generator_model": template.get("generator"),
            "judge_model": template.get("judge"),
            "coder_model": programs[template["template_id"]].get("coder"),
            "validation": "automatic checks only; not manually validated",
        })
    write_jsonl(ROOT / "EnergyX-Max.jsonl", rows)
    pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                  for r in rows]).to_csv(ROOT / "EnergyX-Max.csv", index=False)

    templates = []
    used = Counter(r["template_id"] for r in rows)
    for template_id, template in judged.items():
        program = programs.get(template_id, {})
        templates.append({**template, "program_status": program.get("status"),
                          "program_repaired": program.get("repaired"), "questions": used.get(template_id, 0)})
    write_jsonl(ROOT / "templates.jsonl", templates)

    grid = pd.crosstab(pd.Series([r["category"] for r in rows], name="category"),
                       pd.Series([r["difficulty_level"] for r in rows], name="level"))
    grid = grid.reindex(index=list(CATEGORIES.values()), columns=list(LEVELS), fill_value=0)
    grid["total"] = grid.sum(axis=1)
    generated = read_jsonl(WORK / "templates_generated.jsonl")
    accepted = sum(t["judge_verdict"] == "accept" for t in judged.values())
    working = sum(p["status"] == "ok" for p in programs.values())
    lines = [
        "# EnergyX-Max summary", "",
        f"- Seed questions: {len(seeds)}",
        f"- Templates written by the Generator Agent ({MODELS['generator']}): {len(generated)}",
        f"- Templates accepted by the Judge Agent ({MODELS['judge']}): {accepted}",
        f"- Templates with a working program from the Coding Agent ({MODELS['coder']}): {working}",
        f"- Templates used: {len(used)}",
        f"- Questions: {len(rows)}", "",
        "## Questions by category and level", "",
        "| Category | " + " | ".join(grid.columns) + " |",
        "|---|" + "---|" * len(grid.columns),
    ]
    lines += [f"| {name} | " + " | ".join(str(v) for v in row) + " |" for name, row in grid.iterrows()]
    lines += ["| Total | " + " | ".join(str(v) for v in grid.sum()) + " |"]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} questions from {len(used)} templates")


if __name__ == "__main__":
    main()
