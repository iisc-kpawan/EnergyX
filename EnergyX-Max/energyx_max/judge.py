"""Stage 2, Judge Agent: accept, fix or reject each generated template.

    uv run python -m energyx_max.judge [--limit N]

Reads work/templates_generated.jsonl and writes work/templates_judged.jsonl (every verdict). Resumable.
A fixed template is re-checked here for slots, category and level before it counts as accepted.
"""

from __future__ import annotations

import argparse
import random
import re

from .common import CATEGORIES, LEVELS, MODELS, WORK, append_jsonl, chat, parse_json, read_jsonl
from .prompts import JUDGE_SYSTEM, judge_user
from . import slots as slot_space
from .slots import VOCABULARY

IN = WORK / "templates_generated.jsonl"
OUT = WORK / "templates_judged.jsonl"


def example_question(template: dict) -> str:
    if not slot_space.well_formed(template["template"]):
        return "(no valid slot values exist for this template)"
    rng = random.Random(template["template_id"])
    for _ in range(20):
        fill = slot_space.sample(template["slots"], rng)
        if fill:
            return slot_space.render(template["template"], fill)
    return "(no valid slot values exist for this template)"


def verdict_for(template: dict) -> dict:
    example = example_question(template)
    reply = parse_json(chat("judge", JUDGE_SYSTEM, judge_user(template, example), as_json=True, max_tokens=800)) or {}
    verdict = str(reply.get("verdict", "")).lower()
    fixed = str(reply.get("template") or template["template"]).strip()
    used = re.findall(r"\{(\w+)\}", fixed)
    try:
        category = int(reply.get("category") or template["category"])
    except (TypeError, ValueError):
        category = template["category"]
    level = str(reply.get("level") or template["level"]).upper()
    problems = []
    if not used or not slot_space.well_formed(fixed):
        problems.append("slots missing or malformed after the judge's edit")
    if example.startswith("(no valid"):
        problems.append("no valid slot values")
    if any(slot not in VOCABULARY for slot in used):
        problems.append("unknown slot after the judge's edit")
    if category not in CATEGORIES or level not in LEVELS:
        problems.append("category or level out of range")
    accepted = verdict == "accept" and not problems
    return {**template, "template_generated": template["template"],
            "template": fixed if accepted else template["template"],
            "slots": sorted(set(used), key=used.index) if accepted else template["slots"],
            "category": category if accepted else template["category"],
            "level": level if accepted else template["level"],
            "judge_verdict": "accept" if accepted else "reject",
            "judge_reason": "; ".join(filter(None, [str(reply.get("reason") or ""), *problems])),
            "judge_edited": accepted and fixed != template["template"], "judge": MODELS["judge"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    done = {row["template_id"] for row in read_jsonl(OUT)}
    todo = [t for t in read_jsonl(IN) if t["template_id"] not in done][: args.limit]
    print(f"{len(todo)} templates to judge with {MODELS['judge']}")
    accepted = 0
    for i, template in enumerate(todo, 1):
        row = verdict_for(template)
        accepted += row["judge_verdict"] == "accept"
        append_jsonl(OUT, row)
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} judged, {accepted} accepted", flush=True)
    print(f"done: {accepted} of {len(todo)} accepted")


if __name__ == "__main__":
    main()
