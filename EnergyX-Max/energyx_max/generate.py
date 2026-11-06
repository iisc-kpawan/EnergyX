"""Stage 1, Generator Agent: turn each seed into up to two question templates.

    uv run python -m energyx_max.generate [--limit N]

Writes work/templates_generated.jsonl, one line per template, and can be resumed: seeds already
processed are skipped. The second template of each seed targets the level its category has fewest
templates of so far, so the templates spread over the five levels.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from .common import CATEGORIES, LEVELS, MODELS, WORK, append_jsonl, chat, parse_json, read_jsonl
from .prompts import GENERATOR_SYSTEM, generator_more_user, generator_user
from . import slots as slot_space

OUT = WORK / "templates_generated.jsonl"
DONE = WORK / "generator_done.jsonl"
ANSWER_TYPES = {"numeric", "set", "scalar_text", "record"}
OUTPUT_FORMAT = {"numeric": "Number", "set": "List", "scalar_text": "Label", "record": "Table"}
WEATHER_WORDS = ("temperature", "weather", "dew", "wind", "humid", "precipitation", "pressure", "cloud", "climate")
METADATA_WORDS = ("floor area", "eui", "intensity", "per square", "year built", "built", "floors", "type", "site",
                  "city", "country", "located")
ELECTRICITY_WORDS = ("consumption", "electricity", "kwh", "load", "demand", "reading", "forecast", "anomal",
                     "cluster", "profile", "peak", "missing", "gap", "coverage", "valid", "outlier", "usage", "meter")


def required_data(template: str, slots: list[str]) -> list[str]:
    """The data a template needs, from its wording and slots."""
    text = template.lower()
    needs = []
    if any(s in slots for s in ("site_id", "city", "country", "building_type", "building_type_2")) \
            or any(w in text for w in METADATA_WORDS):
        needs.append("Building Metadata")
    if any(w in text for w in ELECTRICITY_WORDS):
        needs.append("Electricity Consumption Data")
    if "weather_variable" in slots or any(w in text for w in WEATHER_WORDS):
        needs.append("Weather Data")
    return needs or ["Building Metadata"]


def clean(raw: dict, seed: dict) -> dict | None:
    """A template with consistent slots, category and level, or None."""
    template = str(raw.get("template", "")).strip()
    used = re.findall(r"\{(\w+)\}", template)
    if not template or not used or not slot_space.well_formed(template):
        return None
    try:
        category = int(raw.get("category") or seed.get("category") or 0)
    except (TypeError, ValueError):
        return None
    level = str(raw.get("level", "")).upper()
    if category not in CATEGORIES or level not in LEVELS:
        return None
    answer_type = raw.get("answer_type") if raw.get("answer_type") in ANSWER_TYPES else "record"
    return {"template": template, "slots": sorted(set(used), key=used.index), "category": category, "level": level,
            "answer_type": answer_type, "expected_output_format": OUTPUT_FORMAT[answer_type],
            "required_data": required_data(template, used), "sub_task": raw.get("sub_task"),
            "user_role": raw.get("user_role") or seed.get("user_role"), "assumptions": raw.get("assumptions")}


def emit(seed: dict, reply: dict, prefix: str, coverage: Counter) -> int:
    kept = 0
    for n, raw in enumerate(reply.get("templates") or []):
        if not isinstance(raw, dict):
            continue
        template = clean(raw, seed)
        if template is None:
            continue
        template.update({"template_id": f"{seed['seed_id']}-{prefix}{n + 1}", "seed_id": seed["seed_id"],
                         "seed_source": seed["source"], "seed_question": seed["question"],
                         "knowledge_source": seed.get("knowledge_source"),
                         "source_reference": seed.get("source_reference"), "generator": MODELS["generator"]})
        append_jsonl(OUT, template)
        coverage[(template["category"], template["level"])] += 1
        kept += 1
    return kept


def working_coverage() -> Counter:
    """Templates per cell that have a working program, or else that the Judge accepted, or else generated."""
    judged = {t["template_id"]: t for t in read_jsonl(WORK / "templates_judged.jsonl")}
    programs = read_jsonl(WORK / "programs.jsonl")
    if programs:
        rows = [judged[p["template_id"]] for p in programs if p["status"] == "ok" and p["template_id"] in judged]
    elif judged:
        rows = [t for t in judged.values() if t["judge_verdict"] == "accept"]
    else:
        rows = read_jsonl(OUT)
    return Counter((t["category"], t["level"]) for t in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--round", type=int, default=1,
                        help="1 writes two templates per seed; a later round adds two more per seed for the "
                             "categories and levels with the fewest working templates")
    args = parser.parse_args()

    seeds = read_jsonl(WORK / "seeds.jsonl")
    done = {row["seed_id"] for row in read_jsonl(DONE)}
    if args.round == 1:
        coverage = Counter((t["category"], t["level"]) for t in read_jsonl(OUT))
        todo = [s for s in seeds if s["seed_id"] not in done][: args.limit]
    else:
        coverage = working_coverage()
        weakness = {c: sum(coverage[(c, level)] for level in LEVELS) for c in CATEGORIES}
        key = lambda s: f"{s['seed_id']}#R{args.round}"  # noqa: E731
        todo = sorted((s for s in seeds if s.get("category") and key(s) not in done),
                      key=lambda s: (weakness[s["category"]], s["seed_id"]))[: args.limit]
        existing: dict = {}
        for template in read_jsonl(OUT):
            existing.setdefault(template["seed_id"], []).append(template["template"])
    print(f"{len(todo)} seeds to process with {MODELS['generator']} (round {args.round})")
    for i, seed in enumerate(todo, 1):
        category = seed.get("category") or 1
        level = min(LEVELS, key=lambda lv: (coverage[(category, lv)], lv))
        if args.round == 1:
            reply = parse_json(chat("generator", GENERATOR_SYSTEM, generator_user(seed, level), as_json=True)) or {}
            kept = emit(seed, reply, "T", coverage)
            append_jsonl(DONE, {"seed_id": seed["seed_id"], "templates": kept})
        else:
            prompt = generator_more_user(seed, level, existing.get(seed["seed_id"], []))
            reply = parse_json(chat("generator", GENERATOR_SYSTEM, prompt, as_json=True)) or {}
            kept = emit(seed, reply, f"R{args.round}T", coverage)
            append_jsonl(DONE, {"seed_id": key(seed), "templates": kept})
        if i % 10 == 0:
            print(f"  {i}/{len(todo)} seeds, {sum(coverage.values())} templates", flush=True)


if __name__ == "__main__":
    main()
