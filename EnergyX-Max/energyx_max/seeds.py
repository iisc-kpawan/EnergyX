"""Stage 0: build work/seeds.jsonl, the seed questions Max's templates are generated from.

    uv run python -m energyx_max.seeds

Sources: the EnergyX-Seed survey sheets (the team sheet of 122 questions and the advanced-category
sheet of 24) and the example questions extracted from the design conversation that are answerable
on BDG2 as written. Seeds are mapped to the 10 benchmark categories; near-duplicates are dropped.
"""

from __future__ import annotations

import csv
import difflib

import pandas as pd

from . import config
from .common import CATEGORIES, WORK, write_jsonl

# The seed sheets used 12 categories. Comparative & Peer Benchmarking (7) folds into Building
# Performance & Normalization; Advanced Inference & Decision Support (12) has no fixed home and the
# Generator Agent assigns one.
OLD_TO_NEW = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 5, 8: 7, 9: 8, 10: 9, 11: 10, 12: None}
ADVANCED = {"Forecasting & Predictive Analytics": 8, "Anomaly & Change Detection": 9,
            "Clustering, Similarity & Representation Analytics": 10, "Advanced Inference & Decision Support": None}
DUPLICATE = config.setting("pipeline", "seed_duplicate_similarity")


def _level(value) -> str | None:
    text = str(value).strip().upper()
    return text if text in {"L1", "L2", "L3", "L4", "L5"} else None


def load() -> list[dict]:
    seeds = []
    team = pd.read_excel(config.path("seeds", "team"), sheet_name="gold")
    for row in team.itertuples():
        seeds.append({"seed_id": row.question_id, "source": "EnergyX-Seed survey (team sheet)",
                      "question": str(row.question).strip(), "category": OLD_TO_NEW.get(int(row.category)),
                      "level": _level(row.difficulty), "expected_output_format": row.expected_output_format,
                      "required_data": row.required_data, "user_role": None, "knowledge_source": None,
                      "source_reference": None, "notes": None if pd.isna(row.caveats) else row.caveats})
    advanced = pd.read_excel(config.path("seeds", "advanced"))
    for i, row in enumerate(advanced.to_dict("records")):
        seeds.append({"seed_id": f"ADV-{i + 1:02d}", "source": "EnergyX-Seed survey (advanced sheet)",
                      "question": str(row["Question"]).strip(), "category": ADVANCED.get(row["Primary Task Category"]),
                      "level": _level(row["Difficulty Level"]), "expected_output_format": row["Expected Output Format"],
                      "required_data": row["Required Data"], "user_role": row["Intended User Role(s)"],
                      "knowledge_source": row["Knowledge Source"], "source_reference": row["Source Reference"],
                      "notes": row["Notes / Assumptions"]})
    with config.path("seeds", "design_conversation").open() as handle:
        for row in csv.DictReader(handle):
            if row["bdg2_answerable_as_written"] != "True" or row["near_duplicate_of"]:
                continue
            category = OLD_TO_NEW.get(int(row["category"])) if row["category"] else None
            seeds.append({"seed_id": row["seed_id"], "source": "design conversation examples",
                          "question": row["question"].strip(), "category": category, "level": None,
                          "expected_output_format": None, "required_data": None, "user_role": None,
                          "knowledge_source": None, "source_reference": None, "notes": None})
    return seeds


def deduplicate(seeds: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for seed in seeds:
        text = seed["question"].lower()
        if any(difflib.SequenceMatcher(None, text, other["question"].lower()).ratio() >= DUPLICATE for other in kept):
            continue
        kept.append(seed)
    return kept


def main() -> None:
    seeds = load()
    kept = deduplicate(seeds)
    for seed in kept:
        seed["category_name"] = CATEGORIES.get(seed["category"])
    write_jsonl(WORK / "seeds.jsonl", kept)
    by_source = pd.Series([s["source"] for s in kept]).value_counts().to_dict()
    print(f"{len(seeds)} seeds read, {len(kept)} kept after removing near-duplicates: {by_source}")


if __name__ == "__main__":
    main()
