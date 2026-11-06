"""Check the EnergyX-Max release files and write outputs/validation_report.json.

    uv run python -m energyx_max.validate [--sample N]

Checks:
- structure: unique question IDs and texts, valid categories and levels, every program present;
- exclusion: no question copies or nearly copies an EnergyX-Seed or EnergyX-Gold question;
- reproducibility: re-running each sampled question's program on its slot values gives the stored
  answer (all questions when --sample is 0).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from . import config, execute
from .common import CATEGORIES, LEVELS, ROOT, read_jsonl
from .instantiate import Excluder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300, help="questions to re-run; 0 means all")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "EnergyX-Max.jsonl")
    repo = config.REPO_ROOT

    ids = Counter(r["question_id"] for r in rows)
    texts = Counter(r["question"] for r in rows)
    structure = {
        "questions": len(rows),
        "duplicate_ids": [k for k, v in ids.items() if v > 1],
        "duplicate_texts": [k for k, v in texts.items() if v > 1],
        "bad_category": [r["question_id"] for r in rows if r["category"] not in CATEGORIES.values()],
        "bad_level": [r["question_id"] for r in rows if r["difficulty_level"] not in LEVELS],
        "missing_program": sorted({r["program"] for r in rows if not (repo / r["program"]).exists()}),
    }
    excluder = Excluder()
    copies = [r["question_id"] for r in rows if excluder.excluded(r["question"])]

    sample = rows if args.sample == 0 else random.Random(0).sample(rows, min(args.sample, len(rows)))
    mismatches = []
    with execute.Runner() as runner:
        for row in sample:
            result = runner.run_many((repo / row["program"]).read_text(), [row["slots"]])[0]
            if not result["ok"] or result["answer"] != row["answer"]:
                mismatches.append({"question_id": row["question_id"],
                                   "error": result.get("error", "answer differs")[-200:]})

    report = {"structure": structure, "copies_of_seed_or_gold": copies,
              "reproducibility": {"rerun": len(sample), "mismatches": mismatches}}
    ok = not any(v for k, v in structure.items() if k != "questions") and not copies and not mismatches
    report["ok"] = ok
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} questions; {len(copies)} copies of Seed or Gold questions; "
          f"{len(mismatches)} of {len(sample)} re-runs differ; {'OK' if ok else 'PROBLEMS'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
