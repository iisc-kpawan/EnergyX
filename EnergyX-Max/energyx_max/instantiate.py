"""Stage 4: fill each working template's slots with valid values and compute the answers.

    uv run python -m energyx_max.instantiate

The target is 120 questions per category and level, 6,000 in all. Filling runs in three passes, each
drawing more questions from templates that still have distinct slot values left:
1. per cell (category and level), sharing the cell's target among its templates;
2. per category, topping up a category that fell short from its other templates;
3. over the whole set, topping up until the total reaches 6,000.
A question is kept only when its program returns a usable answer, its text is new, and it is not a
copy or near-copy (similarity 0.9 or more) of any EnergyX-Seed or EnergyX-Gold question.
Writes outputs/stages/instances.jsonl.
"""

from __future__ import annotations

import difflib
import math
import random
from collections import Counter, defaultdict

from . import config, execute, slots as slot_space
from .common import CATEGORIES, LEVELS, PROGRAMS, TARGET_PER_CELL, WORK, read_jsonl, write_jsonl

GOLD_QUESTIONS = config.path("gold_questions")
NEAR_COPY = config.setting("pipeline", "near_copy_similarity")
MAX_PER_TEMPLATE = config.setting("pipeline", "max_questions_per_template")
TARGET_TOTAL = TARGET_PER_CELL * len(CATEGORIES) * len(LEVELS)
DRAWS = 40  # slot combinations drawn per question wanted before a template counts as used up


class Excluder:
    """Rejects a question that copies or nearly copies a Seed or Gold question."""

    def __init__(self):
        texts = [row["question"] for row in read_jsonl(GOLD_QUESTIONS)]
        texts += [row["question"] for row in read_jsonl(WORK / "seeds.jsonl")]
        self.texts = [t.lower().strip() for t in texts]
        self.exact = set(self.texts)
        self.rejected = 0

    def excluded(self, question: str) -> bool:
        text = question.lower().strip()
        if text in self.exact:
            self.rejected += 1
            return True
        matcher = difflib.SequenceMatcher(None, text)
        for other in self.texts:
            matcher.set_seq1(other)
            if matcher.real_quick_ratio() >= NEAR_COPY and matcher.quick_ratio() >= NEAR_COPY \
                    and matcher.ratio() >= NEAR_COPY:
                self.rejected += 1
                return True
        return False


class Filler:
    """Draws new questions from templates, keeping track of which ones have run out."""

    def __init__(self, workers, templates: list[dict]):
        self.workers = workers
        self.templates = {t["template_id"]: t for t in templates}
        self.code = {tid: (PROGRAMS / f"{tid}.py").read_text() for tid in self.templates}
        self.rng = {tid: random.Random(f"{config.setting('pipeline', 'random_seed')}-{tid}") for tid in self.templates}
        self.excluder = Excluder()
        self.seen: set[str] = set()
        self.used: Counter = Counter()
        self.exhausted: set[str] = set()
        self.rows: list[dict] = []

    def open(self, template_id: str) -> bool:
        return template_id not in self.exhausted and self.used[template_id] < MAX_PER_TEMPLATE

    def draw(self, template_id: str, wanted: int) -> int:
        """Try to add `wanted` questions from one template. Returns how many were added."""
        wanted = min(wanted, MAX_PER_TEMPLATE - self.used[template_id])
        if wanted <= 0:
            return 0
        template, rng = self.templates[template_id], self.rng[template_id]
        candidates = []
        for _ in range(wanted * DRAWS):
            if len(candidates) >= math.ceil(wanted * 1.5):
                break
            fill = slot_space.sample(template["slots"], rng)
            if not fill:
                continue
            try:
                question = slot_space.render(template["template"], fill)
            except (KeyError, IndexError, ValueError):
                continue
            if question in self.seen or self.excluder.excluded(question):
                continue
            self.seen.add(question)
            candidates.append((question, fill))
        results = self.workers.run_many(self.code[template_id],
                                        [{k: v["value"] for k, v in f.items()} for _, f in candidates])
        added = 0
        for (question, fill), result in zip(candidates, results):
            if result["ok"] and added < wanted:
                self.rows.append({"question": question, "slots": {k: v["value"] for k, v in fill.items()},
                                  "answer": result["answer"], "unit": result["unit"], "template_id": template_id,
                                  "category": template["category"], "level": template["level"]})
                added += 1
        self.used[template_id] += added
        if added < wanted:
            self.exhausted.add(template_id)
        return added

    def fill_to(self, template_ids: list[str], have: int, target: int) -> int:
        """Share `target - have` among the open templates, round after round, until reached or all used up."""
        while have < target:
            active = [t for t in template_ids if self.open(t)]
            if not active:
                break
            share = max(1, math.ceil((target - have) / len(active)))
            for template_id in active:
                if have >= target:
                    break
                have += self.draw(template_id, min(share, target - have))
        return have


def main() -> None:
    judged = {t["template_id"]: t for t in read_jsonl(WORK / "templates_judged.jsonl")}
    working = [judged[p["template_id"]] for p in read_jsonl(WORK / "programs.jsonl") if p["status"] == "ok"]
    cells: dict = defaultdict(list)
    for template in working:
        cells[(template["category"], template["level"])].append(template["template_id"])

    with execute.Runner() as workers:
        filler = Filler(workers, working)
        count = lambda pred: sum(1 for row in filler.rows if pred(row))  # noqa: E731
        for (category, level), ids in sorted(cells.items()):
            have = filler.fill_to(ids, 0, TARGET_PER_CELL)
            print(f"  pass 1, category {category} {level}: {len(ids)} templates, {have} questions", flush=True)
        for category in CATEGORIES:
            ids = [t["template_id"] for t in working if t["category"] == category]
            have = filler.fill_to(ids, count(lambda r: r["category"] == category), TARGET_PER_CELL * len(LEVELS))
            print(f"pass 2, category {category}: {have} questions", flush=True)
        total = filler.fill_to([t["template_id"] for t in working], len(filler.rows), TARGET_TOTAL)
        print(f"pass 3: {total} questions", flush=True)

    write_jsonl(WORK / "instances.jsonl", filler.rows)
    print(f"{len(filler.rows)} questions from {len(filler.used)} templates; "
          f"{filler.excluder.rejected} candidates excluded as copies of Seed or Gold questions")
    if len(filler.rows) < TARGET_TOTAL:
        print(f"SHORT of the {TARGET_TOTAL} target: run `python -m energyx_max.generate --round 2` to add templates")


if __name__ == "__main__":
    main()
