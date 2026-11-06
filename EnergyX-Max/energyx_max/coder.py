"""Stage 3, Coding Agent: write and test the answer program for each accepted template.

    uv run python -m energyx_max.coder [--limit N]

For each accepted template the Coding Agent writes answer(data, **slots). The program is run on three
sampled slot combinations; if none gives a usable answer, the agent gets the error once to fix it. A
template whose program still fails is dropped. Programs are saved to programs/<template_id>.py and
results to work/programs.jsonl. Resumable.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from . import config, execute, slots as slot_space
from .common import CATEGORIES, MODELS, PROGRAMS, WORK, append_jsonl, chat, extract_code, read_jsonl
from .prompts import CODER_SYSTEM, coder_repair, coder_user

IN = WORK / "templates_judged.jsonl"
OUT = WORK / "programs.jsonl"
TRIALS = config.setting("pipeline", "program_trials")
ENOUGH = config.setting("pipeline", "working_templates_per_category")
MAX_TOKENS = 1200
BUDGET = config.setting("pipeline", "coding_attempts_per_run")


def fills_for(template: dict, n: int, seed: str) -> list[dict]:
    rng = random.Random(seed)
    fills = []
    for _ in range(n * 10):
        fill = slot_space.sample(template["slots"], rng)
        if fill and fill not in fills:
            fills.append(fill)
        if len(fills) == n:
            break
    return fills


def trial(runner, code: str, fills: list[dict]) -> list[dict]:
    return runner.run_many(code, [{k: v["value"] for k, v in fill.items()} for fill in fills])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    results = read_jsonl(OUT)
    attempts = Counter(row["template_id"] for row in results)
    working_ids = {row["template_id"] for row in results if row["status"] == "ok"}
    no_values = {row["template_id"] for row in results if row["status"] == "no valid slot values"}
    # A template whose program failed once gets one more attempt, after the untried templates.
    done = working_ids | no_values | {t for t, n in attempts.items() if n >= 2}
    judged = {t["template_id"]: t for t in read_jsonl(IN)}
    working = Counter(judged[t]["category"] for t in working_ids if t in judged)
    queue: dict = defaultdict(list)
    for template in sorted(judged.values(), key=lambda t: attempts[t["template_id"]]):
        if template["judge_verdict"] == "accept" and template["template_id"] not in done:
            queue[template["category"]].append(template)
    print(f"{sum(map(len, queue.values()))} templates left to code with {MODELS['coder']}; "
          f"stopping a category at {ENOUGH} working programs")
    PROGRAMS.mkdir(parents=True, exist_ok=True)
    attempted = 0
    with execute.Runner() as runner:
        while True:
            open_categories = [c for c in queue if queue[c] and working[c] < ENOUGH]
            limit = args.limit or BUDGET
            if not open_categories or (limit and attempted >= limit):
                break
            category = min(open_categories, key=lambda c: (working[c], c))
            template = queue[category].pop(0)
            attempted += 1
            if code_template(runner, template):
                working[category] += 1
            if attempted % 10 == 0:
                print(f"  {attempted} coded; working per category: "
                      + ", ".join(f"{c}:{working[c]}" for c in sorted(CATEGORIES)), flush=True)
    print("done; working per category: " + ", ".join(f"{c}:{working[c]}" for c in sorted(CATEGORIES)))


def code_template(runner, template: dict) -> bool:
    """Write, test and if needed repair one template's program. Returns whether it works."""
    fills = fills_for(template, TRIALS, template["template_id"])
    if not fills:
        append_jsonl(OUT, {"template_id": template["template_id"], "status": "no valid slot values"})
        return False
    example = {k: v["value"] for k, v in fills[0].items()}
    prompt = coder_user(template, example)
    code = extract_code(chat("coder", CODER_SYSTEM, prompt, max_tokens=MAX_TOKENS))
    results = trial(runner, code, fills)
    repaired = False
    if not any(r["ok"] for r in results):
        error = next(r["error"] for r in results)
        reply = chat("coder", CODER_SYSTEM, prompt + "\n\nYour code:\n```python\n" + code + "```\n\n"
                     + coder_repair(error), max_tokens=MAX_TOKENS)
        code, repaired = extract_code(reply), True
        results = trial(runner, code, fills)
    passed = sum(r["ok"] for r in results)
    if passed:
        header = (f'"""EnergyX-Max program for template {template["template_id"]}.\n\n'
                  f'{template["template"]}\n\nWritten by the Coding Agent ({MODELS["coder"]}); '
                  f'not manually validated.\n"""\n\n')
        (PROGRAMS / f"{template['template_id']}.py").write_text(header + code)
    append_jsonl(OUT, {"template_id": template["template_id"], "status": "ok" if passed else "failed",
                       "repaired": repaired, "trials_passed": passed, "trials": len(results),
                       "coder": MODELS["coder"], "errors": [r["error"][-300:] for r in results if not r["ok"]][:2]})
    return bool(passed)


if __name__ == "__main__":
    main()
