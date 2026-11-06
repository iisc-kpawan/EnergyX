"""Run the whole EnergyX-Max pipeline, stage by stage.

    uv run python -m energyx_max.run_all [--from STAGE]

Every stage resumes where it stopped, so after an interruption run the same command again. Stages
run one after another because the three agents use different local models, and loading them side
by side would exhaust memory.

If instantiation ends short of the 6,000-question target, the Generator writes another round of
templates for the categories and levels with the fewest working templates, and the Judge, Coding
and instantiation stages run again (at most MAX_ROUNDS rounds in all).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime

STAGES = ["seeds", "generate", "judge", "coder", "instantiate", "build", "validate"]
MAX_ROUNDS = 3


def run(stage: str, *args: str) -> None:
    print(f"== {stage} {' '.join(args)} {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    sys.argv = [f"energyx_max.{stage}", *args]
    try:
        importlib.import_module(f"energyx_max.{stage}").main()
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise


def short_of_target() -> bool:
    from .common import CATEGORIES, LEVELS, TARGET_PER_CELL, WORK, read_jsonl
    return len(read_jsonl(WORK / "instances.jsonl")) < TARGET_PER_CELL * len(CATEGORIES) * len(LEVELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    args = parser.parse_args()
    stages = STAGES[STAGES.index(args.start):]
    for stage in [s for s in stages if s in ("seeds", "generate", "judge", "coder", "instantiate")]:
        run(stage)
    for round_ in range(2, MAX_ROUNDS + 1):
        if not short_of_target():
            break
        for stage, extra in (("generate", ("--round", str(round_))), ("judge", ()), ("coder", ()), ("instantiate", ())):
            run(stage, *extra)
    for stage in ("build", "validate"):
        run(stage)
    print(f"== done {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
