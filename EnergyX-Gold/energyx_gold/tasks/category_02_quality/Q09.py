"""Q09: Which buildings have more than 72 consecutive hours of zero electricity readings?

Pinned definition: a zero reading is an hour BDG2's cleaning masked (quality_flag ZERO_MASKED; the
snapshot has no stored 0 values). A building qualifies when more than 72 consecutive clock hours
are all zero readings. The evidence also gives the count when missing hours inside a run are
skipped rather than breaking it (the EXG-09 rule).

    python -m energyx_gold.tasks.category_02_quality.Q09
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_02_quality._helpers import ZERO_FLAG, set_result, zero_runs

TASK_ID = "Q09"
MAX_ZERO_RUN_HOURS = 72  # a run qualifies when it is longer than this


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    clock = zero_runs(context, skip_missing=False)
    members = clock[clock["longest_zero_run_hours"] > MAX_ZERO_RUN_HOURS]
    skipping = zero_runs(context, skip_missing=True)
    skipping_members = set(skipping.loc[skipping["longest_zero_run_hours"] > MAX_ZERO_RUN_HOURS, "building_id"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(members["building_id"]),
        "evidence": {
            "definition": {
                "zero_reading": f"an hour flagged {ZERO_FLAG} (a raw zero removed by cleaning)",
                "zero_run_hours_greater_than": MAX_ZERO_RUN_HOURS,
                "run": "consecutive clock hours; a missing hour or a valid reading ends it",
                "clock": "site-local standard time",
            },
            "buildings_with_a_zero_reading": len(clock),
            "min_longest_zero_run_among_members": int(members["longest_zero_run_hours"].min()) if len(members) else None,
            "max_longest_zero_run_among_members": int(members["longest_zero_run_hours"].max()) if len(members) else None,
            "skipping_missing_hours_rule": {
                "count": len(skipping_members),
                "added": sorted(skipping_members - set(members["building_id"])),
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
