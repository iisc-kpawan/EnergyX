"""Q22: Which five office buildings had the highest annual electricity EUI in 2017?

Definition: annual 2017 EUI for every office building with valid floor-area data and at least 90%
of 2017's hours valid. The five highest qualify; the ranking is in the evidence.

An office building has BDG2 primaryspaceusage "Office". A tie is broken by building_id.

    python -m energyx_gold.tasks.category_05_performance.Q22
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as perf

TASK_ID = "Q22"
TOP_N = 5
RANKING_SHOWN = 10


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    cohort = perf.of_use(perf.eui_cohort(context), perf.OFFICE)
    ordered = perf.by_eui(cohort)
    if len(ordered) <= TOP_N:
        raise ValueError(f"only {len(ordered)} office buildings, need more than {TOP_N}")
    if ordered["eui"].iloc[TOP_N - 1] == ordered["eui"].iloc[TOP_N]:
        raise ValueError(f"buildings {TOP_N} and {TOP_N + 1} tie on EUI")
    members = list(ordered.index[:TOP_N])
    ranked = perf.ranking(cohort, limit=RANKING_SHOWN)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": perf.set_result(members),
        "evidence": {
            "definition": {
                "cohort": f"{perf.USE_LABEL_COLUMN} = {perf.OFFICE!r}",
                "period": perf.period_note(),
                "floor_area": perf.AREA_NOTE,
                "qualifies_when": perf.coverage_note(),
                "eui": perf.EUI_NOTE,
                "ranking": f"highest EUI first, ties broken by building_id, top {TOP_N}",
            },
            "cohort_size": int(len(cohort)),
            "ranking": ranked,
            "lowest_selected_eui": ranked[TOP_N - 1]["eui"],
            "highest_rejected_eui": ranked[TOP_N]["eui"],
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
