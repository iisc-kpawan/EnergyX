"""Q08: Which education buildings have electricity gaps longer than seven days?

Pinned definition: education buildings with a run of more than 168 consecutive missing hours
between two valid readings. Missing runs before the first valid reading or after the last one
don't count. An education building has BDG2 primaryspaceusage "Education". An hour is missing
when it has no row or a ZERO_MASKED row.

    python -m energyx_gold.tasks.category_02_quality.Q08
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_02_quality._helpers import inner_gaps, set_result

TASK_ID = "Q08"
BUILDING_TYPE = "Education"
MAX_GAP_HOURS = 168  # a gap qualifies when it is longer than this

EDGE_QUERY = """
SELECT e.building_id,
       date_diff('hour', $period_start, MIN(e.timestamp_local)) AS leading_missing,
       date_diff('hour', MAX(e.timestamp_local), $period_end) AS trailing_missing
FROM electricity e JOIN buildings b ON b.building_id = e.building_id
WHERE e.value IS NOT NULL AND b.building_type_raw = $building_type
GROUP BY e.building_id
"""

COHORT_QUERY = "SELECT COUNT(*) AS n FROM buildings WHERE building_type_raw = $building_type"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    gaps = inner_gaps(context, building_type=BUILDING_TYPE)
    longest = gaps.groupby("building_id")["gap_hours"].max()
    members = longest[longest > MAX_GAP_HOURS]
    cohort = int(context.sql(COHORT_QUERY, {"building_type": BUILDING_TYPE}).iloc[0]["n"])
    period = context.period
    edges = context.sql(EDGE_QUERY, {"building_type": BUILDING_TYPE,
                                     "period_start": period.start, "period_end": period.end})
    edge_only = edges[((edges["leading_missing"] > MAX_GAP_HOURS) | (edges["trailing_missing"] > MAX_GAP_HOURS))
                      & ~edges["building_id"].isin(members.index)]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(members.index),
        "evidence": {
            "definition": {
                "building_type": BUILDING_TYPE,
                "type_column": "building_type_raw",
                "gap_hours_greater_than": MAX_GAP_HOURS,
                "missing": "no row, or a ZERO_MASKED row",
                "edge_runs_count": False,
                "clock": "site-local standard time",
            },
            "education_buildings": cohort,
            "education_buildings_with_a_valid_reading": len(edges),
            "longest_gap_hours_by_member": {b: int(h) for b, h in members.sort_index().items()},
            "added_if_edge_runs_counted": sorted(edge_only["building_id"]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
