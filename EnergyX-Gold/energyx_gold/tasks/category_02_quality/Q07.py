"""Q07: What percentage of hourly readings are missing for Peacock_office_Annie over its observed period?

Pinned definition: Peacock_office_Annie's observation period runs from its first valid reading to
its last, inclusive. Missing readings are the hours in that period with no valid reading, as a
percentage of its hours. An hour is missing when it has no row or a ZERO_MASKED row.

    python -m energyx_gold.tasks.category_02_quality.Q07
"""

from __future__ import annotations

from datetime import timedelta

from energyx_gold.core.context import GoldContext

TASK_ID = "Q07"
BUILDING_ID = "Peacock_office_Annie"
DECIMALS = 6
UNIT = "% of hours"

QUERY = """
SELECT MIN(timestamp_local) AS first_valid,
       MAX(timestamp_local) AS last_valid,
       COUNT(DISTINCT timestamp_local) AS valid_hours,
       date_diff('hour', MIN(timestamp_local), MAX(timestamp_local)) + 1 AS observed_hours
FROM electricity
WHERE building_id = $building_id AND value IS NOT NULL
"""

ZERO_MASKED = """
SELECT COUNT(*) AS zero_masked
FROM electricity
WHERE building_id = $building_id AND quality_flag = 'ZERO_MASKED'
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    row = context.sql(QUERY, {"building_id": BUILDING_ID}).iloc[0]
    valid_hours = int(row["valid_hours"])
    if valid_hours == 0:
        raise ValueError(f"no valid readings for {BUILDING_ID}")
    observed_hours = int(row["observed_hours"])
    missing_hours = observed_hours - valid_hours
    period = context.period
    period_hours = (period.end - period.start) // timedelta(hours=1) + 1
    zero_masked = int(context.sql(ZERO_MASKED, {"building_id": BUILDING_ID}).iloc[0]["zero_masked"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": round(100 * missing_hours / observed_hours, DECIMALS), "unit": UNIT},
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "observation_period": "first valid reading to last valid reading, inclusive",
                "missing": "no row, or a ZERO_MASKED row",
            },
            "first_valid": str(row["first_valid"]),
            "last_valid": str(row["last_valid"]),
            "observed_hours": observed_hours,
            "valid_hours": valid_hours,
            "missing_hours": missing_hours,
            "zero_masked_rows": zero_masked,
            "snapshot_hours": period_hours,
            "percent_over_full_snapshot": round(100 * (period_hours - valid_hours) / period_hours, DECIMALS),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
