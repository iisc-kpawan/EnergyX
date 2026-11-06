"""Q06: Does Bear_education_Alfredo have duplicated electricity timestamps, and how many?

Pinned definition: timestamps that appear more than once in Bear_education_Alfredo's electricity
readings. 0 means there are no duplicates. Every row counts, ZERO_MASKED rows included.

    python -m energyx_gold.tasks.category_02_quality.Q06
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

TASK_ID = "Q06"
BUILDING_ID = "Bear_education_Alfredo"
UNIT = "count (duplicate timestamps)"

DUPLICATES = """
SELECT timestamp_local, COUNT(*) AS row_count
FROM electricity
WHERE building_id = $building_id
GROUP BY timestamp_local
HAVING COUNT(*) > 1
ORDER BY timestamp_local
"""

TOTALS = """
SELECT COUNT(*) AS row_count,
       COUNT(DISTINCT timestamp_local) AS timestamps,
       COUNT(DISTINCT timestamp_utc) AS utc_timestamps,
       COUNT(DISTINCT stream_id) AS streams,
       COUNT(value) AS valid_rows
FROM electricity
WHERE building_id = $building_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    duplicates = context.sql(DUPLICATES, {"building_id": BUILDING_ID})
    totals = context.sql(TOTALS, {"building_id": BUILDING_ID}).iloc[0]
    if int(totals["row_count"]) == 0:
        raise ValueError(f"no electricity rows for {BUILDING_ID}")
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": len(duplicates), "unit": UNIT},
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "duplicate": "a timestamp_local with more than one electricity row",
                "rows_counted": "every row, ZERO_MASKED included",
            },
            "has_duplicates": len(duplicates) > 0,
            "rows": int(totals["row_count"]),
            "valid_rows": int(totals["valid_rows"]),
            "distinct_local_timestamps": int(totals["timestamps"]),
            "distinct_utc_timestamps": int(totals["utc_timestamps"]),
            "streams": int(totals["streams"]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
