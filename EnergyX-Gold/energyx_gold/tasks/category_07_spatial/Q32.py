"""Q32: Which site has the largest number of education buildings?

Pinned definition: the site with the most buildings whose primaryspaceusage is Education. Every
site's count is in the evidence.

- Every Education building in the metadata counts, whether or not it has a reading.
- A tie would be broken by site_id, so the answer is one site.

    python -m energyx_gold.tasks.category_07_spatial.Q32
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_07_spatial._helpers import EDUCATION

TASK_ID = "Q32"

QUERY = """
SELECT site_id, COUNT(*) AS education_buildings
FROM buildings
WHERE building_type_raw = $building_type
GROUP BY site_id
ORDER BY education_buildings DESC, site_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = context.sql(QUERY, {"building_type": EDUCATION})
    winner = str(rows.iloc[0]["site_id"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": winner},
        "evidence": {
            "definition": {
                "building_type": f"BDG2 primaryspaceusage {EDUCATION}",
                "population": "every building in the metadata",
                "readings_required": False,
                "tie_break": "site_id",
            },
            "winner_education_buildings": int(rows.iloc[0]["education_buildings"]),
            "runner_up": str(rows.iloc[1]["site_id"]),
            "runner_up_education_buildings": int(rows.iloc[1]["education_buildings"]),
            "education_buildings_by_site": {
                str(row.site_id): int(row.education_buildings) for row in rows.itertuples()
            },
            "education_buildings_total": int(rows["education_buildings"].sum()),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
