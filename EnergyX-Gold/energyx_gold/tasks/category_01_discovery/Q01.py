"""Q01: How many buildings are located in London?

Pinned definition: buildings in the metadata whose site's city is London (the Mouse, Robin and
Shrew sites).

    python -m energyx_gold.tasks.category_01_discovery.Q01
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

TASK_ID = "Q01"
CITY = "London"
UNIT = "count (buildings)"

QUERY = """
SELECT s.site_id, s.site_name, COUNT(*) AS buildings
FROM buildings b JOIN sites s USING (site_id)
WHERE s.city = $city
GROUP BY s.site_id, s.site_name
ORDER BY s.site_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = context.sql(QUERY, {"city": CITY})
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": int(rows["buildings"].sum()), "unit": UNIT},
        "evidence": {
            "definition": {"city": CITY, "city_column": "sites.city"},
            "buildings_by_site": {row.site_id: int(row.buildings) for row in rows.itertuples()},
            "site_names": {row.site_id: row.site_name for row in rows.itertuples()},
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
