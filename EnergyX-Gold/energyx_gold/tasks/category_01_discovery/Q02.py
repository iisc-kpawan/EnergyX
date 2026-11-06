"""Q02: What building-use categories are represented in the United States, and how many are there?

Pinned definition: distinct BDG2 primaryspaceusage labels among buildings whose site's country is
US. Buildings with no label are left out. The count is in the result. The label is stored as
`building_type_raw`. The five anonymous sites have no country, so they don't count.

    python -m energyx_gold.tasks.category_01_discovery.Q02
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_01_discovery._helpers import set_result

TASK_ID = "Q02"
COUNTRY = "US"
TYPE_COLUMN = "building_type_raw"

QUERY = """
SELECT b.building_type_raw AS building_type, COUNT(*) AS buildings
FROM buildings b JOIN sites s USING (site_id)
WHERE s.country_canonical = $country
GROUP BY b.building_type_raw
ORDER BY building_type NULLS LAST
"""

ANONYMOUS_QUERY = """
SELECT DISTINCT b.building_type_raw AS building_type
FROM buildings b JOIN sites s USING (site_id)
WHERE s.country_canonical IS NULL AND b.building_type_raw IS NOT NULL
ORDER BY building_type
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = context.sql(QUERY, {"country": COUNTRY})
    typed = rows[rows["building_type"].notna()]
    anonymous = set(context.sql(ANONYMOUS_QUERY)["building_type"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(typed["building_type"]),
        "evidence": {
            "definition": {
                "country": COUNTRY,
                "country_column": "sites.country_canonical",
                "type_column": TYPE_COLUMN,
                "source_column": "BDG2 primaryspaceusage",
                "unlabelled_buildings_count": False,
            },
            "buildings_by_type": {row.building_type: int(row.buildings) for row in typed.itertuples()},
            "buildings_at_us_sites": int(rows["buildings"].sum()),
            "unlabelled_buildings_at_us_sites": int(rows.loc[rows["building_type"].isna(), "buildings"].sum()),
            "labels_added_by_anonymous_sites": sorted(anonymous - set(typed["building_type"])),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
