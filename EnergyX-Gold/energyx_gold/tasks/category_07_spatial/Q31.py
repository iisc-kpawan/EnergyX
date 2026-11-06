"""Q31: How many buildings are available in each country?

Pinned definition: buildings in the metadata counted by their site's country. The five anonymous
sites have no country; their buildings are counted under "no country recorded".

- Every building in the metadata counts, whether or not it has a reading.
- The country is the site's country_canonical, so BDG2's US state names all count as US.
- Countries are in alphabetical order, with the no-country group last.

    python -m energyx_gold.tasks.category_07_spatial.Q31
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

TASK_ID = "Q31"
NO_COUNTRY = "no country recorded"
UNIT = "count (buildings)"

QUERY = """
SELECT COALESCE(s.country_canonical, $no_country) AS country,
       COUNT(*) AS buildings,
       COUNT(DISTINCT b.site_id) AS sites,
       STRING_AGG(DISTINCT b.site_id, ', ' ORDER BY b.site_id) AS site_ids
FROM buildings b
JOIN sites s ON s.site_id = b.site_id
GROUP BY 1
ORDER BY country
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = context.sql(QUERY, {"no_country": NO_COUNTRY}).set_index("country")
    named = [country for country in rows.index if country != NO_COUNTRY]
    order = named + ([NO_COUNTRY] if NO_COUNTRY in rows.index else [])
    answer = {country: int(rows.loc[country, "buildings"]) for country in order}
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": {country: UNIT for country in answer}},
        "evidence": {
            "definition": {
                "population": "every building in the metadata",
                "country": "sites.country_canonical",
                "no_country_label": NO_COUNTRY,
                "readings_required": False,
            },
            "total_buildings": int(rows["buildings"].sum()),
            "countries": len(named),
            "sites_by_country": {
                country: {"sites": int(rows.loc[country, "sites"]), "site_ids": rows.loc[country, "site_ids"]}
                for country in order
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
