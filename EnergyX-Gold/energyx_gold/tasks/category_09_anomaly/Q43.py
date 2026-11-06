"""Q43: Using the IQR method, how many hourly electricity readings of office buildings in London were identified as outliers during 2017, and on how many days did at least one outlier occur?

Pinned definition: London office buildings with 2017 readings. For each building, the quartiles of
its valid 2017 readings give the fences Q1 - 1.5 IQR and Q3 + 1.5 IQR, and a reading outside them
is an outlier. The first number counts outlier readings over all buildings; the second counts the
local calendar days of 2017 on which at least one of the buildings has an outlier. Each building's
counts are in the evidence.

- London is sites.city = 'London' (Mouse, Robin and Shrew). An office has primaryspaceusage Office.
- A building qualifies with at least one valid 2017 reading.
- Quartiles use linear interpolation (numpy's default). Outside means strictly below the lower
  fence or strictly above the upper fence.

    python -m energyx_gold.tasks.category_09_anomaly.Q43
"""

from __future__ import annotations

import numpy as np

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_09_anomaly import _helpers as helpers

TASK_ID = "Q43"
CITY = "London"
BUILDING_TYPE = "Office"
YEAR_START = "2017-01-01"
YEAR_END = "2018-01-01"  # exclusive
WHISKER = 1.5

COHORT_QUERY = """
SELECT b.building_id, b.site_id
FROM buildings b JOIN sites s ON s.site_id = b.site_id
WHERE s.city = $city AND b.building_type_raw = $building_type
ORDER BY b.building_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    candidates = context.sql(COHORT_QUERY, {"city": CITY, "building_type": BUILDING_TYPE})
    per_building, outlier_days, without_readings = {}, set(), []
    for site_id, group in candidates.groupby("site_id", sort=True):
        readings = helpers.valid_readings(context, list(group["building_id"]), YEAR_START, YEAR_END)
        for building in sorted(group["building_id"]):
            rows = readings[readings["building_id"] == building]
            if rows.empty:
                without_readings.append(building)
                continue
            values = rows["value"].to_numpy(float)
            q1, q3 = np.percentile(values, [25, 75])
            low, high = q1 - WHISKER * (q3 - q1), q3 + WHISKER * (q3 - q1)
            outside = (values < low) | (values > high)
            days = set(rows.loc[outside, "timestamp_local"].dt.date)
            outlier_days |= days
            per_building[building] = {
                "site_id": site_id,
                "valid_hours": int(len(values)),
                "q1_kwh": helpers.rounded(q1),
                "q3_kwh": helpers.rounded(q3),
                "lower_fence_kwh": helpers.rounded(low),
                "upper_fence_kwh": helpers.rounded(high),
                "outliers_low": int((values < low).sum()),
                "outliers_high": int((values > high).sum()),
                "outliers": int(outside.sum()),
                "days_with_outlier": len(days),
            }

    answer = {
        "outlier_readings": int(sum(v["outliers"] for v in per_building.values())),
        "days_with_outlier": len(outlier_days),
    }
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": {"outlier_readings": "count (hours)",
                                               "days_with_outlier": "count (days)"}},
        "evidence": {
            "definition": {
                "cohort": f"sites.city = {CITY}, primaryspaceusage {BUILDING_TYPE}, a valid 2017 reading",
                "period": f"{YEAR_START} 00:00 to 2017-12-31 23:00, site-local",
                "fences": "per building, Q1 - 1.5 IQR and Q3 + 1.5 IQR over its valid 2017 readings",
                "quartiles": "linear interpolation (numpy default)",
                "outlier": "strictly outside a fence",
                "days": "distinct local calendar dates with an outlier in any cohort building",
            },
            "cohort_size": len(per_building),
            "office_buildings_without_2017_readings": without_readings,
            "outliers_low_total": int(sum(v["outliers_low"] for v in per_building.values())),
            "outliers_high_total": int(sum(v["outliers_high"] for v in per_building.values())),
            "per_building": per_building,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
