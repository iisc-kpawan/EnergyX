"""Q45: Using Isolation Forest, which education buildings in Berkeley, CA had the highest number of anomalous hourly electricity readings during May 2016?

Pinned definition: Education buildings in Berkeley, CA (Bear) with valid readings in May 2016. For
each, an Isolation Forest (scikit-learn, contamination "auto", random_state 0) is fitted on its
valid May 2016 readings with features value, hour of day and day of week, and the readings labelled
-1 are anomalous. The five buildings with the most anomalous readings qualify; every building's
count is in the evidence.

- Berkeley, CA is sites.city = 'Berkeley' (the Bear site). Education is primaryspaceusage Education.
- May 2016 is 1 May 00:00 to 31 May 23:00, site-local.
- One model per building, fitted on its valid hours in time order. Features: value, hour of day
  (0 to 23), day of week (Monday is 0). Every other setting is scikit-learn's default.
- Ties for fifth place are broken by building_id, as the shared convention says.

    python -m energyx_gold.tasks.category_09_anomaly.Q45
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_09_anomaly import _helpers as helpers

TASK_ID = "Q45"
CITY = "Berkeley"
BUILDING_TYPE = "Education"
PERIOD_START = "2016-05-01"
PERIOD_END = "2016-06-01"  # exclusive
CONTAMINATION = "auto"
TOP_N = 5

COHORT_QUERY = """
SELECT b.building_id, b.site_id
FROM buildings b JOIN sites s ON s.site_id = b.site_id
WHERE s.city = $city AND b.building_type_raw = $building_type
ORDER BY b.building_id
"""


def features(timestamps: pd.Series, values: pd.Series) -> np.ndarray:
    """Value, hour of day and day of week (Monday is 0) for each valid hour."""
    return np.column_stack([values.to_numpy(float), timestamps.dt.hour.to_numpy(float),
                            timestamps.dt.dayofweek.to_numpy(float)])


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    candidates = context.sql(COHORT_QUERY, {"city": CITY, "building_type": BUILDING_TYPE})
    sites = sorted(set(candidates["site_id"]))
    per_building, without_readings = {}, []
    for site_id in sites:
        ids = sorted(candidates.loc[candidates["site_id"] == site_id, "building_id"])
        readings = helpers.valid_readings(context, ids, PERIOD_START, PERIOD_END)
        for building in ids:
            rows = readings[readings["building_id"] == building]
            if rows.empty:
                without_readings.append(building)
                continue
            model = IsolationForest(contamination=CONTAMINATION, random_state=context.seed)
            labels = model.fit_predict(features(rows["timestamp_local"], rows["value"]))
            per_building[building] = {"valid_hours": int(len(rows)), "anomalies": int((labels == -1).sum())}

    ranked = sorted(per_building.items(), key=lambda item: (-item[1]["anomalies"], item[0]))
    members = sorted(name for name, _ in ranked[:TOP_N])
    cutoff = ranked[TOP_N - 1][1]["anomalies"]
    tied = sorted(name for name, v in per_building.items() if v["anomalies"] == cutoff)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": members, "count": len(members)},
        "evidence": {
            "definition": {
                "cohort": f"sites.city = {CITY} ({', '.join(sites)}), primaryspaceusage {BUILDING_TYPE}, "
                          "a valid May 2016 reading",
                "period": f"{PERIOD_START} 00:00 to 2016-05-31 23:00, site-local",
                "model": "sklearn IsolationForest, one per building, other settings default",
                "contamination": CONTAMINATION,
                "random_state": context.seed,
                "sklearn_version": sklearn.__version__,
                "features": ["value", "hour of day", "day of week (Monday = 0)"],
                "row_order": "valid hours in time order",
                "anomaly": "label -1",
                "top_n": TOP_N,
                "tie_break": "building_id",
            },
            "cohort_size": len(per_building),
            "education_buildings_without_may_2016_readings": without_readings,
            "fifth_place_anomalies": cutoff,
            "buildings_at_fifth_place_count": tied,
            "sixth_place": {"building_id": ranked[TOP_N][0], "anomalies": ranked[TOP_N][1]["anomalies"]}
            if len(ranked) > TOP_N else None,
            "ranking": [name for name, _ in ranked],
            "per_building": dict(sorted(per_building.items())),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
