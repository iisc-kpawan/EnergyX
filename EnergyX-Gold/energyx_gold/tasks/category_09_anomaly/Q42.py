"""Q42: How many hourly electricity readings of Fox_office_Alice in 2017 were more than 2 standard deviations above its annual mean?

Pinned definition: Fox_office_Alice's valid 2017 readings above its 2017 mean plus two standard
deviations (ddof 1). The test is one-sided.

- Valid readings are those with a value, 1 January 2017 00:00 to 31 December 2017 23:00, site-local.
- The mean and the sample standard deviation come from those readings.
- A reading counts when it is strictly above mean + 2 sd.

    python -m energyx_gold.tasks.category_09_anomaly.Q42
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_09_anomaly import _helpers as helpers

TASK_ID = "Q42"
BUILDING = "Fox_office_Alice"
YEAR_START = "2017-01-01"
YEAR_END = "2018-01-01"  # exclusive
SIGMA = 2.0
UNIT = "count (hours)"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    readings = helpers.valid_readings(context, [BUILDING], YEAR_START, YEAR_END)
    values = readings.set_index("timestamp_local")["value"]
    stats = helpers.sigma_count(values, SIGMA)
    above = stats["above"].sort_values(ascending=False)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": stats["count"], "unit": UNIT},
        "evidence": {
            "definition": {
                "building": BUILDING,
                "period": f"{YEAR_START} 00:00 to 2017-12-31 23:00, site-local",
                "sigma": SIGMA,
                "ddof": helpers.DDOF,
                "test": f"value > mean + {SIGMA:g} sd, one-sided",
                "readings": "valid hourly readings only",
            },
            "valid_hours": stats["valid_hours"],
            "mean_kwh": helpers.rounded(stats["mean"]),
            "sd_kwh": helpers.rounded(stats["sd"]),
            "threshold_kwh": helpers.rounded(stats["threshold"]),
            "max_reading_kwh": helpers.rounded(stats["max"]),
            "gap_between_threshold_and_nearest_reading_kwh": helpers.rounded(
                (values - stats["threshold"]).abs().min()),
            "readings_above_threshold": {helpers.hour_label(t): helpers.rounded(v) for t, v in above.items()},
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
