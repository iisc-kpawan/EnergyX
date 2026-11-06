"""Q26: What was the mean outdoor air temperature in London during July 2017?

Definition: the mean of the valid hourly air temperatures of the London sites (Mouse, Robin and
Shrew) in July 2017, pooled over the sites. Each site's own mean is in the evidence.

The London sites are the sites whose city is London. July is 2017-07-01 00:00 to 2017-08-01 00:00
site-local standard time (UTC+0 for London, so no summer time shift).

    python -m energyx_gold.tasks.category_06_weather.Q26
"""

from __future__ import annotations

from datetime import datetime

from energyx_gold.core.context import GoldContext

from . import _helpers as wx

TASK_ID = "Q26"
CITY = "London"
EXPECTED_SITES = ["Mouse", "Robin", "Shrew"]
MONTH_START = datetime(2017, 7, 1)
MONTH_END = datetime(2017, 8, 1)  # exclusive
UNIT = "°C"

SITES_QUERY = "SELECT site_id FROM sites WHERE city = $city ORDER BY site_id"
WEATHER_QUERY = """
SELECT site_id, timestamp_local, air_temperature
FROM weather
WHERE site_id IN ({placeholders}) AND timestamp_local >= ? AND timestamp_local < ?
  AND air_temperature IS NOT NULL
ORDER BY site_id, timestamp_local
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    sites = sorted(context.sql(SITES_QUERY, {"city": CITY})["site_id"].astype(str))
    if sites != EXPECTED_SITES:
        raise ValueError(f"London sites are {sites}, expected {EXPECTED_SITES}")
    rows = context.sql(WEATHER_QUERY.format(placeholders=", ".join("?" for _ in sites)),
                       [*sites, MONTH_START, MONTH_END])
    if rows.duplicated(["site_id", "timestamp_local"]).any():
        raise ValueError("the weather has duplicate site hours")
    per_site = rows.groupby("site_id", sort=True)["air_temperature"].agg(["mean", "size", "min", "max"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": wx.rounded(rows["air_temperature"].mean()), "unit": UNIT},
        "evidence": {
            "definition": {
                "sites": f"sites with city = {CITY!r}",
                "period": f"{MONTH_START:%Y-%m-%d %H:%M} to {MONTH_END:%Y-%m-%d %H:%M} (end exclusive), site-local",
                "variable": wx.TEMPERATURE,
                "statistic": "mean of every valid site-hour, pooled over the sites",
            },
            "valid_site_hours": int(len(rows)),
            "hours_in_month": 744,
            "per_site": {
                str(site): {"mean_c": wx.rounded(r["mean"]), "valid_hours": int(r["size"]),
                            "min_c": wx.rounded(r["min"]), "max_c": wx.rounded(r["max"])}
                for site, r in per_site.iterrows()
            },
            "mean_of_site_means_c": wx.rounded(per_site["mean"].mean()),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
