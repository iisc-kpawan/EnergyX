"""Q03: Which buildings have at least 12 consecutive months of electricity data?

Pinned definition: a building qualifies when some window of 12 consecutive calendar months within
January 2016 to December 2017 has valid readings for at least 95% of the window's hours. The 13
windows start in January to December 2016 and January 2017. The evidence also gives the count
under the stricter reading in which each of the 12 months must be at least 95% valid.

A window's hours are counted on the local clock, so windows that include February 2016 have 8,784
hours and the others 8,760. A reading is valid when its value isn't null.

    python -m energyx_gold.tasks.category_01_discovery.Q03
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_01_discovery._helpers import hours_between, month_start, set_result

TASK_ID = "Q03"
WINDOW_MONTHS = 12
MIN_COVERAGE_PCT = 95  # compared in integers: 100 * valid >= 95 * hours
FIRST_WINDOW = (2016, 1)
WINDOW_COUNT = 13  # starts January 2016 to January 2017
MONTHS = [(2016, m) for m in range(1, 13)] + [(2017, m) for m in range(1, 13)]

QUERY = """
SELECT building_id,
       CAST(EXTRACT(year FROM timestamp_local) AS INTEGER) AS year,
       CAST(EXTRACT(month FROM timestamp_local) AS INTEGER) AS month,
       COUNT(DISTINCT timestamp_local) AS valid_hours
FROM electricity
WHERE value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end
GROUP BY 1, 2, 3
"""

BUILDINGS = "SELECT building_id FROM buildings ORDER BY building_id"


def month_hours(year: int, month: int) -> int:
    return hours_between(month_start(year, month), month_start(year, month + 1))


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    start = month_start(*MONTHS[0])
    end = month_start(MONTHS[-1][0], MONTHS[-1][1] + 1)
    rows = context.sql(QUERY, {"start": start, "end": end})
    buildings = context.sql(BUILDINGS)["building_id"]
    valid = (rows.pivot_table(index="building_id", columns=["year", "month"], values="valid_hours",
                              aggfunc="sum")
             .reindex(index=buildings, columns=pd.MultiIndex.from_tuples(MONTHS), fill_value=0)
             .fillna(0).astype(int))
    hours = pd.Series([month_hours(y, m) for y, m in MONTHS], index=valid.columns)

    windows = []
    for k in range(WINDOW_COUNT):
        cols = valid.columns[k:k + WINDOW_MONTHS]
        covered = 100 * valid[cols].sum(axis=1) >= MIN_COVERAGE_PCT * int(hours[cols].sum())
        strict = (100 * valid[cols] >= MIN_COVERAGE_PCT * hours[cols]).all(axis=1)
        windows.append((cols[0], covered, strict))

    covered_any = pd.concat([c for _, c, _ in windows], axis=1).any(axis=1)
    passing = covered_any[covered_any].index
    strict_any = pd.concat([s for _, _, s in windows], axis=1).any(axis=1)
    strict_members = strict_any[strict_any].index
    first_window = {}
    for building in passing:
        for (year, month), covered, _ in windows:
            if covered[building]:
                first_window[building] = f"{year}-{month:02d}"
                break

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(passing),
        "evidence": {
            "definition": {
                "window_months": WINDOW_MONTHS,
                "window_starts": [f"{y}-{m:02d}" for (y, m), _, _ in windows],
                "min_window_coverage_pct": MIN_COVERAGE_PCT,
                "window_hours": {f"{y}-{m:02d}": int(hours[valid.columns[k:k + WINDOW_MONTHS]].sum())
                                 for k, ((y, m), _, _) in enumerate(windows)},
                "valid": "value not null (ZERO_MASKED rows are missing)",
                "clock": "site-local standard time",
            },
            "buildings_checked": len(buildings),
            "members_by_first_window": pd.Series(first_window).value_counts().sort_index().to_dict(),
            "strict_reading_every_month_95pct": {
                "count": len(strict_members),
                "not_in_answer": sorted(set(strict_members) - set(passing)),
                "answer_members_not_strict": len(set(passing) - set(strict_members)),
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
