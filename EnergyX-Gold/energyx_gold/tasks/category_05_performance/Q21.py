"""Q21: What was the annual electricity EUI of `Fox_office_Alice` in 2017?

Definition: Fox_office_Alice's valid 2017 readings summed and divided by its floor area in m2.

A reading is valid when its value isn't null. The definition sets no coverage bar, so every valid
hour counts. The building's 2017 coverage is in the evidence.

    python -m energyx_gold.tasks.category_05_performance.Q21
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as perf

TASK_ID = "Q21"
BUILDING = "Fox_office_Alice"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    frame = perf.valid_area(perf.annual_eui(context, building_id=BUILDING))
    if len(frame) != 1:
        raise ValueError(f"{BUILDING}: {len(frame)} rows with 2017 readings and a valid floor area, expected 1")
    row = frame.loc[BUILDING]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": perf.rounded(row["eui"]), "unit": perf.EUI_UNIT},
        "evidence": {
            "definition": {
                "building": BUILDING,
                "period": perf.period_note(),
                "eui": perf.EUI_NOTE,
                "valid_reading": "value is not null",
                "coverage_bar": "none",
            },
            "total_kwh": perf.rounded(row["total_kwh"]),
            "floor_area_m2": perf.rounded(row["floor_area_m2"]),
            "valid_hours": int(row["valid_hours"]),
            "hours_in_year": perf.HOURS_IN_YEAR,
            "coverage": perf.rounded(row["coverage"]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
