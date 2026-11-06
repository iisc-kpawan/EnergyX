"""Q23: How did the median annual electricity EUI differ between office and education buildings in 2017?

Definition: annual 2017 EUI for office and for education buildings with valid floor-area data and
at least 90% of 2017's hours valid. The difference is office minus education and the ratio is
office over education.

Office and education are BDG2 primaryspaceusage "Office" and "Education". The median of an
even-sized group is the mean of the two middle values.

    python -m energyx_gold.tasks.category_05_performance.Q23
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as perf

TASK_ID = "Q23"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    cohort = perf.eui_cohort(context)
    office = perf.of_use(cohort, perf.OFFICE)
    education = perf.of_use(cohort, perf.EDUCATION)
    office_median = float(office["eui"].median())
    education_median = float(education["eui"].median())
    answer = {
        "office_median_eui": perf.rounded(office_median),
        "education_median_eui": perf.rounded(education_median),
        "difference": perf.rounded(office_median - education_median),
        "ratio": perf.rounded(office_median / education_median),
        "office_buildings": int(len(office)),
        "education_buildings": int(len(education)),
    }
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": answer,
            "units": {
                "office_median_eui": perf.EUI_UNIT,
                "education_median_eui": perf.EUI_UNIT,
                "difference": perf.EUI_UNIT,
                "ratio": "dimensionless",
                "office_buildings": "count (buildings)",
                "education_buildings": "count (buildings)",
            },
        },
        "evidence": {
            "definition": {
                "cohorts": f"{perf.USE_LABEL_COLUMN} = {perf.OFFICE!r} and {perf.EDUCATION!r}",
                "period": perf.period_note(),
                "floor_area": perf.AREA_NOTE,
                "qualifies_when": perf.coverage_note(),
                "eui": perf.EUI_NOTE,
                "difference": "office median minus education median",
                "ratio": "office median divided by education median",
            },
            "office": perf.group_numbers(office),
            "education": perf.group_numbers(education),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
