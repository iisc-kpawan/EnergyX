"""Q25: How did the median annual electricity EUI of education buildings in the United States that were 10 years old or newer compare with that of older education buildings in 2017?

Definition: US education buildings with valid floor-area data, a year built and at least 90% of
2017's hours valid. Buildings built in 2007 or later are 10 years old or newer in 2017; the rest
are older. The difference is newer minus older and the ratio is newer over older. Buildings with
no year built are left out and counted in the evidence.

US means the site's country is US. Education is BDG2 primaryspaceusage "Education".

    python -m energyx_gold.tasks.category_05_performance.Q25
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as perf

TASK_ID = "Q25"
COUNTRY = "US"
NEWER_FROM_YEAR = 2007  # built in 2007 or later: 10 years old or newer in 2017


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    cohort = perf.of_use(perf.eui_cohort(context), perf.EDUCATION)
    cohort = cohort[cohort["site_country"] == COUNTRY]
    no_year = cohort[cohort["year_built"].isna()]
    dated = cohort[cohort["year_built"].notna()]
    newer = dated[dated["year_built"] >= NEWER_FROM_YEAR]
    older = dated[dated["year_built"] < NEWER_FROM_YEAR]
    newer_median = float(newer["eui"].median())
    older_median = float(older["eui"].median())
    if newer_median == older_median:
        raise ValueError("the two medians are equal")
    answer = {
        "newer_median_eui": perf.rounded(newer_median),
        "older_median_eui": perf.rounded(older_median),
        "difference": perf.rounded(newer_median - older_median),
        "ratio": perf.rounded(newer_median / older_median),
        "newer_buildings": int(len(newer)),
        "older_buildings": int(len(older)),
        "higher_median_group": "newer" if newer_median > older_median else "older",
    }
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": answer,
            "units": {
                "newer_median_eui": perf.EUI_UNIT,
                "older_median_eui": perf.EUI_UNIT,
                "difference": perf.EUI_UNIT,
                "ratio": "dimensionless",
                "newer_buildings": "count (buildings)",
                "older_buildings": "count (buildings)",
            },
        },
        "evidence": {
            "definition": {
                "cohort": f"{perf.USE_LABEL_COLUMN} = {perf.EDUCATION!r} at sites with country_canonical = {COUNTRY!r}",
                "period": perf.period_note(),
                "floor_area": perf.AREA_NOTE,
                "qualifies_when": perf.coverage_note(),
                "eui": perf.EUI_NOTE,
                "newer": f"year_built >= {NEWER_FROM_YEAR}",
                "older": f"year_built < {NEWER_FROM_YEAR}",
                "difference": "newer median minus older median",
                "ratio": "newer median divided by older median",
            },
            "cohort_with_year_built": int(len(dated)),
            "left_out_no_year_built": int(len(no_year)),
            "newer": perf.group_numbers(newer) | {"year_built_range": _years(newer)},
            "older": perf.group_numbers(older) | {"year_built_range": _years(older)},
            "queries": list(context.queries_run),
        },
    }


def _years(frame) -> list[int]:
    return [int(frame["year_built"].min()), int(frame["year_built"].max())]


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
