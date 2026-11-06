"""Q24: Which buildings were above the 90th percentile of annual electricity EUI within their own primary-use category in 2017?

Definition: annual 2017 EUI for every building with valid floor-area data, at least 90% of 2017's
hours valid and a primaryspaceusage label. Within each label, a building qualifies when its EUI is
strictly above that label's 90th percentile. A label with fewer than 10 such buildings is left
out, and the evidence names it.

The percentile uses linear interpolation between neighbouring values.

    python -m energyx_gold.tasks.category_05_performance.Q24
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as perf

TASK_ID = "Q24"
PERCENTILE = 0.90
MIN_LABEL_SIZE = 10


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    cohort = perf.eui_cohort(context)
    labelled = cohort[cohort["use_label"].notna()]

    members: list[str] = []
    labels, left_out = [], []
    for label, group in labelled.groupby("use_label", sort=True):
        threshold = perf.percentile(group["eui"], PERCENTILE)
        above = sorted(group.index[group["eui"] > threshold].astype(str))
        entry = {"use_label": str(label), "buildings": int(len(group)), "p90_eui": perf.rounded(threshold)}
        if len(group) < MIN_LABEL_SIZE:
            left_out.append(str(label))
            entry["qualifying"] = []
        else:
            members.extend(above)
            entry["qualifying"] = above
        labels.append(entry)

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": perf.set_result(members),
        "evidence": {
            "definition": {
                "cohort": "every building with a primaryspaceusage label",
                "grouping": f"BDG2 primaryspaceusage ({perf.USE_LABEL_COLUMN})",
                "period": perf.period_note(),
                "floor_area": perf.AREA_NOTE,
                "qualifies_when": perf.coverage_note(),
                "eui": perf.EUI_NOTE,
                "percentile": PERCENTILE,
                "interpolation": "linear",
                "comparison": "strictly above the label's percentile",
                "min_label_size": MIN_LABEL_SIZE,
            },
            "cohort_size": int(len(labelled)),
            "buildings_without_use_label": int(cohort["use_label"].isna().sum()),
            "labels": labels,
            "labels_left_out": left_out,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
