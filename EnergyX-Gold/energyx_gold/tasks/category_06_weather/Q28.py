"""Q28: Which office buildings had an absolute Pearson correlation above 0.75 between hourly electricity consumption and outdoor air temperature during 2017?

Definition: office buildings with at least 90% of 2017's hours valid. For each, the Pearson
correlation between its valid 2017 readings and its site's air temperature over hours where both
are valid. A building qualifies when the absolute correlation is above 0.75. Every building's
correlation is in the evidence.

"Above" is strict: |r| > 0.75.

    python -m energyx_gold.tasks.category_06_weather.Q28
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as wx

TASK_ID = "Q28"
VARIABLE = wx.TEMPERATURE
METHOD = "pearson"
THRESHOLD = 0.75


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    return screened(context, TASK_ID, VARIABLE, METHOD, THRESHOLD)


def screened(context, task_id, variable, method, threshold) -> dict:
    found = wx.screen(context, variable, method, threshold)
    return {
        "task_id": task_id,
        "status": "success",
        "result": wx.set_result(found["members"]),
        "evidence": {
            "definition": {
                "cohort": f"{wx.USE_LABEL_COLUMN} = {wx.OFFICE!r}",
                "period": wx.period_note(),
                "qualifies_when": wx.coverage_note(),
                "variable": variable,
                "hours": f"hours with both a valid reading and a valid {variable}",
                "statistic": f"{method} correlation" + (", average ranks for ties" if method == "spearman" else ""),
                "selection": f"|correlation| > {threshold}",
            },
            **found["evidence"],
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
