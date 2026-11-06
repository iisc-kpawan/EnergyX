"""Q27: What was the Pearson correlation between hourly electricity consumption of `Fox_office_Alice` and outdoor air temperature during 2017?

Definition: Pearson correlation between Fox_office_Alice's valid hourly readings in 2017 and the
Fox site's air temperature at the same hours, over hours where both are valid.

    python -m energyx_gold.tasks.category_06_weather.Q27
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as wx

TASK_ID = "Q27"
BUILDING = "Fox_office_Alice"
SITE = "Fox"
VARIABLE = wx.TEMPERATURE
METHOD = "pearson"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    return single_building(context, TASK_ID, BUILDING, SITE, VARIABLE, METHOD)


def single_building(context, task_id, building, site, variable, method) -> dict:
    frame = wx.paired(context, [building], variable)
    if frame.empty:
        raise ValueError(f"{building}: no hour of 2017 has both a reading and {variable}")
    correlation = wx.STATISTICS[method](frame, variable)
    return {
        "task_id": task_id,
        "status": "success",
        "result": {"answer": wx.rounded(correlation), "unit": None},
        "evidence": {
            "definition": {
                "building": building,
                "site": site,
                "variable": variable,
                "period": wx.period_note(),
                "hours": f"hours with both a valid reading and a valid {variable}",
                "statistic": f"{method} correlation" + (", average ranks for ties" if method == "spearman" else ""),
            },
            "paired_hours": int(len(frame)),
            "hours_in_year": wx.HOURS_IN_YEAR,
            "mean_kwh": wx.rounded(frame["value"].mean()),
            f"mean_{variable}_c": wx.rounded(frame[variable].mean()),
            "other_statistic": {
                ("spearman" if method == "pearson" else "pearson"):
                    wx.rounded(wx.STATISTICS["spearman" if method == "pearson" else "pearson"](frame, variable)),
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
