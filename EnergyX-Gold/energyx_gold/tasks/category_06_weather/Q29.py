"""Q29: What was the Spearman correlation between hourly electricity consumption of `Fox_office_Alice` and dew-point temperature during 2017?

Definition: Spearman rank correlation (average ranks for ties) between Fox_office_Alice's valid
hourly readings in 2017 and the Fox site's dew-point temperature at the same hours, over hours
where both are valid.

    python -m energyx_gold.tasks.category_06_weather.Q29
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as wx
from .Q27 import single_building

TASK_ID = "Q29"
BUILDING = "Fox_office_Alice"
SITE = "Fox"
VARIABLE = wx.DEW_POINT
METHOD = "spearman"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    return single_building(context, TASK_ID, BUILDING, SITE, VARIABLE, METHOD)


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
