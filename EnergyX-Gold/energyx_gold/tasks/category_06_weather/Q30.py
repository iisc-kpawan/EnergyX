"""Q30: Which office buildings had an absolute Spearman correlation above 0.75 between hourly electricity consumption and dew-point temperature during 2017?

Definition: office buildings with at least 90% of 2017's hours valid. For each, the Spearman
correlation between its valid 2017 readings and its site's dew-point temperature over hours where
both are valid. A building qualifies when the absolute correlation is above 0.75. Every
building's correlation is in the evidence.

Ties get average ranks. "Above" is strict: |rho| > 0.75.

    python -m energyx_gold.tasks.category_06_weather.Q30
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext

from . import _helpers as wx
from .Q28 import screened

TASK_ID = "Q30"
VARIABLE = wx.DEW_POINT
METHOD = "spearman"
THRESHOLD = 0.75


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    return screened(context, TASK_ID, VARIABLE, METHOD, THRESHOLD)


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
