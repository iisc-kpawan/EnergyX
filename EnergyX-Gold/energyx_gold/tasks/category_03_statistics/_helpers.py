"""Helpers shared by the category 3 tasks: rounding, local-time windows and the 90% coverage rule.

A valid reading is a row in `electricity` whose value isn't null. Masked zeros and hours with no
row are both missing. Readings are kWh per hour.
"""

from __future__ import annotations

from datetime import datetime

DECIMALS = 6
COVERAGE = 0.9  # share of the year's hours that must be valid for a yearly measure


def rounded(value) -> float:
    """A float rounded to the result precision."""
    return round(float(value), DECIMALS)


def month_window(year: int, month: int, months: int = 1) -> tuple[datetime, datetime]:
    """Start (inclusive) and end (exclusive) of `months` local calendar months from year-month."""
    end_index = (year * 12 + month - 1) + months
    return datetime(year, month, 1), datetime(end_index // 12, end_index % 12 + 1, 1)


def year_window(year: int) -> tuple[datetime, datetime]:
    """Start (inclusive) and end (exclusive) of a local calendar year."""
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def window_params(window: tuple[datetime, datetime]) -> dict:
    """The start and end of a window as query parameters."""
    return {"start": window[0], "end": window[1]}


def window_hours(window: tuple[datetime, datetime]) -> int:
    """The number of hours in a window."""
    return int((window[1] - window[0]).total_seconds() // 3600)


def min_valid_hours(window: tuple[datetime, datetime], share: float = COVERAGE) -> float:
    """The fewest valid hours a building needs in a window: `share` of its hours."""
    return share * window_hours(window)
