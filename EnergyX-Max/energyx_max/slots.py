"""The slot vocabulary for EnergyX-Max templates, and valid values for each slot.

A template names its slots in braces, e.g. "What was the total electricity consumption of
{building_id} in {month}?". Each slot has a value the program receives and a text that goes into
the question. Values are drawn from the data so that every combination can be answered: a building
comes with a year in which at least 90% of its hours are valid, a building type comes with a place
that has at least three such buildings, and so on.
"""

from __future__ import annotations

import functools
import random
import re

import numpy as np
import pandas as pd

from . import bdg2

VOCABULARY = {
    "building_id": "a metered building, e.g. Fox_office_Alice",
    "building_id_2": "a second metered building of the same type, for comparisons",
    "site_id": "a BDG2 site code, e.g. Fox (write 'the {site_id} site')",
    "city": "a city, e.g. London, Berkeley, Dublin, Tempe",
    "country": "a country: United States, United Kingdom, Ireland or Canada",
    "building_type": "a building-use type, e.g. office, education, lodging/residential",
    "building_type_2": "a second building-use type, for comparisons",
    "year": "2016 or 2017",
    "month": "a month, e.g. July 2016",
    "date": "a day, e.g. 8 May 2016",
    "hour": "an hour of the day, e.g. 14:00",
    "top_n": "a ranking size: 3, 5 or 10",
    "k": "a number of clusters from 2 to 6",
    "threshold_pct": "a percentage threshold: 10, 20, 25 or 50",
    "coverage_pct": "a data-coverage threshold: 80, 90 or 95",
    "sd_multiplier": "a number of standard deviations: 2, 2.5 or 3",
    "horizon_hours": "a forecast horizon in hours: 12, 24 or 48",
    "history_days": "a training-history length in days: 7, 14 or 28",
    "weather_variable": "a weather variable: outdoor air temperature, dew-point temperature or wind speed",
}
PLACE_SLOTS = ("site_id", "city", "country")
WEATHER = {"air_temperature": "outdoor air temperature", "dew_temperature": "dew-point temperature",
           "wind_speed": "wind speed"}
MIN_COHORT = 3


@functools.cache
def _year_share() -> pd.DataFrame:
    """Share of valid hours, rows building_id, columns year."""
    wide = bdg2.electricity_wide()
    return wide.notna().groupby(wide.index.year).mean().T


@functools.cache
def _month_share() -> pd.DataFrame:
    wide = bdg2.electricity_wide()
    return wide.notna().groupby(wide.index.strftime("%Y-%m")).mean().T


@functools.cache
def _day_count() -> pd.DataFrame:
    wide = bdg2.electricity_wide()
    return wide.notna().groupby(wide.index.strftime("%Y-%m-%d")).sum().T


def _qualifying(year: int | None) -> pd.Index:
    share = _year_share()
    if year is None:
        return share.index[share.mean(axis=1) >= 0.8]
    return share.index[share[year] >= 0.9]


def _text(slot: str, value) -> str:
    if slot in ("building_type", "building_type_2"):
        return str(value).lower()
    if slot == "month":
        return pd.Timestamp(f"{value}-01").strftime("%B %Y")
    if slot == "date":
        stamp = pd.Timestamp(value)
        return f"{stamp.day} {stamp.strftime('%B %Y')}"
    if slot == "hour":
        return f"{int(value):02d}:00"
    if slot == "weather_variable":
        return WEATHER[value]
    return str(value)


def sample(slots: list[str], rng: random.Random) -> dict | None:
    """One valid combination for the slots: {slot: {"value": ..., "text": ...}}, or None."""
    meta = bdg2.metadata()
    values: dict = {}
    year = rng.choice(bdg2.YEARS) if {"year", "month", "date", "building_id"} & set(slots) else None
    if "year" in slots:
        values["year"] = year
    pool = meta.loc[meta.index.isin(_qualifying(year))]

    place = next((s for s in PLACE_SLOTS if s in slots), None)
    column = {"site_id": "site_id", "city": "city", "country": "country"}.get(place)
    if "building_type" in slots:
        groups = pool.dropna(subset=[column] if column else []).groupby(
            [column, "building_type"] if column else ["building_type"]).size()
        groups = groups[groups >= MIN_COHORT]
        if groups.empty:
            return None
        key = rng.choice(list(groups.index))
        if column:
            values[place], values["building_type"] = key
        else:
            values["building_type"] = key if isinstance(key, str) else key[0]
        if "building_type_2" in slots:
            others = pool[pool[column] == values[place]] if column else pool
            counts = others["building_type"].value_counts()
            counts = counts[(counts >= MIN_COHORT) & (counts.index != values["building_type"])]
            if counts.empty:
                return None
            values["building_type_2"] = rng.choice(sorted(counts.index))
    elif column:
        counts = pool[column].dropna().value_counts()
        counts = counts[counts >= MIN_COHORT]
        if counts.empty:
            return None
        values[place] = rng.choice(sorted(counts.index))
    for other in PLACE_SLOTS:
        if other in slots and other not in values:
            values[other] = rng.choice(sorted(pool[{"site_id": "site_id", "city": "city",
                                                     "country": "country"}[other]].dropna().unique()))

    if "building_id" in slots:
        candidates = pool
        if column and place in values:
            candidates = candidates[candidates[column] == values[place]]
        if "building_type" in values:
            candidates = candidates[candidates["building_type"] == values["building_type"]]
        if candidates.empty:
            return None
        values["building_id"] = rng.choice(sorted(candidates.index))
        if "building_id_2" in slots:
            same = pool[(pool["building_type"] == meta.at[values["building_id"], "building_type"])
                        & (pool.index != values["building_id"])]
            if same.empty:
                return None
            values["building_id_2"] = rng.choice(sorted(same.index))

    if "month" in slots:
        months = [m for m in _month_share().columns if year is None or m.startswith(str(year))]
        if "building_id" in values:
            share = _month_share().loc[values["building_id"], months]
            months = list(share.index[share >= 0.8])
        if not months:
            return None
        values["month"] = rng.choice(months)
    if "date" in slots:
        days = [d for d in _day_count().columns if d.startswith(values.get("month", str(year) if year else ""))]
        days = days[35:] if len(days) > 60 else days  # leave room for a training history
        if "building_id" in values:
            counts = _day_count().loc[values["building_id"], days]
            days = list(counts.index[counts == 24])
        if not days:
            return None
        values["date"] = rng.choice(days)

    fixed = {"hour": list(range(24)), "top_n": [3, 5, 10], "k": [2, 3, 4, 5, 6], "threshold_pct": [10, 20, 25, 50],
             "coverage_pct": [80, 90, 95], "sd_multiplier": [2, 2.5, 3], "horizon_hours": [12, 24, 48],
             "history_days": [7, 14, 28], "weather_variable": list(WEATHER)}
    for slot, choices in fixed.items():
        if slot in slots:
            values[slot] = rng.choice(choices)

    missing = [slot for slot in slots if slot not in values]
    if missing:
        return None
    return {slot: {"value": _plain(values[slot]), "text": _text(slot, values[slot])} for slot in slots}


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def well_formed(template: str) -> bool:
    """Every brace in the template belongs to a plain slot name from the vocabulary, e.g. {year}."""
    names = re.findall(r"\{(\w+)\}", template)
    rest = re.sub(r"\{\w+\}", "", template)
    return bool(names) and "{" not in rest and "}" not in rest and all(n in VOCABULARY for n in names)


def render(template: str, filled: dict) -> str:
    return template.format(**{slot: item["text"] for slot, item in filled.items()})
