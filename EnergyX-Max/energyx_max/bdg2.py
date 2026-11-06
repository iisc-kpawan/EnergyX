"""The BDG2 data that EnergyX-Max programs compute their answers from.

Each program the Coding Agent writes receives this module as `data`. It reads BDG2's cleaned
electricity file, the weather file and the site-annotated metadata once, and exposes them through a
few functions. Conventions match EnergyX-Gold:

- Timestamps are site-local (BDG2's own timestamps).
- A reading is valid when it isn't NaN. BDG2's cleaning replaced zero readings with NaN.
- Only electricity is in scope: 1,578 buildings with an electricity column.
- Locations come from the site metadata. The five anonymous sites have no city or country.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config

SOURCE = config.path("bdg2_source_dir")
CACHE = config.path("cache")

YEARS = (2016, 2017)
COUNTRY_NAMES = {"US": "United States", "UK": "United Kingdom", "Ireland": "Ireland", "Canada": "Canada"}
# The site metadata puts a US state in its Country column for most US sites.
US_STATES = {"California", "Texas", "New York", "Arizona", "Florida", "New Jersey", "US"}
CITY_NAMES = {"Ottawa, Ontario": "Ottawa"}
WEATHER_COLUMNS = {
    "airTemperature": "air_temperature", "dewTemperature": "dew_temperature", "windSpeed": "wind_speed",
    "windDirection": "wind_direction", "seaLvlPressure": "sea_level_pressure", "cloudCoverage": "cloud_coverage",
    "precipDepth1HR": "precip_depth_1hr", "precipDepth6HR": "precip_depth_6hr",
}


def _cached(name: str, loader) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = loader()
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


@functools.cache
def electricity_wide() -> pd.DataFrame:
    """Hourly kWh, one column per building, indexed by site-local timestamp. NaN is missing."""
    return _cached("electricity_cleaned", lambda: pd.read_csv(
        SOURCE / "electricity_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp"))


@functools.cache
def metadata() -> pd.DataFrame:
    """One row per building with an electricity column, indexed by building_id.

    Columns: site_id, city, country, building_type (BDG2 primaryspaceusage), building_subtype,
    floor_area_m2, year_built, number_of_floors, latitude, longitude, timezone.
    """
    def load():
        raw = pd.read_csv(SOURCE / "BDG2_metadata.csv")
        country = raw["Country"].map(lambda c: "US" if c in US_STATES else c)
        frame = pd.DataFrame({
            "building_id": raw["building_id"],
            "site_id": raw["site_id"],
            "city": raw["City"].replace({"Anonymous": np.nan}).replace(CITY_NAMES),
            "country": country.replace({"Anonymous": np.nan}).map(COUNTRY_NAMES),
            "building_type": raw["primaryspaceusage"],
            "building_subtype": raw["sub_primaryspaceusage"],
            "floor_area_m2": raw["sqm"].where(raw["sqm"] > 0),
            "year_built": raw["yearbuilt"],
            "number_of_floors": raw["numberoffloors"],
            "latitude": raw["lat"],
            "longitude": raw["lng"],
            "timezone": raw["timezone"],
        })
        return frame
    frame = _cached("metadata", load).set_index("building_id")
    return frame[frame.index.isin(electricity_wide().columns)]


@functools.cache
def weather_long() -> pd.DataFrame:
    """Hourly weather per site: timestamp, site_id and the WEATHER_COLUMNS variables."""
    return _cached("weather", lambda: pd.read_csv(SOURCE / "weather.csv", parse_dates=["timestamp"])
                   .rename(columns=WEATHER_COLUMNS))


# -- the functions programs call ---------------------------------------------------------------


def buildings(site_id=None, city=None, country=None, building_type=None) -> list[str]:
    """Sorted building_ids of electricity-metered buildings matching every filter given."""
    meta = metadata()
    mask = pd.Series(True, index=meta.index)
    for column, value in (("site_id", site_id), ("city", city), ("country", country),
                          ("building_type", building_type)):
        if value is not None:
            mask &= meta[column] == value
    return sorted(meta.index[mask])


def readings(building_ids, start=None, end=None) -> pd.DataFrame:
    """Hourly kWh for the buildings, from `start` (inclusive) to `end` (exclusive).

    Wide: one column per building, a complete hourly DatetimeIndex, NaN where missing.
    `start` and `end` are anything pd.Timestamp accepts, e.g. "2017-01-01".
    """
    if isinstance(building_ids, str):
        building_ids = [building_ids]
    frame = electricity_wide().reindex(columns=list(building_ids))  # an unknown id gives an all-NaN column
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index < pd.Timestamp(end)]
    return frame


def weather(site_id: str, start=None, end=None) -> pd.DataFrame:
    """Hourly weather for one site indexed by timestamp, from `start` (inclusive) to `end` (exclusive)."""
    frame = weather_long()
    frame = frame[frame["site_id"] == site_id].set_index("timestamp").drop(columns="site_id").sort_index()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index < pd.Timestamp(end)]
    return frame


def year_range(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of a calendar year, end exclusive."""
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{int(year) + 1}-01-01")


def month_range(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of a month given as "YYYY-MM", end exclusive."""
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthBegin(1)


def valid_share(frame: pd.DataFrame) -> pd.Series:
    """Per column, the share of hours with a valid reading."""
    return frame.notna().mean()


def qualifying(building_ids, year: int, min_share: float = 0.9) -> list[str]:
    """Buildings with a valid reading for at least `min_share` of the year's hours."""
    share = valid_share(readings(building_ids, *year_range(year)))
    return sorted(share.index[share >= min_share])


def eui(building_ids, year: int, min_share: float = 0.9) -> pd.Series:
    """Annual electricity EUI (kWh/m2) of qualifying buildings with a floor area, by building_id."""
    ids = qualifying(building_ids, year, min_share)
    area = metadata().loc[ids, "floor_area_m2"].dropna()
    totals = readings(list(area.index), *year_range(year)).sum()
    return (totals / area).sort_index()
