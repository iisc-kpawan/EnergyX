"""EnergyX Canonical Data Model (ECDM).

Five entities -- Dataset, Site, Building, ElectricityStream, WeatherSource --
plus an extensible attribute layer and the measurement tables.

This module owns three things that everything downstream depends on:

1. **Identifiers.** Minted once, here, and stable across snapshots. The knowledge
   graph projects these directly into IRIs (see docs/KNOWLEDGE_GRAPH.md), so a
   fact asserted in one run still addresses the same entity in the next.
2. **Controlled vocabularies.** Canonical values with the source value always
   preserved alongside, so every mapping is reversible.
3. **Table schemas.** Column order and dtype, enforced on write.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
# IRIs are minted lazily in the graph layer; these functions define the *local*
# id and the IRI path so the two can never drift apart.

IRI_BASE = "http://energyx.org/id/"


def stream_id(building_id: str) -> str:
    """One electricity stream per building in the current scope."""
    return f"{building_id}.elec"


def weather_source_id(site_id: str) -> str:
    return f"{site_id}.weather"


def iri(kind: str, *parts: str) -> str:
    """Stable IRI for an entity. Mirrors docs/KNOWLEDGE_GRAPH.md section 2."""
    return IRI_BASE + "/".join((kind, *parts))


def building_iri(dataset_id: str, building_id: str) -> str:
    return iri("building", dataset_id, building_id)


def site_iri(dataset_id: str, site_id: str) -> str:
    return iri("site", dataset_id, site_id)


def stream_iri(dataset_id: str, stream_id: str) -> str:
    """IRI for one stream.

    Keyed on ``stream_id``, not ``building_id``: a whole-building dataset has
    one stream per building, but an appliance dataset has many, and minting
    from the building collapsed all fourteen SFAC channels onto one node.
    """
    return iri("stream", dataset_id, stream_id)


def weather_iri(dataset_id: str, site_id: str) -> str:
    return iri("weather", dataset_id, site_id)


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

BUILDING_TYPES = (
    # "Residential" arrived with the first residential dataset. BDG2 is
    # non-residential and maps "Lodging/residential" to Lodging, so the
    # vocabulary genuinely lacked the term until a dataset needed it.
    "Residential",
    "Office", "Education", "Assembly", "Lodging", "PublicServices",
    "Healthcare", "Retail", "Warehouse", "Industrial", "Parking",
    "Services", "TechnologyScience", "FoodService", "Utility",
    "ReligiousWorship", "Other",
)

# BDG2 primaryspaceusage -> canonical. Source value is retained as *_raw.
BUILDING_TYPE_MAP = {
    "Residential": "Residential",
    "Education": "Education",
    "Office": "Office",
    "Entertainment/public assembly": "Assembly",
    "Lodging/residential": "Lodging",
    "Public services": "PublicServices",
    "Healthcare": "Healthcare",
    "Parking": "Parking",
    "Warehouse/storage": "Warehouse",
    "Manufacturing/industrial": "Industrial",
    "Retail": "Retail",
    "Services": "Services",
    "Technology/science": "TechnologyScience",
    "Food sales and service": "FoodService",
    "Utility": "Utility",
    "Religious worship": "ReligiousWorship",
    "Other": "Other",
}

HEATING_TYPES = (
    "Gas", "Electricity", "DistrictHeating", "Steam", "Oil", "Biomass", "Other",
)

HEATING_TYPE_MAP = {
    "Gas": "Gas",
    "Gas Boilers": "Gas",
    "Electricity": "Electricity",
    "Electric": "Electricity",
    "Heat network": "DistrictHeating",
    "District Heating": "DistrictHeating",
    "Heat network and steam": "DistrictHeating",
    "Heat network but not ours": "DistrictHeating",
    "Steam": "Steam",
    "Oil": "Oil",
    "Biomass": "Biomass",
    "Boiler fed central heating": "Other",
}

# The BDG2 `Country` column mixes countries, US states and a placeholder;
# these are the state names, mapped to their country on ingestion.
US_STATES = {
    "Arizona", "Florida", "Texas", "New York", "California", "New Jersey",
}
COUNTRY_MAP = {
    "US": ("US", None),
    "UK": ("UK", None),
    "Ireland": ("Ireland", None),
    "Canada": ("Canada", None),
    "Anonymous": (None, None),
    **{s: ("US", s) for s in US_STATES},
}

COUNTRY_CODES = {"US": "USA", "UK": "GBR", "Ireland": "IRL", "Canada": "CAN"}

STREAM_LEVELS = ("whole_building", "submeter", "circuit", "appliance")

# Canonical end uses, kept separate from stream_level so that
# (appliance, refrigeration) and (whole_building, null) coexist cleanly.
END_USES = (
    "hvac", "cooling", "heating", "lighting", "plug_load", "refrigeration",
    "cooking", "laundry", "water_heating", "ev", "other", "unknown",
)

QUALITY_STATUSES = ("excellent", "good", "usable", "poor", "unusable")


def canonicalize_building_type(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return BUILDING_TYPE_MAP.get(str(raw).strip())


def canonicalize_heating_type(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return HEATING_TYPE_MAP.get(str(raw).strip())


def canonicalize_country(raw: Any) -> tuple[str | None, str | None]:
    """Return ``(country_canonical, state_region)``.

    Unmapped values return ``(None, None)`` rather than guessing -- a silent
    wrong country is worse than a null one.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return (None, None)
    return COUNTRY_MAP.get(str(raw).strip(), (None, None))


#: Values in the source's City column that are not cities. BDG2 puts the
#: institution's name there for one site: "Cornell" is Cornell University,
#: whose campus is in Ithaca, New York -- which the site's own coordinates
#: (42.4598, -76.4853) confirm. Left uncorrected it travels into every answer
#: that groups by city, and "Cornell" then appears in a ranking of cities
#: beside Dublin and London as though it were one.
CITY_CORRECTIONS = {"Cornell": ("Ithaca", "New York")}


def split_city(raw: Any) -> tuple[str | None, str | None]:
    """``"Ottawa, Ontario"`` -> ``("Ottawa", "Ontario")``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return (None, None)
    text = str(raw).strip()
    if text == "Anonymous":
        return (None, None)
    if "," in text:
        city, region = text.split(",", 1)
        city, region = city.strip(), region.strip()
    else:
        city, region = text, None
    if city in CITY_CORRECTIONS:
        return CITY_CORRECTIONS[city]
    return (city, region)


# ---------------------------------------------------------------------------
# Table schemas -- column order and dtype, enforced on write
# ---------------------------------------------------------------------------

STR, F64, I64, BOOL, TS = "string", "float64", "Int64", "boolean", "datetime64[ns, UTC]"

SCHEMAS: dict[str, dict[str, str]] = {
    "datasets": {
        "dataset_id": STR, "dataset_name": STR, "dataset_version": STR,
        "description": STR, "source_url": STR, "license": STR,
        "citation": STR, "building_sector": STR,
    },
    "sites": {
        "site_id": STR, "dataset_id": STR, "site_name": STR, "city": STR,
        "state_region": STR, "country_canonical": STR, "country_raw": STR,
        "country_code": STR, "latitude": F64, "longitude": F64,
        "timezone": STR, "utc_offset_hours": F64, "climate_zone": STR,
        "weather_source_id": STR,
        "n_buildings": I64, "iri": STR,
    },
    "buildings": {
        "building_id": STR, "site_id": STR, "dataset_id": STR,
        "building_type_canonical": STR, "building_type_raw": STR,
        "building_subtype": STR, "floor_area_m2": F64, "year_built": I64,
        "number_of_floors": I64, "occupants": I64,
        "heating_type_canonical": STR, "heating_type_raw": STR,
        "industry": STR, "subindustry": STR,
        "latitude": F64, "longitude": F64, "timezone": STR,
        "city": STR, "state_region": STR, "country_canonical": STR, "iri": STR,
    },
    "building_attributes": {
        "building_id": STR, "attribute_name": STR, "attribute_value": STR,
        "attribute_unit": STR, "source": STR,
    },
    "electricity_streams": {
        "stream_id": STR, "building_id": STR, "site_id": STR, "dataset_id": STR,
        "stream_level": STR, "end_use": STR, "unit": STR,
        "resolution_minutes": I64, "start_timestamp": TS, "end_timestamp": TS,
        "n_observations": I64, "expected_observations": I64,
        "coverage_fraction": F64, "source_stream_name": STR,
        "parquet_path": STR, "iri": STR,
    },
    "weather_sources": {
        "weather_source_id": STR, "site_id": STR, "dataset_id": STR,
        "source_name": STR, "resolution_minutes": I64,
        "start_timestamp": TS, "end_timestamp": TS, "n_observations": I64,
        "coverage_fraction": F64, "parquet_path": STR, "iri": STR,
    },
    "data_quality": {
        "stream_id": STR, "coverage_fraction": F64, "missing_fraction": F64,
        "zero_masked_fraction": F64, "duplicate_count": I64,
        "longest_gap_hours": F64, "longest_flat_run_hours": F64,
        "flat_fraction": F64, "first_valid_timestamp": TS,
        "last_valid_timestamp": TS, "quality_status": STR,
        "dst_duplicate_hours": I64, "dst_missing_hours": I64,
    },
    "capabilities": {
        "building_id": STR,
        "can_consumption_analysis": BOOL, "can_eui": BOOL,
        "can_weather_analysis": BOOL, "can_occupancy_normalisation": BOOL,
        "can_peer_benchmark": BOOL, "can_forecasting": BOOL,
        "can_spatial_analysis": BOOL, "has_coordinates": BOOL,
    },
}

WEATHER_VARIABLES = (
    "air_temperature", "dew_temperature", "sea_level_pressure",
    "wind_direction", "wind_speed", "cloud_coverage",
    "precip_depth_1hr", "precip_depth_6hr",
)

# BDG2 weather column -> canonical
WEATHER_COLUMN_MAP = {
    "airTemperature": "air_temperature",
    "dewTemperature": "dew_temperature",
    "seaLvlPressure": "sea_level_pressure",
    "windDirection": "wind_direction",
    "windSpeed": "wind_speed",
    "cloudCoverage": "cloud_coverage",
    "precipDepth1HR": "precip_depth_1hr",
    "precipDepth6HR": "precip_depth_6hr",
}


def conform(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Project a frame onto its declared schema: column order and dtype.

    Missing columns are added as nulls so a partially-populated source never
    silently changes the shape of a catalog table.
    """
    schema = SCHEMAS[table]
    out = pd.DataFrame(index=df.index)
    for col, dtype in schema.items():
        series = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
        if dtype == I64:
            series = pd.to_numeric(series, errors="coerce").round().astype("Int64")
        elif dtype == F64:
            series = pd.to_numeric(series, errors="coerce").astype("float64")
        elif dtype == BOOL:
            series = series.astype("boolean")
        elif dtype == STR:
            series = series.astype("object").where(series.notna(), None).astype("string")
        elif dtype == TS:
            series = pd.to_datetime(series, utc=True, errors="coerce")
        out[col] = series
    return out.reset_index(drop=True)
