"""Project the catalog into RDF triples.

**Adds no new information.** Every triple restates a catalog row, which is why
this phase is cheap and why the catalog stays authoritative: if the two ever
disagree, the projector is broken, not the catalog.

Derived facts are a separate concern and are never emitted here -
everything this module produces carries ``assertionType = "asserted"``.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from .. import ecdm
from . import ontology as o

Triple = tuple[str, str, object]          # (subject IRI, predicate IRI, value)
ASSERTED = "asserted"


def _iri(kind: str, *parts: str) -> str:
    return ecdm.iri(kind, *(str(p).replace(" ", "_") for p in parts))


def _lit(value, datatype: str) -> tuple[object, str]:
    return (value, datatype)


def _present(value) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value)) \
        and not (hasattr(value, "__len__") and not isinstance(value, str)
                 and len(value) == 0) and value is not pd.NA


class CatalogProjector:
    """Reads the catalog tables and yields triples."""

    def __init__(self, db, dataset_id: str = "bdg2"):
        self.db = db
        self.dataset_id = dataset_id
        self.counts: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------
    def _table(self, name: str) -> pd.DataFrame:
        return self.db.df(f"SELECT * FROM {name}")

    @staticmethod
    def _type(subject: str, klass: str) -> Triple:
        return (subject, f"{o.RDF}type", o.term(klass))

    # -- projections -------------------------------------------------------
    def datasets(self) -> Iterator[Triple]:
        for row in self._table("datasets").to_dict("records"):
            node = _iri("dataset", row["dataset_id"])
            yield self._type(node, "Dataset")
            for column, prop, kind in (("dataset_name", "datasetName", "string"),
                                       ("dataset_version", "datasetVersion", "string")):
                if _present(row.get(column)):
                    yield (node, o.term(prop), _lit(row[column], kind))

    def geography(self, sites: pd.DataFrame) -> Iterator[Triple]:
        """Country / region / city nodes, deduplicated across sites."""
        seen: set[str] = set()
        for row in sites.to_dict("records"):
            country = row.get("country_canonical")
            if not _present(country):
                continue
            country_node = _iri("country", country)
            if country_node not in seen:
                seen.add(country_node)
                yield self._type(country_node, "Country")
                yield (country_node, o.term("countryName"), _lit(country, "string"))
            region = row.get("state_region")
            region_node = None
            if _present(region):
                region_node = _iri("region", country, region)
                if region_node not in seen:
                    seen.add(region_node)
                    yield self._type(region_node, "Region")
                    yield (region_node, o.term("regionName"), _lit(region, "string"))
                    yield (region_node, o.term("inCountry"), country_node)
            city = row.get("city")
            if _present(city):
                city_node = _iri("city", country, city)
                if city_node not in seen:
                    seen.add(city_node)
                    yield self._type(city_node, "City")
                    yield (city_node, o.term("cityName"), _lit(city, "string"))
                    yield (city_node, o.term("inCountry"), country_node)
                    if region_node:
                        yield (city_node, o.term("inRegion"), region_node)

    def sites(self, sites: pd.DataFrame) -> Iterator[Triple]:
        for row in sites.to_dict("records"):
            node = ecdm.site_iri(row["dataset_id"], row["site_id"])
            yield self._type(node, "Site")
            yield (node, o.term("inDataset"), _iri("dataset", row["dataset_id"]))
            for column, prop, kind in (
                    ("site_name", "siteName", "string"),
                    ("timezone", "timezone", "string"),
                    ("utc_offset_hours", "utcOffsetHours", "double"),
                    ("latitude", "latitude", "double"),
                    ("longitude", "longitude", "double")):
                if _present(row.get(column)):
                    yield (node, o.term(prop), _lit(row[column], kind))
            if _present(row.get("city")) and _present(row.get("country_canonical")):
                yield (node, o.term("inCity"),
                       _iri("city", row["country_canonical"], row["city"]))
            if _present(row.get("weather_source_id")):
                yield (node, o.term("servedByWeather"),
                       ecdm.weather_iri(row["dataset_id"], row["site_id"]))

    def classifications(self, buildings: pd.DataFrame) -> Iterator[Triple]:
        for column, klass, kind, prop in (
                ("building_type_canonical", "BuildingType", "buildingtype", "buildingTypeName"),
                ("industry", "Industry", "industry", "buildingTypeName"),
                ("heating_type_canonical", "HeatingType", "heatingtype", "buildingTypeName")):
            for value in sorted(buildings[column].dropna().unique()):
                node = _iri(kind, value)
                yield self._type(node, klass)
                yield (node, o.term(prop), _lit(str(value), "string"))

    def buildings(self, buildings: pd.DataFrame) -> Iterator[Triple]:
        for row in buildings.to_dict("records"):
            node = ecdm.building_iri(row["dataset_id"], row["building_id"])
            yield self._type(node, "Building")
            yield (node, o.term("inDataset"), _iri("dataset", row["dataset_id"]))
            yield (node, o.term("atSite"),
                   ecdm.site_iri(row["dataset_id"], row["site_id"]))
            if _present(row.get("building_type_canonical")):
                yield (node, o.term("hasBuildingType"),
                       _iri("buildingtype", row["building_type_canonical"]))
            if _present(row.get("industry")):
                yield (node, o.term("hasIndustry"), _iri("industry", row["industry"]))
            if _present(row.get("heating_type_canonical")):
                yield (node, o.term("hasHeatingType"),
                       _iri("heatingtype", row["heating_type_canonical"]))
            for column, prop, kind in (
                    ("floor_area_m2", "floorAreaM2", "double"),
                    ("year_built", "yearBuilt", "integer"),
                    ("occupants", "occupants", "integer"),
                    ("number_of_floors", "numberOfFloors", "integer"),
                    ("latitude", "latitude", "double"),
                    ("longitude", "longitude", "double"),
                    ("timezone", "timezone", "string")):
                if _present(row.get(column)):
                    yield (node, o.term(prop), _lit(row[column], kind))

    def streams(self, streams: pd.DataFrame, quality: pd.DataFrame) -> Iterator[Triple]:
        quality_index = quality.set_index("stream_id").to_dict("index")
        for row in streams.to_dict("records"):
            node = ecdm.stream_iri(row["dataset_id"], row["stream_id"])
            building = ecdm.building_iri(row["dataset_id"], row["building_id"])
            yield self._type(node, "ElectricityStream")
            yield (building, o.term("hasStream"), node)
            yield (node, o.term("inDataset"), _iri("dataset", row["dataset_id"]))
            for column, prop, kind in (
                    ("stream_level", "streamLevel", "string"),
                    ("end_use", "endUse", "string"),
                    ("unit", "unit", "string"),
                    ("resolution_minutes", "resolutionMinutes", "integer"),
                    ("coverage_fraction", "coverageFraction", "double"),
                    ("n_observations", "nObservations", "integer"),
                    ("start_timestamp", "startTime", "dateTime"),
                    ("end_timestamp", "endTime", "dateTime")):
                if _present(row.get(column)):
                    yield (node, o.term(prop), _lit(row[column], kind))
            q = quality_index.get(row["stream_id"])
            if q:
                for column, prop, kind in (
                        ("missing_fraction", "missingFraction", "double"),
                        ("zero_masked_fraction", "zeroMaskedFraction", "double"),
                        ("longest_gap_hours", "longestGapHours", "double"),
                        ("quality_status", "qualityStatus", "string")):
                    if _present(q.get(column)):
                        yield (node, o.term(prop), _lit(q[column], kind))

    def weather(self, weather: pd.DataFrame) -> Iterator[Triple]:
        for row in weather.to_dict("records"):
            node = ecdm.weather_iri(row["dataset_id"], row["site_id"])
            yield self._type(node, "WeatherSource")
            yield (node, o.term("inDataset"), _iri("dataset", row["dataset_id"]))
            for column, prop, kind in (
                    ("coverage_fraction", "coverageFraction", "double"),
                    ("resolution_minutes", "resolutionMinutes", "integer"),
                    ("n_observations", "nObservations", "integer")):
                if _present(row.get(column)):
                    yield (node, o.term(prop), _lit(row[column], kind))

    def capabilities(self, capabilities: pd.DataFrame,
                     buildings: pd.DataFrame) -> Iterator[Triple]:
        dataset_of = dict(zip(buildings["building_id"], buildings["dataset_id"]))
        for row in capabilities.to_dict("records"):
            dataset = dataset_of.get(row["building_id"], self.dataset_id)
            node = ecdm.building_iri(dataset, row["building_id"])
            for prop, column in o.CAPABILITY_PROPERTIES.items():
                value = row.get(column)
                if _present(value):
                    yield (node, o.term(prop), _lit(bool(value), "boolean"))

    # -- everything --------------------------------------------------------
    def project(self) -> Iterator[Triple]:
        sites = self._table("sites")
        buildings = self._table("buildings")
        streams = self._table("electricity_streams")
        quality = self._table("data_quality")
        weather = self._table("weather_sources")
        capabilities = self._table("capabilities")

        sections = {
            "ontology": (t for t in o.ontology_triples()),
            "datasets": self.datasets(),
            "geography": self.geography(sites),
            "sites": self.sites(sites),
            "classifications": self.classifications(buildings),
            "buildings": self.buildings(buildings),
            "streams": self.streams(streams, quality),
            "weather": self.weather(weather),
            "capabilities": self.capabilities(capabilities, buildings),
        }
        for name, generator in sections.items():
            count = 0
            for triple in generator:
                count += 1
                yield triple
            self.counts[name] = count
