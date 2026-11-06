"""Export the canonical model to Brick, Haystack and BEDES.

This is the third level of the hierarchy the design left open:
**source schema -> canonical model -> standards mappings**. It is what
makes the interoperability claim real rather than aspirational, and it is an
*export*, never an ingestion requirement: no dataset has to conform to anything
to enter EnergyX.

Only terms that genuinely correspond are mapped. Where a standard has no
equivalent the field is reported as unmapped rather than forced into the nearest
term, because a wrong mapping is worse than a missing one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BRICK = "https://brickschema.org/schema/Brick#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# --- term mappings ----------------------------------------------------------

BRICK_CLASSES = {
    "Building": "Building", "Site": "Site",
    "ElectricityStream/whole_building": "Building_Electrical_Meter",
    "ElectricityStream/submeter": "Electrical_Meter",
    "ElectricityStream/circuit": "Electrical_Meter",
    "ElectricityStream/appliance": "Electrical_Meter",
}
BRICK_PROPERTIES = {
    "floorAreaM2": "area", "atSite": "isPartOf", "hasStream": "isPointOf",
}
BRICK_UNMAPPED = ("yearBuilt", "occupants", "timezone", "coverageFraction",
                  "buildingTypeName")

HAYSTACK_TAGS = {
    "Building": ["site"],
    "ElectricityStream/whole_building": ["equip", "meter", "elec", "siteMeter"],
    "ElectricityStream/submeter": ["equip", "meter", "elec", "submeterOf"],
    "ElectricityStream/circuit": ["equip", "meter", "elec", "submeterOf"],
    "ElectricityStream/appliance": ["equip", "meter", "elec", "submeterOf"],
}
HAYSTACK_PROPERTIES = {
    "floorAreaM2": "area", "timezone": "tz", "yearBuilt": "yearBuilt",
    "cityName": "geoCity", "countryName": "geoCountry",
    "latitude": "geoCoord.lat", "longitude": "geoCoord.lng",
    "servedByWeather": "weatherStationRef", "atSite": "siteRef",
}
HAYSTACK_UNMAPPED = ("occupants", "coverageFraction", "qualityStatus")

BEDES_TERMS = {
    "Building": "Premises",
    "buildingTypeName": "Premises Type",
    "floorAreaM2": "Gross Floor Area",
    "yearBuilt": "Year Of Construction",
    "occupants": "Occupant Quantity",
    "cityName": "City",
    "countryName": "Country",
    "ElectricityStream": "Energy Resource: Electricity",
    "unit": "Energy Units",
    "resolutionMinutes": "Interval Frequency",
}
BEDES_UNMAPPED = ("coverageFraction", "qualityStatus", "utcOffsetHours")


@dataclass
class Export:
    target: str
    entities: int = 0
    triples: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    mapped_terms: dict = field(default_factory=dict)
    unmapped_terms: list[str] = field(default_factory=list)

    def turtle(self) -> str:
        prefixes = {
            "brick": "@prefix brick: <https://brickschema.org/schema/Brick#> .",
            "haystack": "@prefix ph: <https://project-haystack.org/def/ph/3.9.14#> .",
        }.get(self.target, "")
        header = "\n".join(filter(None, [
            "@prefix ekgd: <http://energyx.org/id/> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            prefixes, ""]))
        return header + "\n".join(self.triples)

    def to_dict(self) -> dict:
        return {"target": self.target, "entities": self.entities,
                "triples": len(self.triples),
                "mapped_terms": self.mapped_terms,
                "unmapped_terms": self.unmapped_terms}


def _short(iri: str) -> str:
    return "ekgd:" + iri.split("/id/")[-1] if "/id/" in iri else f"<{iri}>"


def export(rows: list[dict], target: str = "brick") -> Export:
    """Project building rows into a standard's terms.

    ``rows`` come from the graph: building IRI plus the canonical fields.
    """
    if target == "brick":
        return _brick(rows)
    if target == "haystack":
        return _haystack(rows)
    if target == "bedes":
        return _bedes(rows)
    raise ValueError(f"unknown target {target!r}; expected brick, haystack or bedes")


def _brick(rows: list[dict]) -> Export:
    out = Export(target="brick", mapped_terms={**BRICK_CLASSES, **BRICK_PROPERTIES},
                 unmapped_terms=list(BRICK_UNMAPPED))
    for row in rows:
        subject = _short(row["building"])
        out.triples.append(f"{subject} rdf:type brick:{BRICK_CLASSES['Building']} .")
        if row.get("site"):
            site = _short(row["site"])
            out.triples.append(f"{subject} brick:{BRICK_PROPERTIES['atSite']} {site} .")
            out.triples.append(f"{site} rdf:type brick:{BRICK_CLASSES['Site']} .")
        if row.get("area") is not None:
            out.triples.append(
                f'{subject} brick:{BRICK_PROPERTIES["floorAreaM2"]} '
                f'[ brick:value "{row["area"]}" ; brick:hasUnit "M2" ] .')
        if row.get("stream"):
            level = row.get("stream_level", "whole_building")
            klass = BRICK_CLASSES.get(f"ElectricityStream/{level}",
                                      "Electrical_Meter")
            meter = _short(row["stream"])
            out.triples.append(f"{meter} rdf:type brick:{klass} .")
            out.triples.append(
                f"{meter} brick:{BRICK_PROPERTIES['hasStream']} {subject} .")
        out.entities += 1
    return out


def _haystack(rows: list[dict]) -> Export:
    out = Export(target="haystack", mapped_terms=HAYSTACK_PROPERTIES,
                 unmapped_terms=list(HAYSTACK_UNMAPPED))
    for row in rows:
        record = {"id": row["building"].split("/")[-1],
                  "dis": row["building"].split("/")[-1]}
        for tag in HAYSTACK_TAGS["Building"]:
            record[tag] = "M"          # Haystack marker tag
        if row.get("area") is not None:
            record[HAYSTACK_PROPERTIES["floorAreaM2"]] = f'{row["area"]}m²'
        for source, tag in (("timezone", "timezone"), ("city", "cityName"),
                            ("country", "countryName")):
            if row.get(source):
                record[HAYSTACK_PROPERTIES[tag]] = row[source]
        if row.get("year") is not None:
            record[HAYSTACK_PROPERTIES["yearBuilt"]] = int(row["year"])
        if row.get("stream"):
            level = row.get("stream_level", "whole_building")
            record["meterTags"] = " ".join(
                HAYSTACK_TAGS.get(f"ElectricityStream/{level}", ["meter", "elec"]))
            record["meterRef"] = row["stream"].split("/id/")[-1]
        out.records.append(record)
        out.entities += 1
    return out


def _bedes(rows: list[dict]) -> Export:
    out = Export(target="bedes", mapped_terms=BEDES_TERMS,
                 unmapped_terms=list(BEDES_UNMAPPED))
    for row in rows:
        record = {"BEDES Term: Premises Identifier": row["building"].split("/")[-1]}
        if row.get("type"):
            record[BEDES_TERMS["buildingTypeName"]] = row["type"]
        if row.get("area") is not None:
            record[BEDES_TERMS["floorAreaM2"]] = row["area"]
            record["Gross Floor Area Units"] = "Square Meters"
        if row.get("year") is not None:
            record[BEDES_TERMS["yearBuilt"]] = int(row["year"])
        if row.get("occupants") is not None:
            record[BEDES_TERMS["occupants"]] = int(row["occupants"])
        if row.get("city"):
            record[BEDES_TERMS["cityName"]] = row["city"]
        if row.get("country"):
            record[BEDES_TERMS["countryName"]] = row["country"]
        record[BEDES_TERMS["ElectricityStream"]] = "Electricity"
        record[BEDES_TERMS["unit"]] = "kWh"
        out.records.append(record)
        out.entities += 1
    return out
