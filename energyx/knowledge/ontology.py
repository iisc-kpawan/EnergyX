"""The EnergyX Knowledge Graph (EKG) ontology.

Terms only -- no data. The vocabulary mirrors the canonical model: the
graph is a *projection* of the ECDM, not a second schema, so every class here
corresponds to something the catalog already holds, plus the derived classes the
system learns.

IRIs are minted from ECDM primary keys through ``energyx.ecdm``, so an entity
has one identity across the catalog, the graph and any asserted fact.
"""
from __future__ import annotations

EKG = "http://energyx.org/ns#"
EKGD = "http://energyx.org/id/"
EKGP = "http://energyx.org/prov#"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

PREFIXES = {"ekg": EKG, "ekgd": EKGD, "ekgp": EKGP, "xsd": XSD,
            "rdf": RDF, "rdfs": RDFS}

SPARQL_PREFIXES = "\n".join(f"PREFIX {p}: <{u}>" for p, u in PREFIXES.items())

# --- classes ---------------------------------------------------------------

ASSERTED_CLASSES = (
    "Dataset", "Site", "Building", "ElectricityStream", "WeatherSource",
    "Country", "Region", "City", "BuildingType", "Industry", "HeatingType",
)

DERIVED_CLASSES = (
    "PeerGroup", "LoadShapeCluster", "Representation",
    "Finding", "ChangePoint", "AnomalyEvent", "TrendSegment",
    "OperatingSchedule", "PerformanceScore",
    # A weighted or provenanced relation, reified as a node.
    #
    # The design specified RDF-star for per-fact metadata, but pyoxigraph's
    # Python bindings do not accept a quoted triple as a subject, so weighted
    # relations are reified instead. Both forms are written: the plain
    # edge stays traversable, and the assertion node carries the weight and
    # provenance.
    "RelationAssertion",
)

PROVENANCE_CLASSES = ("AnalysisRun", "ToolInvocation", "Method")

CLASSES = ASSERTED_CLASSES + DERIVED_CLASSES + PROVENANCE_CLASSES

# Finding subtypes, for validation and for tools that assert them.
FINDING_TYPES = ("ChangePoint", "AnomalyEvent", "TrendSegment",
                 "OperatingSchedule", "PerformanceScore")

# --- properties ------------------------------------------------------------

OBJECT_PROPERTIES = {
    "inDataset": ("Building|Site|ElectricityStream|WeatherSource", "Dataset"),
    "atSite": ("Building", "Site"),
    "inCity": ("Site", "City"),
    "inRegion": ("City", "Region"),
    "inCountry": ("Region|City", "Country"),
    "hasStream": ("Building", "ElectricityStream"),
    "servedByWeather": ("Site", "WeatherSource"),
    "hasBuildingType": ("Building", "BuildingType"),
    "hasIndustry": ("Building", "Industry"),
    "hasHeatingType": ("Building", "HeatingType"),
    # derived
    "memberOf": ("Building", "PeerGroup"),
    "inCluster": ("Building", "LoadShapeCluster"),
    "similarLoadShape": ("Building", "Building"),
    "percentileIn": ("Building", "PeerGroup"),
    "hasFinding": ("Building", "Finding"),
    "hasRepresentation": ("Building", "Representation"),
    "aboutBuilding": ("Finding", "Building"),
    # reified relations
    "relSubject": ("RelationAssertion", "Building"),
    "relObject": ("RelationAssertion", "Building"),
    "relPredicate": ("RelationAssertion", "Building"),
    "assertsRelation": ("Building", "RelationAssertion"),
}

DATATYPE_PROPERTIES = {
    "siteName": "string", "cityName": "string", "countryName": "string",
    "regionName": "string", "buildingTypeName": "string",
    "datasetName": "string", "datasetVersion": "string",
    "floorAreaM2": "double", "yearBuilt": "gYear", "occupants": "integer",
    "numberOfFloors": "integer", "latitude": "double", "longitude": "double",
    "timezone": "string", "utcOffsetHours": "double",
    "streamLevel": "string", "endUse": "string", "unit": "string",
    "resolutionMinutes": "integer", "coverageFraction": "double",
    "missingFraction": "double", "zeroMaskedFraction": "double",
    "longestGapHours": "double", "qualityStatus": "string",
    "startTime": "dateTime", "endTime": "dateTime",
    "nObservations": "integer",
    # findings
    "findingValue": "double", "findingUnit": "string",
    "validFrom": "dateTime", "validTo": "dateTime",
    "similarity": "double", "percentileValue": "double",
    "weight": "double", "clusterLabel": "integer",
    "peerGroupCriteria": "string", "representationKind": "string",
    "vectorRef": "string", "dimension": "integer",
}

CAPABILITY_PROPERTIES = {
    "canConsumptionAnalysis": "can_consumption_analysis",
    "canComputeEUI": "can_eui",
    "canWeatherAnalysis": "can_weather_analysis",
    "canOccupancyNormalise": "can_occupancy_normalisation",
    "canPeerBenchmark": "can_peer_benchmark",
    "canForecast": "can_forecasting",
    "canSpatialAnalysis": "can_spatial_analysis",
    "hasCoordinates": "has_coordinates",
}

PROVENANCE_PROPERTIES = {
    "assertionType": "string",      # asserted | derived -- see 
    "snapshotVersion": "string",
    "method": "string", "params": "string",
    "confidence": "double",
    "computedAt": "dateTime", "computedByRun": "string",
    # Facts are superseded, never overwritten: an earlier belief stays
    # inspectable, which is what makes assertion time meaningful.
    "supersededBy": "string", "supersedes": "string",
    "factKey": "string",            # identity for dedupe: what this fact is *about*
}

ALL_PROPERTIES = (
    set(OBJECT_PROPERTIES)
    | set(DATATYPE_PROPERTIES)
    | set(CAPABILITY_PROPERTIES)
)
ALL_PROV_PROPERTIES = set(PROVENANCE_PROPERTIES)

ASSERTION_TYPES = ("asserted", "derived")


def term(name: str) -> str:
    return f"{EKG}{name}"


def prov_term(name: str) -> str:
    return f"{EKGP}{name}"


def known_term(iri: str) -> bool:
    """Is this IRI part of the declared vocabulary?

    Used to reject a generated SPARQL query before it runs: a query against a
    property that does not exist returns zero rows, which is indistinguishable
    from a correct query with no matches.
    """
    if iri.startswith(EKG):
        local = iri[len(EKG):]
        return local in CLASSES or local in ALL_PROPERTIES
    if iri.startswith(EKGP):
        return iri[len(EKGP):] in ALL_PROV_PROPERTIES
    return iri.startswith((EKGD, RDF, RDFS, XSD))


def ontology_triples() -> list[tuple[str, str, object]]:
    """The vocabulary itself, as triples, so the store is self-describing."""
    triples: list[tuple[str, str, str]] = []
    for name in CLASSES:
        triples.append((term(name), f"{RDF}type", f"{RDFS}Class"))
        triples.append((term(name), f"{RDFS}label", (name, "string")))
    for name in FINDING_TYPES:
        triples.append((term(name), f"{RDFS}subClassOf", term("Finding")))
    for name, (domain, target) in OBJECT_PROPERTIES.items():
        triples.append((term(name), f"{RDF}type", f"{RDF}Property"))
        triples.append((term(name), f"{RDFS}range", term(target.split("|")[0])))
    for name in DATATYPE_PROPERTIES:
        triples.append((term(name), f"{RDF}type", f"{RDF}Property"))
    for name in CAPABILITY_PROPERTIES:
        triples.append((term(name), f"{RDF}type", f"{RDF}Property"))
    return triples
