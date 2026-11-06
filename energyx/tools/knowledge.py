"""Knowledge & Semantic Reasoning tools (6).

The graph answers "what exists and how does it relate" without scanning a single
measurement. These tools are the eighth specialist's surface.

Two disciplines are enforced here rather than trusted:

* **Generated SPARQL is validated against the declared vocabulary before it
  runs**. An unknown property returns zero rows, which is
  indistinguishable from a correct query with no matches.
* **Derived facts are never asserted unless write-back is explicitly enabled**,
  so a measured run executes against a frozen, read-only graph.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ..knowledge import ontology as o
from ..knowledge.validation import guard, validate_query
from .base import ToolResult, empty, frame_result, insufficient, ok
from .registry import ToolContext, tool

RELATIONS = {
    "atSite": "Building -> Site", "inCity": "Site -> City",
    "inRegion": "City -> Region", "inCountry": "Region|City -> Country",
    "hasStream": "Building -> ElectricityStream",
    "servedByWeather": "Site -> WeatherSource",
    "hasBuildingType": "Building -> BuildingType",
    "hasIndustry": "Building -> Industry",
    "hasHeatingType": "Building -> HeatingType",
}


def _resolve(ctx: ToolContext, name: str) -> str | None:
    """Name or id -> IRI. Buildings, sites, cities and countries."""
    if name.startswith("http"):
        return name
    rows = ctx.graph.query(f"""
        SELECT ?s WHERE {{
          {{ ?s a ekg:Building . FILTER(STRENDS(STR(?s), "/{name}")) }}
          UNION {{ ?s a ekg:Site . FILTER(STRENDS(STR(?s), "/{name}")) }}
          UNION {{ ?s a ekg:City ; ekg:cityName "{name}" }}
          UNION {{ ?s a ekg:Country ; ekg:countryName "{name}" }}
          UNION {{ ?s a ekg:BuildingType ; ekg:buildingTypeName "{name}" }}
        }} LIMIT 1""")
    return rows[0]["s"] if rows else None


@tool("knowledge")
def query_knowledge_graph(ctx: ToolContext, sparql: str,
                          limit: int = 200) -> ToolResult:
    """Run a validated SPARQL SELECT/ASK against the knowledge graph.

    Prefixes ekg:, ekgd:, ekgp:, rdf:, rdfs: and xsd: are supplied; the query
    is rejected before execution if it references a term the ontology does not
    declare.
    """
    check = validate_query(sparql)
    if not check.ok:
        return insufficient("invalid_sparql", check.message(),
                            evidence={"unknown_terms": check.unknown_terms,
                                      "form": check.form})
    try:
        rows = ctx.graph.query(guard(sparql, limit))
    except Exception as exc:  # noqa: BLE001 - surface the engine's own message
        return insufficient("sparql_error", f"{type(exc).__name__}: {exc}")
    if not rows:
        return empty("the query is valid but matched nothing",
                     evidence={"form": check.form})
    return frame_result(pd.DataFrame(rows), ctx.store, name="sparql",
                        evidence={"form": check.form, "rows": len(rows),
                                  "engine": "oxigraph"})


@tool("knowledge")
def traverse_relations(
    ctx: ToolContext,
    entity: str,
    relation: Literal["atSite", "inCity", "inRegion", "inCountry", "hasStream",
                      "servedByWeather", "hasBuildingType", "hasIndustry",
                      "hasHeatingType"] = "atSite",
    direction: Literal["out", "in"] = "out",
    depth: int = 1,
    limit: int = 100,
) -> ToolResult:
    """Follow one relation from an entity, up to a bounded depth.

    Depth is bounded deliberately: an unbounded traversal over a graph this
    shape returns most of it.
    """
    iri = _resolve(ctx, entity)
    if iri is None:
        return insufficient("unknown_entity",
                            f"{entity!r} is not a building, site, city, country "
                            "or building type in the graph")
    depth = max(1, min(int(depth), 4))
    # SPARQL 1.1 dropped the {n,m} path quantifier, so bounded repetition is
    # spelled out as alternated chains rather than assumed.
    chains = [" / ".join([f"ekg:{relation}"] * hop) for hop in range(1, depth + 1)]
    path = " | ".join(f"({c})" for c in chains)
    pattern = (f"<{iri}> {path} ?other" if direction == "out"
               else f"?other {path} <{iri}>")
    rows = ctx.graph.query(f"SELECT DISTINCT ?other WHERE {{ {pattern} }} "
                           f"LIMIT {int(limit)}")
    if not rows:
        return empty(f"no {relation} {direction}-edges from {entity}",
                     evidence={"entity": iri, "relation": relation})
    return frame_result(pd.DataFrame(rows), ctx.store, name="traversal",
                        evidence={"entity": iri, "relation": relation,
                                  "direction": direction, "depth": depth,
                                  "meaning": RELATIONS.get(relation, "")})


@tool("knowledge")
def lookup_derived_facts(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    finding_type: Literal["ChangePoint", "AnomalyEvent", "TrendSegment",
                          "OperatingSchedule", "PerformanceScore",
                          "any"] = "any",
    as_evidence: bool = False,
    at: str | None = None,
) -> ToolResult:
    """Read previously computed findings instead of recomputing them.

    Only facts pinned to the active snapshot are served: a finding computed
    against a superseded snapshot is stale by definition. Superseded
    beliefs are never served.

    ``as_evidence`` pivots the result into one row per building with a column
    per method, which is the shape ``combine_evidence`` and
    ``prioritize_entities`` consume -- so a decision-support question can read
    what four specialists previously computed instead of re-running them.

    ``at`` queries as of a point in **valid** time, distinct from when the fact
    was asserted.
    """
    from ..knowledge.curator import KnowledgeCurator

    curator = KnowledgeCurator(ctx.graph, ctx.snapshot_version)
    rows = curator.current_facts(
        kind=None if finding_type == "any" else finding_type,
        building_ids=building_ids, at=at)
    evidence = {"finding_type": finding_type,
                "snapshot": ctx.snapshot_version,
                "requested_buildings": len(building_ids or []),
                "as_of_valid_time": at,
                "note": "derived facts are written by the Knowledge Curator; "
                        "an empty result means nothing has been computed yet"}
    if not rows:
        return empty("no derived facts recorded for this snapshot",
                     evidence=evidence)

    frame = pd.DataFrame(rows)
    if as_evidence:
        # One row per building, one column per method -- the shape the
        # decision-support tools consume.
        from ..knowledge.probe import method_column
        frame["column"] = frame["method"].map(method_column)
        frame = (frame.pivot_table(index="building_id", columns="column",
                                   values="value", aggfunc="first")
                      .reset_index())
        frame.columns.name = None
        evidence["methods"] = [c for c in frame.columns if c != "building_id"]
        evidence["shape"] = "evidence table"
    return frame_result(frame, ctx.store, name="derived_facts",
                        evidence=evidence)


@tool("knowledge")
def assert_findings(
    ctx: ToolContext,
    findings: list[dict],
    method: str,
    confidence: float = 1.0,
    params: str = "",
) -> ToolResult:
    """Assert derived findings into the graph through the Knowledge Curator.

    Every fact is pinned with snapshot, method, params, confidence and
    ``assertionType = "derived"`` -- without that last field a wrong
    finding becomes indistinguishable from ground truth.

    Refuses unless write-back is explicitly enabled, because a run that writes
    facts a later run reads is no longer independent of it.
    """
    if not ctx.write_back_enabled:
        return insufficient(
            "write_back_disabled",
            "write-back is off for this run, which executes against a frozen, "
            "read-only graph. Enable it with options={'write_back': True}.",
            evidence={"findings_offered": len(findings)})
    if not findings:
        return empty("nothing to assert")

    import uuid
    from datetime import datetime
    quads: list[tuple] = []
    accepted, rejected = 0, []
    for item in findings:
        kind = item.get("type")
        building = item.get("building_id")
        if kind not in o.FINDING_TYPES:
            rejected.append(f"{kind!r} is not a declared finding type")
            continue
        if not building:
            rejected.append("finding without a building_id")
            continue
        node = f"{o.EKGD}finding/{uuid.uuid4().hex[:12]}"
        target = f"{o.EKGD}building/bdg2/{building}"
        quads += [
            (node, f"{o.RDF}type", o.term(kind)),
            (node, f"{o.RDF}type", o.term("Finding")),
            (node, o.term("aboutBuilding"), target),
            (target, o.term("hasFinding"), node),
            (node, o.prov_term("assertionType"), ("derived", "string")),
            (node, o.prov_term("snapshotVersion"), (ctx.snapshot_version, "string")),
            (node, o.prov_term("method"), (method, "string")),
            (node, o.prov_term("confidence"), (float(confidence), "double")),
            (node, o.prov_term("computedAt"),
             (datetime.utcnow().isoformat() + "Z", "dateTime")),
            (node, o.prov_term("computedByRun"), (ctx.store.run_id, "string")),
        ]
        if params:
            quads.append((node, o.prov_term("params"), (params, "string")))
        if item.get("value") is not None:
            quads.append((node, o.term("findingValue"), (float(item["value"]), "double")))
        if item.get("unit"):
            quads.append((node, o.term("findingUnit"), (str(item["unit"]), "string")))
        for key, prop in (("valid_from", "validFrom"), ("valid_to", "validTo")):
            if item.get(key):
                quads.append((node, o.term(prop), (item[key], "dateTime")))
        accepted += 1

    written = ctx.graph.add_facts(quads) if quads else 0
    return ok({"facts_asserted": accepted, "triples_written": written,
               "rejected": rejected},
              evidence={"method": method, "confidence": confidence,
                        "snapshot": ctx.snapshot_version,
                        "assertion_type": "derived", "run": ctx.store.run_id})


@tool("knowledge")
def find_entity_paths(ctx: ToolContext, source: str, target: str,
                      max_depth: int = 3) -> ToolResult:
    """How are two entities related? Returns the connecting path, if any."""
    a, b = _resolve(ctx, source), _resolve(ctx, target)
    missing = [n for n, iri in ((source, a), (target, b)) if iri is None]
    if missing:
        return insufficient("unknown_entity", f"not in the graph: {missing}")
    depth = max(1, min(int(max_depth), 4))
    predicates = " | ".join(f"ekg:{r}" for r in RELATIONS)
    step = f"({predicates})"
    for hop in range(1, depth + 1):
        chain = " / ".join([step] * hop)
        if ctx.graph.ask(f"ASK {{ <{a}> {chain} <{b}> }}"):
            rows = ctx.graph.query(
                f"SELECT DISTINCT ?mid WHERE {{ <{a}> {step}+ ?mid . "
                f"?mid {step}* <{b}> }} LIMIT 20")
            return ok({"connected": True, "hops": hop,
                       "source": a, "target": b,
                       "intermediates": [r["mid"] for r in rows][:10]},
                      evidence={"max_depth": depth,
                                "relations_followed": sorted(RELATIONS)})
    return empty(f"no path of {depth} hops or fewer between {source} and {target}",
                 evidence={"source": a, "target": b, "max_depth": depth})


@tool("knowledge")
def export_semantic_model(
    ctx: ToolContext,
    target: Literal["brick", "haystack", "bedes"] = "brick",
    building_ids: list[str] | None = None,
    limit: int = 25,
) -> ToolResult:
    """Project the graph outward to Brick, Haystack or BEDES terms.

    This is what makes the interoperability claim real rather than aspirational:
    because the canonical model is already RDF, these are mappings rather than
    rewrites. It is the third level of the hierarchy the design left open --
    source schema -> canonical model -> standards mappings.
    """
    from ..knowledge.standards import export as project

    filter_clause = ""
    if building_ids:
        ids = ", ".join(f'"{b}"' for b in building_ids)
        filter_clause = ('BIND(REPLACE(STR(?b), "^.*/", "") AS ?bid) '
                         f'FILTER(?bid IN ({ids}))')
    rows = ctx.graph.query(f"""
        SELECT ?building ?site ?stream ?level ?area ?type ?year ?occupants
               ?city ?country ?timezone WHERE {{
          ?b a ekg:Building . BIND(?b AS ?building)
          OPTIONAL {{ ?b ekg:atSite ?site .
                     OPTIONAL {{ ?site ekg:timezone ?timezone }}
                     OPTIONAL {{ ?site ekg:inCity ?c .
                                ?c ekg:cityName ?city .
                                OPTIONAL {{ ?c ekg:inCountry ?co .
                                           ?co ekg:countryName ?country }} }} }}
          OPTIONAL {{ ?b ekg:hasStream ?stream .
                     OPTIONAL {{ ?stream ekg:streamLevel ?level }} }}
          OPTIONAL {{ ?b ekg:floorAreaM2 ?area }}
          OPTIONAL {{ ?b ekg:hasBuildingType ?t . ?t ekg:buildingTypeName ?type }}
          OPTIONAL {{ ?b ekg:yearBuilt ?year }}
          OPTIONAL {{ ?b ekg:occupants ?occupants }}
          {filter_clause}
        }} LIMIT {int(limit)}""")
    if not rows:
        return empty("no buildings matched for export")
    for row in rows:
        row["stream_level"] = row.pop("level", None) or "whole_building"

    projection = project(rows, target=target)
    payload = {"target": target, **projection.to_dict()}
    if projection.triples:
        path = ctx.store.root / f"export_{target}.ttl"
        path.write_text(projection.turtle())
        payload["turtle_path"] = str(path)
        payload["sample"] = projection.triples[:6]
        return ok(payload, evidence={
            "mapped_terms": projection.mapped_terms,
            "unmapped_terms": projection.unmapped_terms,
            "claim": "EnergyX does not require datasets to conform to a "
                     "standard; it exports to one where mappings exist"})
    records = projection.records
    return frame_result(pd.DataFrame(records), ctx.store,
                        name=f"export_{target}",
                        evidence={"target": target,
                                  "mapped_terms": projection.mapped_terms,
                                  "unmapped_terms": projection.unmapped_terms,
                                  "buildings": len(records),
                                  "claim": "EnergyX does not require datasets to "
                                           "conform to a standard; it exports to "
                                           "one where mappings exist"})
