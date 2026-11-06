"""The Knowledge Curator.

A **Layer 3 service, not an agent**. It validates, pins, deduplicates,
supersedes and asserts -- and it does not reason. Letting a write-back path
reason is how a speculative conclusion becomes a permanent fact.

Three disciplines it exists to enforce:

* **Every fact is pinned** with snapshot, method, params, confidence, assertion
  time, originating run and ``assertionType``. Without the last one a
  wrong finding is indistinguishable from ground truth.
* **Facts are superseded, never overwritten.** A re-computation records a new
  belief and points the old one at it, so assertion time stays meaningful and an
  earlier answer remains inspectable.
* **Bi-temporal.** Valid time (``validFrom``/``validTo`` -- when the fact was
  true of the building) is distinct from assertion time (``computedAt`` -- when
  EnergyX came to believe it). Building energy is inherently temporal: an
  operating schedule that changed in March 2017 is not wrong, it is bounded.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import ontology as o


@dataclass
class Fact:
    """One derived assertion, before it reaches the store."""

    kind: str                              # a declared Finding subtype
    building_id: str
    value: float | None = None
    unit: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    method: str = ""
    params: dict = field(default_factory=dict)
    confidence: float = 1.0
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        """Identity for dedupe: what this fact is *about*, not what it says.

        Two computations of the same quantity over the same window with the same
        method are the same fact; the later one supersedes the earlier.
        """
        payload = json.dumps({
            "kind": self.kind, "building": self.building_id,
            "method": self.method,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "valid_from": self.valid_from, "valid_to": self.valid_to,
        }, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass
class Relation:
    """A weighted relation between two entities."""

    predicate: str                          # similarLoadShape | memberOf | ...
    subject_id: str
    object_iri: str
    weight: float | None = None
    weight_property: str = "weight"
    method: str = ""
    params: dict = field(default_factory=dict)

    def key(self) -> str:
        payload = f"{self.predicate}|{self.subject_id}|{self.object_iri}|{self.method}"
        return hashlib.sha256(payload.encode()).hexdigest()[:20]


@dataclass
class CurationReport:
    asserted: int = 0
    superseded: int = 0
    duplicates: int = 0
    rejected: list[str] = field(default_factory=list)
    triples: int = 0
    facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"asserted": self.asserted, "superseded": self.superseded,
                "duplicates": self.duplicates, "rejected": self.rejected[:10],
                "triples": self.triples}


class KnowledgeCurator:
    """Validate, pin, dedupe, supersede, assert. No reasoning."""

    def __init__(self, graph, snapshot_version: str, run_id: str = ""):
        self.graph = graph
        self.snapshot_version = snapshot_version
        self.run_id = run_id

    # -- lookup ------------------------------------------------------------
    def _existing(self, fact_key: str) -> str | None:
        """The current (non-superseded) node for a fact key, if any."""
        rows = self.graph.query(f"""
            SELECT ?f WHERE {{
              ?f ekgp:factKey "{fact_key}" ;
                 ekgp:snapshotVersion "{self.snapshot_version}" .
              FILTER NOT EXISTS {{ ?f ekgp:supersededBy ?newer }}
            }} LIMIT 1""")
        return rows[0]["f"] if rows else None

    # -- assertion ---------------------------------------------------------
    def assert_facts(self, facts: list[Fact],
                     supersede: bool = True) -> CurationReport:
        report = CurationReport()
        now = datetime.now(timezone.utc).isoformat()
        quads: list[tuple] = []

        for fact in facts:
            if fact.kind not in o.FINDING_TYPES:
                report.rejected.append(f"{fact.kind!r} is not a declared finding type")
                continue
            if not fact.building_id:
                report.rejected.append(f"{fact.kind} without a building_id")
                continue
            if fact.value is not None and not _finite(fact.value):
                report.rejected.append(
                    f"{fact.kind} for {fact.building_id} has a non-finite value")
                continue

            key = fact.key()
            previous = self._existing(key)
            if previous and not supersede:
                report.duplicates += 1
                continue

            node = f"{o.EKGD}finding/{uuid.uuid4().hex[:12]}"
            target = f"{o.EKGD}building/bdg2/{fact.building_id}"
            quads += [
                (node, f"{o.RDF}type", o.term(fact.kind)),
                (node, f"{o.RDF}type", o.term("Finding")),
                (node, o.term("aboutBuilding"), target),
                (target, o.term("hasFinding"), node),
                (node, o.prov_term("factKey"), (key, "string")),
                (node, o.prov_term("assertionType"), ("derived", "string")),
                (node, o.prov_term("snapshotVersion"), (self.snapshot_version, "string")),
                (node, o.prov_term("method"), (fact.method, "string")),
                (node, o.prov_term("confidence"), (float(fact.confidence), "double")),
                (node, o.prov_term("computedAt"), (now, "dateTime")),
                (node, o.prov_term("computedByRun"), (self.run_id, "string")),
            ]
            if fact.params:
                quads.append((node, o.prov_term("params"),
                              (json.dumps(fact.params, sort_keys=True, default=str),
                               "string")))
            if fact.value is not None:
                quads.append((node, o.term("findingValue"), (float(fact.value), "double")))
            if fact.unit:
                quads.append((node, o.term("findingUnit"), (fact.unit, "string")))
            for attribute, prop in (("valid_from", "validFrom"), ("valid_to", "validTo")):
                value = getattr(fact, attribute)
                if value:
                    quads.append((node, o.term(prop), (value, "dateTime")))
            if previous:
                # Point the old belief at the new one rather than deleting it.
                quads.append((previous, o.prov_term("supersededBy"), (node, "string")))
                quads.append((node, o.prov_term("supersedes"), (previous, "string")))
                report.superseded += 1
            report.asserted += 1
            report.facts.append(node)

        report.triples = self.graph.add_facts(quads) if quads else 0
        return report

    def assert_relations(self, relations: list[Relation]) -> CurationReport:
        """Write a weighted relation as both a plain edge and an assertion node.

        The edge keeps traversal simple; the node carries the weight and
        provenance, because pyoxigraph cannot quote a triple.
        """
        report = CurationReport()
        now = datetime.now(timezone.utc).isoformat()
        quads: list[tuple] = []
        for relation in relations:
            if relation.predicate not in o.OBJECT_PROPERTIES:
                report.rejected.append(f"{relation.predicate!r} is not declared")
                continue
            subject = f"{o.EKGD}building/bdg2/{relation.subject_id}"
            node = f"{o.EKGD}rel/{uuid.uuid4().hex[:12]}"
            quads += [
                (subject, o.term(relation.predicate), relation.object_iri),
                (node, f"{o.RDF}type", o.term("RelationAssertion")),
                (node, o.term("relSubject"), subject),
                (node, o.term("relObject"), relation.object_iri),
                (node, o.term("relPredicate"), o.term(relation.predicate)),
                (subject, o.term("assertsRelation"), node),
                (node, o.prov_term("factKey"), (relation.key(), "string")),
                (node, o.prov_term("assertionType"), ("derived", "string")),
                (node, o.prov_term("snapshotVersion"),
                 (self.snapshot_version, "string")),
                (node, o.prov_term("method"), (relation.method, "string")),
                (node, o.prov_term("computedAt"), (now, "dateTime")),
                (node, o.prov_term("computedByRun"), (self.run_id, "string")),
            ]
            if relation.weight is not None and _finite(relation.weight):
                quads.append((node, o.term(relation.weight_property),
                              (float(relation.weight), "double")))
            if relation.params:
                quads.append((node, o.prov_term("params"),
                              (json.dumps(relation.params, sort_keys=True,
                                          default=str), "string")))
            report.asserted += 1
        report.triples = self.graph.add_facts(quads) if quads else 0
        return report

    # -- reading -----------------------------------------------------------
    def current_facts(self, kind: str | None = None,
                      building_ids: list[str] | None = None,
                      at: str | None = None) -> list[dict]:
        """Facts believed now, optionally as of a point in valid time.

        Superseded beliefs and facts pinned to another snapshot are never
        served(, 2.6).
        """
        type_clause = f"?f a ekg:{kind} ." if kind else "?f a ekg:Finding ."
        valid_clause = ""
        if at:
            valid_clause = f"""
              OPTIONAL {{ ?f ekg:validFrom ?vf }}
              OPTIONAL {{ ?f ekg:validTo ?vt }}
              FILTER(!BOUND(?vf) || ?vf <= "{at}"^^xsd:dateTime)
              FILTER(!BOUND(?vt) || ?vt >= "{at}"^^xsd:dateTime)"""
        building_clause = ""
        if building_ids:
            ids = ", ".join(f'"{b}"' for b in building_ids)
            building_clause = f"""
              BIND(REPLACE(STR(?b), "^.*/", "") AS ?bid)
              FILTER(?bid IN ({ids}))"""
        rows = self.graph.query(f"""
            SELECT DISTINCT ?f ?type ?b ?value ?unit ?method ?confidence
                            ?computedAt ?validFrom ?validTo WHERE {{
              {type_clause}
              ?f a ?type ;
                 ekg:aboutBuilding ?b ;
                 ekgp:snapshotVersion "{self.snapshot_version}" ;
                 ekgp:method ?method ;
                 ekgp:computedAt ?computedAt .
              FILTER(?type != ekg:Finding)
              FILTER NOT EXISTS {{ ?f ekgp:supersededBy ?newer }}
              OPTIONAL {{ ?f ekg:findingValue ?value }}
              OPTIONAL {{ ?f ekg:findingUnit ?unit }}
              OPTIONAL {{ ?f ekgp:confidence ?confidence }}
              OPTIONAL {{ ?f ekg:validFrom ?validFrom }}
              OPTIONAL {{ ?f ekg:validTo ?validTo }}
              {valid_clause}
              {building_clause}
            }} LIMIT 1000""")
        for row in rows:
            row["type"] = str(row.get("type", "")).split("#")[-1]
            row["building_id"] = str(row.get("b", "")).split("/")[-1]
        return rows

    def stats(self) -> dict:
        by_type = self.graph.query("""
            SELECT ?t (COUNT(?f) AS ?n) WHERE {
              ?f a ekg:Finding ; a ?t .
              FILTER(?t != ekg:Finding)
              FILTER NOT EXISTS { ?f ekgp:supersededBy ?newer }
            } GROUP BY ?t""")
        superseded = self.graph.query("""
            SELECT (COUNT(?f) AS ?n) WHERE { ?f ekgp:supersededBy ?newer }""")
        relations = self.graph.query("""
            SELECT (COUNT(?r) AS ?n) WHERE { ?r a ekg:RelationAssertion }""")
        return {
            "current_facts": {str(r["t"]).split("#")[-1]: int(r["n"]) for r in by_type},
            "superseded": int(superseded[0]["n"]) if superseded else 0,
            "relations": int(relations[0]["n"]) if relations else 0,
        }


def _finite(value) -> bool:
    try:
        return pd.notna(value) and float(value) not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False
