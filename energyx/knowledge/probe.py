"""What the knowledge layer already knows, before a plan is written.

Planning decides the **strategy** -- whether computation is needed at all
 -- and it cannot decide that without knowing what has already been
computed. This probe answers that question cheaply, before any step runs, so
``cached_fact`` becomes a strategy the planner can actually choose rather than
one it merely declares.

Deliberately coarse: it reports which methods have facts and for how many
buildings, not the values. A planner needs to know a path is *available*; the
tools fetch what it says.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re

from .curator import KnowledgeCurator


def method_column(method: str) -> str:
    """Column name a method takes in an evidence table.

    Shared with ``lookup_derived_facts(as_evidence=True)`` so a planner can name
    a criterion before the column exists.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", str(method)).strip("_")


@dataclass
class KnownFacts:
    methods: dict[str, int] = field(default_factory=dict)   # method -> buildings
    buildings: int = 0
    total_facts: int = 0

    def covers(self, methods: list[str], minimum: int = 1) -> bool:
        """Are all these methods available for at least ``minimum`` buildings?"""
        return all(self.methods.get(m, 0) >= minimum for m in methods)

    def available(self, prefix: str) -> list[str]:
        return sorted(m for m in self.methods if m.startswith(prefix))

    def summary(self, limit: int = 8) -> str:
        if not self.methods:
            return "no derived facts recorded yet"
        top = sorted(self.methods.items(), key=lambda kv: -kv[1])[:limit]
        return "; ".join(f"{m} ({n} buildings)" for m, n in top)

    def to_dict(self) -> dict:
        return {"methods": self.methods, "buildings": self.buildings,
                "total_facts": self.total_facts}


def probe(graph, snapshot_version: str) -> KnownFacts:
    """Summarise the current (non-superseded) facts for this snapshot."""
    try:
        curator = KnowledgeCurator(graph, snapshot_version)
        facts = curator.current_facts()
    except Exception:  # noqa: BLE001 - an unavailable graph means cold planning
        return KnownFacts()
    known = KnownFacts(total_facts=len(facts))
    seen: dict[str, set[str]] = {}
    buildings: set[str] = set()
    for fact in facts:
        method = str(fact.get("method") or "")
        building = str(fact.get("building_id") or "")
        if not method or not building:
            continue
        seen.setdefault(method, set()).add(building)
        buildings.add(building)
    known.methods = {m: len(b) for m, b in seen.items()}
    known.buildings = len(buildings)
    return known
