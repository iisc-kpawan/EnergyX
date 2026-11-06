"""The Oxigraph-backed knowledge store.

Embedded RDF with SPARQL 1.1 -- no server. The store is **rebuilt
deterministically from the catalog**: it is a cache of a projection, never a
source of truth, so losing it costs a rebuild and nothing else.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyoxigraph as ox

from ..config import PATHS, SNAPSHOT_VERSION
from . import ontology as o
from .projector import CatalogProjector

XSD_MAP = {
    "string": None,
    "double": f"{o.XSD}double",
    "integer": f"{o.XSD}integer",
    "boolean": f"{o.XSD}boolean",
    "dateTime": f"{o.XSD}dateTime",
    "gYear": f"{o.XSD}gYear",
}


def _node(value) -> ox.NamedNode | ox.Literal:
    """A bare string that looks like an IRI is a node; a tuple is a literal."""
    if isinstance(value, tuple):
        raw, kind = value
        datatype = XSD_MAP.get(kind)
        if kind == "boolean":
            raw = "true" if bool(raw) else "false"
        elif kind == "dateTime":
            raw = pd.Timestamp(raw).isoformat()
        elif kind == "gYear":
            raw = str(int(raw))
        elif kind in ("double", "integer"):
            raw = str(int(raw)) if kind == "integer" else repr(float(raw))
        else:
            raw = str(raw)
        return (ox.Literal(raw, datatype=ox.NamedNode(datatype)) if datatype
                else ox.Literal(raw))
    return ox.NamedNode(str(value))


# Oxigraph is backed by RocksDB, which permits a single writer per directory.
# Two components each opening the store in one process collide, so handles are
# shared per path rather than per caller -- the agent runtime builds a fresh
# ToolContext per run and would otherwise deadlock itself.
_HANDLES: dict[str, ox.Store] = {}


class KnowledgeStore:
    """Build, persist and query the EnergyX knowledge graph."""

    def __init__(self, path: Path | None = None, snapshot_version: str = SNAPSHOT_VERSION):
        self.path = Path(path) if path else PATHS.snapshot / "graph"
        self.snapshot_version = snapshot_version
        self.read_only = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def store(self) -> ox.Store:
        """The graph, opened read-write here and read-only elsewhere.

        RocksDB admits one writer. A second process that opens the same store
        read-write does not raise -- it **segfaults** on the first query, taking
        the whole process with it: no exception, no traceback, a core dump from
        inside a SPARQL query.

        Opening read-only when another process holds the lock is safe and
        sufficient: every path that only reads facts -- the cache, the probe,
        the knowledge tools -- works unchanged, and only write-back needs the
        writer.
        """
        key = str(self.path.resolve())
        handle = _HANDLES.get(key)
        if handle is None:
            self.path.mkdir(parents=True, exist_ok=True)
            try:
                handle = ox.Store(str(self.path))
            except OSError:
                # Someone else is writing. Read-only shares cleanly.
                handle = ox.Store.read_only(str(self.path))
                self.read_only = True
            _HANDLES[key] = handle
        return handle

    def close(self) -> None:
        _HANDLES.pop(str(self.path.resolve()), None)

    def exists(self) -> bool:
        return self.path.exists() and any(self.path.iterdir())

    def build(self, db, dataset_id: str = "bdg2", rebuild: bool = True) -> dict:
        """Project the catalog into triples and load them."""
        if rebuild and self.path.exists():
            self.close()
            shutil.rmtree(self.path)
        projector = CatalogProjector(db, dataset_id=dataset_id)
        store = self.store
        written = 0
        for subject, predicate, value in projector.project():
            store.add(ox.Quad(ox.NamedNode(subject), ox.NamedNode(predicate),
                              _node(value)))
            written += 1
        report = {"triples": written, "sections": dict(projector.counts),
                  "snapshot_version": self.snapshot_version,
                  "built_at": datetime.utcnow().isoformat() + "Z",
                  "path": str(self.path)}
        return report

    # -- querying ----------------------------------------------------------
    def query(self, sparql: str, with_prefixes: bool = True) -> list[dict]:
        """Run a SELECT and return plain dicts with IRIs shortened."""
        text = f"{o.SPARQL_PREFIXES}\n{sparql}" if with_prefixes else sparql
        results = self.store.query(text)
        names = [str(v).lstrip("?") for v in getattr(results, "variables", [])]
        rows: list[dict] = []
        for solution in results:
            row: dict[str, Any] = {}
            for index, name in enumerate(names):
                row[name] = _plain(solution[index])
            rows.append(row)
        return rows

    def ask(self, sparql: str) -> bool:
        return bool(self.store.query(f"{o.SPARQL_PREFIXES}\n{sparql}"))

    def count(self) -> int:
        return len(self.store)

    def stats(self) -> dict:
        by_class = self.query("""
            SELECT ?c (COUNT(?s) AS ?n) WHERE { ?s a ?c }
            GROUP BY ?c ORDER BY DESC(?n)""")
        return {"triples": self.count(),
                "by_class": {r["c"].split("#")[-1]: int(r["n"]) for r in by_class}}

    def add_facts(self, quads: list[tuple]) -> int:
        """Insert derived facts, or none when this process cannot write.

        The store above opens read-only when another process holds the writer,
        which is the normal state for anything running beside the service. Only
        this method needs the writer, and it checks before writing rather than
        raising from inside the write: a cache write must never be the reason an
        answer does not arrive, when the answer itself is already complete. The count returned
        is what was written, so a caller can say nothing was.

        The flag alone is not enough to decide: handles are shared per path, so
        an instance can inherit a read-only handle it did not open. The write
        itself is the only reliable test.
        """
        written = 0
        for subject, predicate, value in quads:
            try:
                self.store.add(ox.Quad(ox.NamedNode(subject),
                                       ox.NamedNode(predicate), _node(value)))
            except RuntimeError:
                self.read_only = True
                return written
            written += 1
        return written


def _plain(value) -> Any:
    if isinstance(value, ox.NamedNode):
        return value.value
    if isinstance(value, ox.Literal):
        datatype = value.datatype.value if value.datatype else ""
        raw = value.value
        try:
            if datatype.endswith(("double", "float", "decimal")):
                return float(raw)
            if datatype.endswith(("integer", "int", "long")):
                return int(raw)
            if datatype.endswith("boolean"):
                return raw in ("true", "1")
        except ValueError:
            return raw
        return raw
    return str(value) if value is not None else None
