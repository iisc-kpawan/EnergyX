"""EnergyX's knowledge graph, read-only.

The graph is an Oxigraph store built from the snapshot's catalog (EnergyX's scripts/build_graph.py).
EnergyX's agents also write facts they compute back into it, so gold tasks should query only
facts projected from the catalog: datasets, sites, buildings, streams and places.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyoxigraph as ox

PREFIXES = {
    "ekg": "http://energyx.org/ns#",
    "ekgd": "http://energyx.org/id/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
_XSD = PREFIXES["xsd"]
_INTEGERS = {f"{_XSD}{t}" for t in ("integer", "int", "long", "short", "nonNegativeInteger")}
_FLOATS = {f"{_XSD}{t}" for t in ("decimal", "double", "float")}


def _python_value(term):
    if term is None:
        return None
    if isinstance(term, ox.Literal):
        datatype = term.datatype.value
        if datatype in _INTEGERS:
            return int(term.value)
        if datatype in _FLOATS:
            return float(term.value)
        if datatype == f"{_XSD}boolean":
            return term.value == "true"
        return term.value
    return term.value  # a NamedNode's IRI or a BlankNode's id


class KnowledgeGraph:
    def __init__(self, graph_dir: Path):
        if not graph_dir.exists():
            raise FileNotFoundError(f"no knowledge graph at {graph_dir}; build it with EnergyX's scripts/build_graph.py")
        self._store = ox.Store.read_only(str(graph_dir))

    def query(self, sparql: str) -> pd.DataFrame:
        prefixes = "\n".join(f"PREFIX {p}: <{iri}>" for p, iri in PREFIXES.items())
        solutions = self._store.query(f"{prefixes}\n{sparql}")
        if not isinstance(solutions, ox.QuerySolutions):
            raise ValueError("query_kg supports SELECT queries only")
        columns = [variable.value for variable in solutions.variables]
        rows = [[_python_value(solution[name]) for name in columns] for solution in solutions]
        return pd.DataFrame(rows, columns=columns)
