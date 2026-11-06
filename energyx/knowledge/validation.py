"""Validate SPARQL before it runs.

Two failure modes this prevents, both of which look like a correct empty answer:

* a query against a property that does not exist returns zero rows;
* a typo in a class name returns zero rows.

Generated queries are therefore checked against the declared vocabulary first
. Anything that would mutate the store is refused outright: the graph is
a projection rebuilt from the catalog, and a generated query has no business
writing to it -- derived facts go through the Knowledge Curator instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import ontology as o

PREFIXED = re.compile(r"\b(ekg|ekgp)\s*:\s*([A-Za-z_][A-Za-z0-9_\-]*)")
FULL_IRI = re.compile(r"<(https?://[^>]+)>")
FORBIDDEN = re.compile(
    r"\b(INSERT|DELETE|DROP|CLEAR|LOAD|CREATE|COPY|MOVE|ADD)\b", re.I)
QUERY_FORM = re.compile(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.I)
SERVICE = re.compile(r"\bSERVICE\b", re.I)


@dataclass
class QueryValidation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)
    form: str | None = None

    def message(self) -> str:
        return "; ".join(self.problems)


def suggest(term: str, vocabulary) -> str | None:
    """Nearest declared term, so an error teaches rather than just refuses."""
    import difflib
    matches = difflib.get_close_matches(term, sorted(vocabulary), n=1, cutoff=0.72)
    return matches[0] if matches else None


def validate_query(sparql: str) -> QueryValidation:
    problems: list[str] = []
    unknown: list[str] = []

    if FORBIDDEN.search(sparql):
        problems.append("update operations are not permitted; the graph is a "
                        "projection rebuilt from the catalog")
    if SERVICE.search(sparql):
        problems.append("federated SERVICE calls are not permitted")

    form_match = QUERY_FORM.search(sparql)
    form = form_match.group(1).upper() if form_match else None
    if form is None:
        problems.append("no SELECT, ASK, CONSTRUCT or DESCRIBE form found")

    classes_and_props = set(o.CLASSES) | o.ALL_PROPERTIES
    for prefix, local in PREFIXED.findall(sparql):
        if prefix == "ekg":
            if local not in classes_and_props:
                unknown.append(f"ekg:{local}")
                hint = suggest(local, classes_and_props)
                problems.append(f"unknown term ekg:{local}"
                                + (f" (did you mean ekg:{hint}?)" if hint else ""))
        elif prefix == "ekgp" and local not in o.ALL_PROV_PROPERTIES:
            unknown.append(f"ekgp:{local}")
            hint = suggest(local, o.ALL_PROV_PROPERTIES)
            problems.append(f"unknown term ekgp:{local}"
                            + (f" (did you mean ekgp:{hint}?)" if hint else ""))

    for iri in FULL_IRI.findall(sparql):
        if iri.startswith((o.EKG, o.EKGP)) and not o.known_term(iri):
            unknown.append(iri)
            problems.append(f"unknown term <{iri}>")

    return QueryValidation(ok=not problems, problems=problems,
                           unknown_terms=unknown, form=form)


def guard(sparql: str, limit: int = 1000) -> str:
    """Attach a LIMIT to an unbounded SELECT so one query cannot flood a context."""
    if re.search(r"\bLIMIT\s+\d+", sparql, re.I):
        return sparql
    if re.search(r"\bSELECT\b", sparql, re.I):
        return f"{sparql.rstrip().rstrip(';')}\nLIMIT {limit}"
    return sparql
