"""Serving analysis results from the knowledge layer.

A tool consults this before computing. The rule is deliberately strict: a cached
answer is served **only when every requested building has a current fact
computed by the same method with the same parameters**. A partial hit is not a
hit -- silently mixing cached and fresh values would make a result
irreproducible.

This is where the "improves with use" claim becomes measurable rather than
rhetorical: the same question costs a daily join the first time and a graph
lookup the second.
"""
from __future__ import annotations

import json

import pandas as pd

from .curator import KnowledgeCurator


def _params_match(recorded: str | None, wanted: dict) -> bool:
    if not wanted:
        return True
    try:
        stored = json.loads(recorded) if recorded else {}
    except (TypeError, ValueError):
        return False
    return all(str(stored.get(k)) == str(v) for k, v in wanted.items())


MAX_CACHE_COHORT = 400


def try_cached(ctx, kind: str, columns: dict[str, str], building_ids: list[str],
               sort_by: str | None = None, ascending: bool = False,
               params: dict | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Serve cached values, or ``None`` when the cache cannot reproduce the result.

    ``columns`` maps each output column to the method that produced it, because
    one tool can compute several quantities and a partial set is not a result.
    This is not pedantry: caching only the regression slope
    while the tool ranked by correlation, so a warm run returned a **differently
    ordered** answer to the same question. A cache that changes ordering changes
    answers.

    ``sort_by`` reproduces the computed path's ordering explicitly rather than
    inheriting whatever order the store returns.

    The second element always explains the decision, so a trace shows *why* a
    run was warm or cold rather than only that it was.
    """
    if not ctx.options.get("prefer_cached", True) or not building_ids:
        return None, {"cache": "disabled" if building_ids else "no_targets"}

    # A cohort this large is not what the cache is for, and asking anyway has
    # cost a process: current_facts builds a SPARQL FILTER holding one literal
    # per building, and at portfolio scale pyoxigraph 0.5.11 segfaults inside
    # query() -- no exception, no traceback, a core dump that takes the run with
    # it. The crash is not reliably reproducible in isolation, so this is a
    # guard against the query shape rather than a fix for the library. Computing
    # is always correct; the cache is only ever an optimisation.
    if len(building_ids) > MAX_CACHE_COHORT:
        return None, {"cache": "skipped",
                      "reason": f"{len(building_ids)} buildings exceeds the "
                                f"{MAX_CACHE_COHORT}-id cache lookup limit"}
    try:
        curator = KnowledgeCurator(ctx.graph, ctx.snapshot_version)
        facts = curator.current_facts(kind=kind, building_ids=list(building_ids))
    except Exception as exc:  # noqa: BLE001 - a cache miss must never fail a run
        return None, {"cache": "unavailable", "reason": str(exc)[:120]}

    # _params_match was written for this and never called, so a fact computed
    # over one window answered a question about another: Fox_office_Alice's
    # 2017 hourly temperature correlation came back as 0.3905 from a fact
    # computed days earlier over the whole record, where 2017 alone is 0.4557.
    # A fact that does not record the parameters it was computed under cannot
    # be shown to match, and computing is always correct.
    by_method: dict[str, dict] = {}
    unmatched = 0
    for fact in facts:
        if not fact.get("building_id"):
            continue
        if not _params_match(fact.get("params"), params or {}):
            unmatched += 1
            continue
        by_method.setdefault(fact.get("method", ""), {})[fact["building_id"]] = fact

    rows: dict[str, dict] = {b: {"building_id": b} for b in building_ids}
    for column, method in columns.items():
        available = by_method.get(method, {})
        missing = [b for b in building_ids if b not in available]
        if missing:
            return None, {"cache": "miss", "method": method, "column": column,
                          "hits": len(available), "misses": len(missing),
                          "missing_sample": missing[:5],
                          "rejected_on_params": unmatched or None,
                          "wanted_params": params or None}
        for b in building_ids:
            rows[b][column] = available[b].get("value")
            rows[b]["computed_at"] = available[b].get("computedAt")

    frame = pd.DataFrame(list(rows.values()))
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=ascending)
    return frame, {"cache": "hit", "methods": list(columns.values()),
                   "facts": len(frame), "columns": list(frame.columns),
                   "sorted_by": sort_by,
                   "note": "served from the knowledge layer; no measurements "
                           "scanned. Auxiliary diagnostics from the original "
                           "computation (sample counts, fit statistics) are not "
                           "cached -- disable the cache to recompute them."}
