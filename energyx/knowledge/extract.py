"""Turn analytical results into curatable facts and relations.

The router already knows which tools produce reusable output
(``CACHEABLE_TOOLS``); this module knows how to read each one. Extraction is
deliberately mechanical -- it reads columns and copies values. Anything that
required judgement would belong in an agent, not in a write-back path.

Every extractor records the ``params`` that produced the result, because a
change-point computed with a cooling model and one computed with a heating model
are different facts about the same building, not competing answers.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import ontology as o
from .curator import Fact, Relation


def _frame(result, store) -> pd.DataFrame | None:
    if isinstance(result.data, list) and result.data:
        return pd.DataFrame(result.data)
    if result.handle and store is not None:
        return store.get_or_none(result.handle)
    if isinstance(result.data, dict):
        return pd.DataFrame([result.data])
    return None


def _params(args: dict, keys: tuple[str, ...]) -> dict:
    return {k: args[k] for k in keys if k in args and args[k] is not None}


def _iso(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).tz_localize(None).isoformat() + "Z"
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:  # noqa: BLE001
            return None


# What a tool calls the single row it returns when asked to pool a cohort.
POOLED_LABELS = ("all buildings",)


def extract(tool: str, args: dict, result, store=None
            ) -> tuple[list[Fact], list[Relation]]:
    """Facts and relations implied by one successful tool result."""
    handler = _HANDLERS.get(tool)
    if handler is None or not result.ok:
        return [], []
    frame = _frame(result, store)
    if frame is None or frame.empty:
        return [], []
    # A pooled result is one row about a cohort, not about a building: the
    # tools label it "all buildings", and the knowledge store has no subject
    # for a cohort. Minting a fact for it fails on the space in the id --
    # "Invalid IRI code point" -- which surfaced as a 500 rather than as the
    # nothing it should be.
    if "building_id" in frame.columns:
        frame = frame[~frame["building_id"].isin(POOLED_LABELS)]
        if frame.empty:
            return [], []
    try:
        return handler(args, frame, result)
    except Exception:  # noqa: BLE001 - extraction must never break a run
        return [], []


# --- per-tool extractors ----------------------------------------------------

def _change_points(args, frame, result):
    model = args.get("model", "cooling")
    column = next((c for c in frame.columns if c.endswith("change_point_c")), None)
    if column is None:
        return [], []
    params = _params(args, ("model", "start", "end"))
    facts = [Fact(kind="ChangePoint", building_id=row["building_id"],
                  value=row[column], unit="degC",
                  method=f"piecewise_changepoint:{model}", params=params,
                  valid_from=_iso(args.get("start")), valid_to=_iso(args.get("end")),
                  confidence=float(row.get("r_squared", 1.0) or 0.0))
             for _, row in frame.iterrows() if pd.notna(row.get(column))]
    return facts, []


def _anomalies(args, frame, result):
    if "when" not in frame.columns:
        return [], []
    method = args.get("method", "daily_profile")
    params = _params(args, ("method", "threshold", "min_coverage", "start", "end"))
    facts = [Fact(kind="AnomalyEvent", building_id=row["building_id"],
                  value=row.get("score"), unit="z",
                  method=f"anomaly:{method}", params=params,
                  valid_from=_iso(row["when"]), valid_to=_iso(row["when"]))
             for _, row in frame.iterrows() if pd.notna(row.get("when"))]
    return facts, []


def _trend_segments(args, frame, result):
    if "change_date" not in frame.columns:
        return [], []
    params = _params(args, ("min_segment_days", "min_shift_pct"))
    facts = [Fact(kind="TrendSegment", building_id=row["building_id"],
                  value=row.get("shift_pct"), unit="percent",
                  method="binary_segmentation", params=params,
                  valid_from=_iso(row["change_date"]))
             for _, row in frame.iterrows() if pd.notna(row.get("change_date"))]
    return facts, []


def _operating(args, frame, result):
    if "peak_hour" not in frame.columns:
        return [], []
    params = _params(args, ("threshold", "start", "end"))
    facts = [Fact(kind="OperatingSchedule", building_id=row["building_id"],
                  value=row.get("peak_hour"), unit="hour_local",
                  method="operating_pattern", params=params,
                  valid_from=_iso(args.get("start")), valid_to=_iso(args.get("end")),
                  extra={"start_hour": row.get("start_hour"),
                         "end_hour": row.get("end_hour")})
             for _, row in frame.iterrows() if pd.notna(row.get("peak_hour"))]
    return facts, []


def _performance(args, frame, result):
    metric = args.get("metric", "eui")
    if metric not in frame.columns:
        return [], []
    unit = {"eui": "kWh/m2", "peak_per_area": "kW/m2",
            "per_occupant": "kWh/person"}.get(metric, "")
    params = _params(args, ("metric", "year", "min_coverage"))
    facts = [Fact(kind="PerformanceScore", building_id=row["building_id"],
                  value=row[metric], unit=unit,
                  method=f"normalised:{metric}", params=params,
                  valid_from=_iso(f"{args['year']}-01-01") if args.get("year") else None,
                  valid_to=_iso(f"{int(args['year']) + 1}-01-01") if args.get("year") else None)
             for _, row in frame.iterrows() if pd.notna(row.get(metric))]
    return facts, []


def _peer_group(args, frame, result):
    data = result.data if isinstance(result.data, dict) else {}
    members = data.get("peer_group") or []
    if not members:
        return [], []
    predicate = data.get("predicate", "")
    group = f"{o.EKGD}peergroup/{abs(hash(predicate)) % 10**12}"
    relations = [Relation(predicate="memberOf", subject_id=member,
                          object_iri=group, method="construct_peer_group",
                          params={"criteria": args.get("match_on"),
                                  "predicate": predicate})
                 for member in members]
    target = args.get("building_id")
    if target:
        relations.append(Relation(predicate="memberOf", subject_id=target,
                                  object_iri=group, method="construct_peer_group",
                                  params={"role": "target"}))
    return [], relations


def _clusters(args, frame, result):
    if not {"building_id", "cluster"} <= set(frame.columns):
        return [], []
    params = _params(args, ("n_clusters", "normalize", "method"))
    relations = [Relation(predicate="inCluster", subject_id=row["building_id"],
                          object_iri=f"{o.EKGD}cluster/k{args.get('n_clusters', 4)}"
                                     f"/{int(row['cluster'])}",
                          weight=float(row["cluster"]),
                          weight_property="clusterLabel",
                          method="kmeans_load_shape", params=params)
                 for _, row in frame.iterrows()]
    return [], relations


def _similarity(args, frame, result):
    target = args.get("building_id")
    if not target or "building_id" not in frame.columns:
        return [], []
    params = _params(args, ("metric", "normalize", "k"))
    relations = [Relation(predicate="similarLoadShape", subject_id=target,
                          object_iri=f"{o.EKGD}building/bdg2/{row['building_id']}",
                          weight=float(row.get("similarity", 0.0)),
                          weight_property="similarity",
                          method="daily_profile_similarity", params=params)
                 for _, row in frame.iterrows() if pd.notna(row.get("similarity"))]
    return [], relations


def _benchmark(args, frame, result):
    data = result.data if isinstance(result.data, dict) else {}
    if "percentile" not in data:
        return [], []
    building = data.get("building_id") or args.get("building_id")
    params = _params(args, ("metric", "year", "min_coverage"))
    fact = Fact(kind="PerformanceScore", building_id=building,
                value=data["percentile"], unit="percentile",
                method=f"peer_percentile:{data.get('metric', 'eui')}",
                params=params,
                valid_from=_iso(f"{args['year']}-01-01") if args.get("year") else None)
    return [fact], []


def _weather_relationship(args, frame, result):
    """Cache every quantity the tool can rank on, not only the one requested.

    The tool ranks by correlation even when asked for a regression, so caching
    the slope alone would let a warm run return a differently ordered answer.
    """
    variable = args.get("variable", "air_temperature")
    params = _params(args, ("variable", "method", "grain", "start", "end"))
    units = {"slope": "kWh/unit", "correlation": "r"}
    facts: list[Fact] = []
    for column in ("correlation", "slope"):
        if column not in frame.columns:
            continue
        for _, row in frame.iterrows():
            if pd.isna(row.get(column)):
                continue
            facts.append(Fact(
                kind="PerformanceScore", building_id=row["building_id"],
                value=row[column], unit=units[column],
                method=f"weather_sensitivity:{variable}:{column}", params=params,
                valid_from=_iso(args.get("start")), valid_to=_iso(args.get("end")),
                confidence=float(row.get("r_squared", 1.0) or 0.0)
                if "r_squared" in frame.columns else 1.0))
    return facts, []


def _baseload(args, frame, result):
    if "baseload" not in frame.columns:
        return [], []
    method = args.get("method", "night_mean")
    params = _params(args, ("method", "percentile", "start", "end"))
    facts: list[Fact] = []
    # Every column the tool returns, not just the headline two. A cache that
    # can rebuild only part of a result cannot serve it -- a partial set is a
    # different answer, and that is why this tool could never be served
    # warm despite writing facts on every run.
    for column, unit in (("baseload", "kW"), ("baseload_ratio", "ratio"),
                         ("mean_demand", "kW"), ("n_observations", "count")):
        if column not in frame.columns:
            continue
        for _, row in frame.iterrows():
            if pd.isna(row.get(column)):
                continue
            facts.append(Fact(
                kind="PerformanceScore", building_id=row["building_id"],
                value=row[column], unit=unit,
                method=f"baseload:{method}:{column}", params=params,
                valid_from=_iso(args.get("start")), valid_to=_iso(args.get("end"))))
    return facts, []


_HANDLERS: dict[str, Any] = {
    "calculate_baseload": _baseload,
    "analyze_weather_relationship": _weather_relationship,
    "fit_temperature_response": _change_points,
    "detect_anomalies": _anomalies,
    "detect_change_points": _trend_segments,
    "analyze_operating_pattern": _operating,
    "calculate_normalized_performance": _performance,
    "construct_peer_group": _peer_group,
    "cluster_timeseries": _clusters,
    "find_similar_timeseries": _similarity,
    "benchmark_building": _benchmark,
}

EXTRACTABLE_TOOLS = tuple(_HANDLERS)
