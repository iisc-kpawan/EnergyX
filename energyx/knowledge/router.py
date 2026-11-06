"""The unified query router (Layer 4).

Planning decides the **strategy** -- whether computation is needed at all. This
router decides the **engine** -- which store answers a given sub-query. Keeping
them apart matters: the first is reasoning informed by capability metadata and
fact freshness, the second is mechanical dispatch. Conflating them
buries the interesting decision inside the data layer.

The router also makes strategy accuracy measurable: it knows the engine each
tool actually uses, so a plan that claims ``graph_only`` for a measurement scan
can be scored as wrong rather than merely being slow.
"""
from __future__ import annotations

from dataclasses import dataclass

GRAPH, CATALOG, MEASUREMENT, MODEL, PRESENTATION = (
    "graph", "catalog", "measurement", "model", "presentation")

# Which store each tool actually reads. Derived from the implementations, not
# aspirational: `search_catalog` reads catalog parquet, `fetch_electricity`
# scans measurements.
TOOL_ENGINE: dict[str, str] = {
    # knowledge
    "query_knowledge_graph": GRAPH, "traverse_relations": GRAPH,
    "lookup_derived_facts": GRAPH, "assert_findings": GRAPH,
    "find_entity_paths": GRAPH, "export_semantic_model": GRAPH,
    # catalog-only
    "search_catalog": CATALOG, "find_buildings": CATALOG,
    "get_building_metadata": CATALOG, "get_data_availability": CATALOG,
    "check_task_eligibility": CATALOG, "profile_data_quality": CATALOG,
    "aggregate_geographically": CATALOG, "compare_geographies": CATALOG,
    "analyze_spatiotemporal_patterns": CATALOG,
    "prepare_geospatial_analysis": CATALOG,
    # measurement scans
    "fetch_electricity": MEASUREMENT, "fetch_weather": MEASUREMENT,
    "fetch_contextual_data": MEASUREMENT, "detect_data_issues": MEASUREMENT,
    "clean_timeseries": MEASUREMENT, "resample_timeseries": MEASUREMENT,
    "impute_missing_values": MEASUREMENT, "align_timeseries": MEASUREMENT,
    "convert_timezone": MEASUREMENT,
    "calculate_consumption_statistics": MEASUREMENT,
    "run_structured_query": MEASUREMENT,
    "calculate_demand_metrics": MEASUREMENT, "calculate_load_profile": MEASUREMENT,
    "compare_temporal_profiles": MEASUREMENT, "calculate_baseload": MEASUREMENT,
    "analyze_operating_pattern": MEASUREMENT,
    "analyze_consumption_trend": MEASUREMENT,
    "calculate_normalized_performance": MEASUREMENT,
    "construct_peer_group": CATALOG, "benchmark_building": MEASUREMENT,
    "compare_building_groups": MEASUREMENT, "rank_buildings": MEASUREMENT,
    "identify_peer_outliers": MEASUREMENT,
    "analyze_weather_relationship": MEASUREMENT,
    "fit_temperature_response": MEASUREMENT,
    "weather_normalize_consumption": MEASUREMENT,
    # model-backed
    "forecast_timeseries": MODEL, "evaluate_forecast": MODEL,
    "detect_anomalies": MODEL, "detect_change_points": MODEL,
    "cluster_timeseries": MODEL, "find_similar_timeseries": MODEL,
    "generate_timeseries_representations": MODEL,
    "analyze_ai_model_results": MODEL,
    # presentation
    "combine_evidence": PRESENTATION, "prioritize_entities": PRESENTATION,
    "generate_decision_support": PRESENTATION, "create_visualization": PRESENTATION,
    "generate_report": PRESENTATION,
    # onboarding
    "inspect_dataset": CATALOG, "map_to_energyx_schema": CATALOG,
    "validate_dataset": CATALOG, "register_dataset": CATALOG,
}

# The resolution strategy a correct plan should declare for each engine.
ENGINE_RESOLUTION = {
    GRAPH: "graph_only",
    CATALOG: "graph_only",        # catalog lookups scan no measurements either
    MEASUREMENT: "compute",
    MODEL: "compute",
    PRESENTATION: "compute",
}

CACHEABLE_TOOLS = {
    "calculate_baseload": "PerformanceScore",
    "analyze_weather_relationship": "PerformanceScore",
    "benchmark_building": "PerformanceScore",
    "construct_peer_group": "PeerGroup",
    "cluster_timeseries": "LoadShapeCluster",
    "find_similar_timeseries": "similarLoadShape",
    "fit_temperature_response": "ChangePoint",
    "detect_anomalies": "AnomalyEvent",
    "detect_change_points": "TrendSegment",
    "analyze_operating_pattern": "OperatingSchedule",
    "calculate_normalized_performance": "PerformanceScore",
}


@dataclass
class Routing:
    tool: str
    engine: str
    expected_resolution: str
    declared_resolution: str
    scans_measurements: bool
    cacheable_as: str | None = None

    @property
    def consistent(self) -> bool:
        """Did the plan's strategy match the engine the tool actually uses?

        `cached_fact` and `hybrid` are accepted against a compute engine: both
        legitimately mean "some of this may be avoidable".

        `cached_fact` is also accepted against the *graph* engine, because
        that is where a derived fact lives: reading one is a graph read by
        construction.
        """
        if self.declared_resolution == self.expected_resolution:
            return True
        if self.expected_resolution == "compute" and \
                self.declared_resolution in ("hybrid", "cached_fact"):
            return True
        if self.expected_resolution == "graph_only" and \
                self.declared_resolution == "cached_fact":
            return True
        return False

    def to_dict(self) -> dict:
        return {"tool": self.tool, "engine": self.engine,
                "expected_resolution": self.expected_resolution,
                "declared_resolution": self.declared_resolution,
                "consistent": self.consistent,
                "scans_measurements": self.scans_measurements,
                "cacheable_as": self.cacheable_as}


class QueryRouter:
    """Mechanical dispatch from a plan step to the engine that answers it."""

    def engine_for(self, tool: str) -> str:
        return TOOL_ENGINE.get(tool, MEASUREMENT)

    def expected_resolution(self, tool: str) -> str:
        return ENGINE_RESOLUTION[self.engine_for(tool)]

    def route(self, tool: str, declared_resolution: str = "compute",
              result=None) -> Routing:
        """Engine for a step. A cache hit is served by the graph whatever the
        tool would otherwise have scanned, and the trace should reflect that."""
        engine = self.engine_for(tool)
        if tool == "profile_data_quality" and result is not None and (
                getattr(result, "evidence", {}) or {}).get("period"):
            engine = MEASUREMENT
        if result is not None and (getattr(result, "evidence", {}) or {}
                                   ).get("cache") == "hit":
            engine = GRAPH
        return Routing(tool=tool, engine=engine,
                       expected_resolution=ENGINE_RESOLUTION[engine],
                       declared_resolution=declared_resolution,
                       scans_measurements=engine in (MEASUREMENT, MODEL),
                       cacheable_as=CACHEABLE_TOOLS.get(tool))

    def plan_profile(self, plan) -> dict:
        """Engine mix and strategy accuracy for a whole plan."""
        routings = [self.route(s.tool, s.resolution) for s in plan.steps]
        by_engine: dict[str, int] = {}
        for r in routings:
            by_engine[r.engine] = by_engine.get(r.engine, 0) + 1
        consistent = sum(1 for r in routings if r.consistent)
        return {
            "steps": len(routings),
            "by_engine": by_engine,
            "strategy_consistent": consistent,
            "strategy_accuracy": consistent / len(routings) if routings else 1.0,
            "measurement_steps": sum(1 for r in routings if r.scans_measurements),
            "cacheable_steps": [r.tool for r in routings if r.cacheable_as],
            "mismatches": [r.to_dict() for r in routings if not r.consistent],
        }
