"""Exercise every registered tool against the real snapshot.

Each tool is callable, returns a typed
ToolResult, and either produces a result or refuses with a named predicate.
Nothing here asserts a scientific finding -- it asserts the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyx.access import SnapshotDB           # noqa: E402
from energyx.config import SNAPSHOT_VERSION     # noqa: E402
from energyx.tools import (                     # noqa: E402
    ArtifactStore, Status, ToolContext, call, load_all, registry,
)

load_all()


def main() -> int:
    db = SnapshotDB()
    store = ArtifactStore.for_run("r_smoke")
    ctx = ToolContext(db=db, store=store, snapshot_version=SNAPSHOT_VERSION)

    seed = call("find_buildings", ctx, country="UK", building_type="Office",
                requires=["eui_analysis", "weather_analysis"], limit=8)
    ids = [r["building_id"] for r in (seed.data or [])] or ["Robin_office_Addie"]
    one = ids[0]
    peers = call("construct_peer_group", ctx, building_id=one,
                 match_on=["building_type", "country"])
    peer_ids = (peers.data or {}).get("peer_group", ids)[:20]

    elec = call("fetch_electricity", ctx, building_ids=ids[:3],
                start="2017-07-01", end="2017-08-01")
    forecast = call("forecast_timeseries", ctx, building_id=one,
                    horizon_hours=24, cutoff="2017-07-31")
    perf = call("calculate_normalized_performance", ctx, building_ids=peer_ids,
                metric="eui", year=2017)
    profile = call("calculate_load_profile", ctx, building_ids=ids[:5],
                   profile="hourly")

    plan: list[tuple[str, dict]] = [
        ("search_catalog", {"entity": "countries"}),
        # Metadata completeness reads the catalog, not the meters, so it
        # exercises a path no other quality tool covers.
        ("profile_metadata_completeness",
         {"building_type": "Industrial",
          "fields": ["year_built", "floor_area", "occupants"]}),
        # The years entity reads the yearly aggregate rather than the
        # catalog tables the other entities use, so it is a separate path.
        ("search_catalog", {"entity": "years"}),
        ("find_buildings", {"country": "UK", "building_type": "Office", "limit": 5}),
        ("get_building_metadata", {"building_ids": ids[:3], "include_attributes": True}),
        ("get_data_availability", {"building_ids": ids[:3]}),
        ("check_task_eligibility", {"building_ids": ids[:3], "task": "eui_analysis"}),
        ("fetch_electricity", {"building_ids": ids[:2], "start": "2017-07-01",
                               "end": "2017-07-08"}),
        ("fetch_weather", {"building_ids": ids[:2], "start": "2017-07-01",
                           "end": "2017-07-08"}),
        ("fetch_contextual_data", {"building_ids": ids[:2], "start": "2017-07-01",
                                   "end": "2017-07-08"}),
        ("profile_data_quality", {"building_ids": ids[:5]}),
        ("detect_data_issues", {"building_ids": ids[:5]}),
        ("clean_timeseries", {"handle": elec.handle, "drop_flags": ["ZERO_MASKED"]}),
        ("resample_timeseries", {"handle": elec.handle, "resolution": "daily"}),
        ("impute_missing_values", {"handle": elec.handle, "method": "linear"}),
        ("align_timeseries", {"handles": [elec.handle, elec.handle]}),
        ("convert_timezone", {"handle": elec.handle, "to": "utc"}),
        ("calculate_consumption_statistics", {"building_ids": ids[:5],
                                              "start": "2017-01-01", "end": "2018-01-01"}),
        ("calculate_demand_metrics", {"building_ids": ids[:5]}),
        ("calculate_load_profile", {"building_ids": ids[:5], "profile": "seasonal"}),
        ("compare_temporal_profiles", {"building_ids": ids[:5],
                                       "comparison": "weekday_weekend"}),
        ("calculate_baseload", {"building_ids": ids[:5], "method": "night_mean"}),
        ("analyze_operating_pattern", {"building_ids": ids[:5]}),
        ("analyze_consumption_trend", {"building_ids": ids[:5], "grain": "monthly"}),
        ("calculate_normalized_performance", {"building_ids": peer_ids, "metric": "eui",
                                              "year": 2017}),
        ("construct_peer_group", {"building_id": one,
                                  "match_on": ["building_type", "country"]}),
        ("benchmark_building", {"building_id": one, "peer_ids": peer_ids,
                                "metric": "eui", "year": 2017}),
        ("compare_building_groups", {"group_a": peer_ids[:10], "group_b": peer_ids[10:20],
                                     "metric": "eui", "year": 2017}),
        ("rank_buildings", {"building_ids": peer_ids, "metric": "eui", "year": 2017,
                            "exclude_outliers_above": 500}),
        ("identify_peer_outliers", {"building_ids": peer_ids, "metric": "eui",
                                    "year": 2017}),
        ("calculate_weather_statistics", {"site_ids": ["Bobcat"],
                                          "variables": ["air_temperature",
                                                        "wind_speed"],
                                          "start": "2017-01-01",
                                          "end": "2018-01-01"}),
        ("analyze_weather_relationship", {"building_ids": ids[:5],
                                          "method": "regression"}),
        ("fit_temperature_response", {"building_ids": ids[:5], "model": "cooling"}),
        ("weather_normalize_consumption", {"building_ids": ids[:5],
                                           "baseline_start": "2016-01-01",
                                           "baseline_end": "2017-01-01",
                                           "target_start": "2017-06-01",
                                           "target_end": "2017-09-01"}),
        ("aggregate_geographically", {"level": "city", "metric": "eui",
                                      "building_type": "Office", "year": 2017}),
        ("compare_geographies", {"level": "city", "values": ["London", "Tempe"],
                                 "metric": "eui", "building_type": "Office",
                                 "year": 2017}),
        ("analyze_spatiotemporal_patterns", {"level": "city", "pattern": "peak_timing",
                                             "building_type": "Office"}),
        ("prepare_geospatial_analysis", {"level": "site", "metric": "eui",
                                         "year": 2017}),
        ("forecast_timeseries", {"building_id": one, "horizon_hours": 24,
                                 "cutoff": "2017-07-31", "model": "gradient_boosting"}),
        ("evaluate_forecast", {"forecast_handle": forecast.handle, "building_id": one}),
        ("detect_anomalies", {"building_ids": ids[:5], "method": "daily_profile"}),
        ("detect_change_points", {"building_ids": ids[:5]}),
        ("cluster_timeseries", {"building_ids": peer_ids, "n_clusters": 3}),
        ("find_similar_timeseries", {"building_id": one, "candidate_ids": peer_ids,
                                     "k": 5}),
        ("generate_timeseries_representations", {"building_ids": ids[:5],
                                                 "kind": "features"}),
        ("analyze_ai_model_results", {"results": [{"model": "a", "mae": 1.0},
                                                  {"model": "b", "mae": 2.0}]}),
        ("combine_evidence", {"sources": [perf.handle or "", profile.handle or ""],
                              "labels": ["perf", "profile"]}),
        ("create_visualization", {"handle": profile.handle or "",
                                  "kind": "load_profile", "group": "building_id",
                                  "title": "smoke"}),
        ("generate_report", {"question": "smoke", "answer": "ok",
                             "evidence": {"n": 1}}),
        # knowledge
        ("query_knowledge_graph",
         {"sparql": "SELECT (COUNT(?b) AS ?n) WHERE { ?b a ekg:Building }"}),
        ("traverse_relations", {"entity": one, "relation": "atSite"}),
        ("lookup_derived_facts", {"finding_type": "ChangePoint"}),
        ("assert_findings", {"method": "smoke",
                             "findings": [{"type": "ChangePoint",
                                           "building_id": one, "value": 1.0}]}),
        ("find_entity_paths", {"source": one, "target": "UK"}),
        ("export_semantic_model", {"target": "brick", "building_ids": [one]}),
        ("inspect_dataset", {"path": "BDG2"}),
        ("map_to_energyx_schema", {"path": "BDG2",
                                   "manifest_path": "datasets/bdg2/energyx.yaml"}),
        ("validate_dataset", {"path": "BDG2",
                              "manifest_path": "datasets/bdg2/energyx.yaml"}),
        ("register_dataset", {"path": "BDG2",
                              "manifest_path": "datasets/bdg2/energyx.yaml"}),
    ]

    results, failures = [], 0
    for name, kwargs in plan:
        res = call(name, ctx, **kwargs)
        bad = res.status is Status.ERROR
        failures += bad
        detail = res.message or res.predicate or ""
        if res.handle:
            detail = f"handle={res.handle} rows={res.rows}"
        elif res.rows:
            detail = f"rows={res.rows}"
        results.append((res.status.value, name, str(detail)[:78], bad))

    pri = call("prioritize_entities", ctx, evidence_handle=perf.handle or "",
               criteria={"eui": 1.0}, limit=5)
    ds = call("generate_decision_support", ctx,
              priorities_handle=pri.handle or "", question="smoke")
    for res, name in ((pri, "prioritize_entities"), (ds, "generate_decision_support")):
        bad = res.status is Status.ERROR
        failures += bad
        results.append((res.status.value, name, (res.message or "ok")[:78], bad))

    width = max(len(n) for _, n, _, _ in results)
    print()
    for status, name, detail, bad in results:
        mark = "!!" if bad else "  "
        print(f" {mark} {status:<18} {name:<{width}}  {detail}")

    called = {n for _, n, _, _ in results}
    uncalled = sorted(set(registry.REGISTRY) - called)
    print(f"\n  registered: {len(registry.REGISTRY)}   exercised: {len(called)}")
    if uncalled:
        print(f"  NOT exercised: {uncalled}")
    print(f"  hard failures (Status.ERROR): {failures}")
    print(f"  artifact store: {store.bytes_used/1e6:.1f} MB in {store.run_id}")
    return 1 if failures or uncalled else 0


if __name__ == "__main__":
    raise SystemExit(main())
