# EnergyX — Architecture

Five layers, two cross-cutting planes. Layer 4 carries three co-equal stores behind
one router; the knowledge store is written back to by Layer 3 and read by Layer 2.

Governing principle throughout:
> **Agents reason and decide what to do next. Deterministic tools compute the numbers.**

---

## 1. The layered view

```
┌──────────────────────────┐ ┌──────────────────────────────────────────────────────────┐
│ 1. USER, APPLICATIONS    │ │ Conversational interface · Dashboards · API/SDK ·        │
│    & WORKFLOWS           │ │ Notebooks · Reporting & evaluation workflows · BYOD      │
│                          │ │ Users: energy analysts · researchers · facility and      │
│                          │ │ energy managers · utilities · owners/operators           │
└──────────────────────────┘ └──────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────┐ ┌──────────────────────────────────────────────────────────┐
│ 2. AGENTIC ENERGY DATA   │ │ ┌──────────────────── CORE AGENTS ────────────────────┐  │
│    INTELLIGENCE          │ │ │ Interaction & Intent    Planning                    │  │
│                          │ │ │ Orchestration           Verification & QA           │  │
│                          │ │ └─────────────────────────────────────────────────────┘  │
│  mode = single | multi   │ │ ┌───────────────── SPECIALIST AGENTS ─────────────────┐  │
│                          │ │ │ Data Discovery & Access                             │  │
│                          │ │ │ Data Quality & Preprocessing                        │  │
│                          │ │ │ Consumption, Demand & Load Analytics                │  │
│                          │ │ │ Building Performance & Peer Benchmarking            │  │
│                          │ │ │ Weather, Climate & Spatio-Temporal Analytics        │  │
│                          │ │ │ Advanced AI Analytics                               │  │
│                          │ │ │ Decision Support & Reporting                        │  │
│                          │ │ │ Knowledge & Semantic Reasoning            ← new     │  │
│                          │ │ └─────────────────────────────────────────────────────┘  │
│                          │ │ ┌────────── SHARED STATE / MEMORY / CONTEXT ──────────┐  │
│                          │ │ │ user intent · conversation state · plan DAG ·       │  │
│                          │ │ │ handles to intermediate results · tool outputs ·    │  │
│                          │ │ │ evidence ledger · provenance · execution history    │  │
│                          │ │ └─────────────────────────────────────────────────────┘  │
└──────────────────────────┘ └──────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────┐ ┌──────────────────────────────────────────────────────────┐
│ 3. TOOLS & SERVICES      │ │ Data discovery & query      Data quality & preprocessing │
│                          │ │ Consumption & load          Performance & benchmarking   │
│                          │ │ Weather & climate           Spatial & spatio-temporal    │
│                          │ │ Forecasting & prediction    Anomaly & change detection   │
│                          │ │ Clustering & similarity     Representation / embeddings  │
│                          │ │ Knowledge & graph query     Decision support             │
│                          │ │ Visualization & reporting                                │
│                          │ │                                                          │
│                          │ │ TOOL REGISTRY · MODEL REGISTRY · statistical methods ·   │
│                          │ │ ML / TSFM inference · KNOWLEDGE CURATOR (write-back)     │
└──────────────────────────┘ └──────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────┐ ┌──────────────────────────────────────────────────────────┐
│ 4. SEMANTIC, KNOWLEDGE   │ │ ┌────────────── UNIFIED QUERY ROUTER ─────────────────┐  │
│    & DATA ACCESS         │ │ │ resolves each sub-query to the cheapest sufficient  │  │
│                          │ │ │ engine, given the strategy Planning assigned        │  │
│                          │ │ └──┬──────────────────┬──────────────────┬───────────┘  │
│                          │ │    ▼                  ▼                  ▼              │
│                          │ │ KNOWLEDGE LAYER   CATALOG LAYER    MEASUREMENT LAYER    │
│                          │ │ Energy KG (RDF)   datasets         DuckDB over          │
│                          │ │ Vector index      sites            partitioned Parquet  │
│                          │ │ Derived facts     buildings        electricity + weather│
│                          │ │ Provenance graph  streams          raw + aggregates     │
│                          │ │ Standards export  weather_sources                       │
│                          │ │      ▲            data_quality                          │
│                          │ │      │            capabilities                          │
│                          │ │      │                 │                                │
│                          │ │      │  generated from ┘                                │
│                          │ │      │                                                  │
│                          │ │      └──── write-back from Knowledge Curator (L3)       │
│                          │ │                                                          │
│                          │ │ ECDM canonical model · schema registry & field mappings ·│
│                          │ │ provenance & lineage · task/data capability metadata     │
└──────────────────────────┘ └──────────────────────────────────────────────────────────┘
                                                  ▲
                                                  │
┌──────────────────────────┐ ┌──────────────────────────────────────────────────────────┐
│ 5. DATA INGESTION &      │ │ Native EnergyX datasets    │  Bring Your Own Dataset     │
│    ONBOARDING (optional  │ │                                                          │
│    at runtime)           │ │ Inspect → manifest / schema map → validate →             │
│                          │ │ transform & partition → catalog & quality profiling →    │
│                          │ │ graph projection → register                              │
│                          │ │ Sources: CSV · Parquet · SQL/DB · API                    │
└──────────────────────────┘ └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
  │ CROSS-CUTTING: RUNTIME           │   │ CROSS-CUTTING: OBSERVABILITY     │
  │ Local LLMs · Cloud LLMs · Hybrid │   │ Run traces · plan artifacts ·    │
  │ Local/cloud ML · compute runtime │   │ evidence ledger · token & cost   │
  │ supports layers 2 and 3          │   │ accounting · replay              │
  └──────────────────────────────────┘   └──────────────────────────────────┘
```

### What each layer answers

| Layer | Question it answers |
|---|---|
| 1 | Who uses EnergyX, through which interface or workflow |
| 2 | What the user means, what should be done, which specialist, is the result trustworthy |
| 3 | What is the numerical result |
| 4 | What the data mean, what exists, what we already know, and where it lives |
| 5 | How new datasets enter EnergyX |

---

## 2. Execution flow

```
                    user question
                          │
                          ▼
            ┌─────────────────────────┐
            │ Interaction & Intent    │  entities · time range · metric
            └────────────┬────────────┘  geography · output format · ambiguity
                         ▼
            ┌─────────────────────────┐  emits a typed plan DAG; annotates each
            │ Planning                │  step with resolution strategy:
            └────────────┬────────────┘  graph_only | cached_fact | hybrid | compute
                         ▼
            ┌─────────────────────────┐
            │ Orchestration           │  executes the DAG, honours dependencies,
            └────────────┬────────────┘  writes every result to shared state
                         │
       ┌─────────────────┼─────────────────┬──────────────────┐
       ▼                 ▼                 ▼                  ▼
  Knowledge &       Data Discovery    Consumption /      Advanced AI
  Semantic          & Access          Weather /          Analytics
  Reasoning                           Benchmarking
       │                 │                 │                  │
       └─────────────────┴────────┬────────┴──────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │ Decision Support &      │  combines evidence,
                    │ Reporting               │  ranks, visualises
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐  coverage · units · timezone
                    │ Verification & QA       │  joins · aggregation · peer validity
                    └────────────┬────────────┘  claims vs evidence
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
               response                Knowledge Curator
                                       (write-back, if enabled)
```

Verification is **mandatory before any complex answer is returned**. It may reject
once and force a bounded retry.

---

## 3. Agents

### Core (4)

| Agent | Responsibility |
|---|---|
| **Interaction & Intent** | Resolve entities, time ranges, metrics, geography, requested output; surface ambiguity. Never touches measurements. |
| **Planning** | Convert the request into a typed plan DAG; assign each step an agent, a tool and a **resolution strategy**; consult capability metadata and the tool registry. |
| **Orchestration** | Execute the DAG, route to specialists, manage dependencies and intermediate handles. Selects and invokes; does not itself execute tool bodies. |
| **Verification & QA** | Validate coverage, units, timezone, joins, sample counts, aggregation, normalisation, peer-group validity; check every claim against collected evidence. |

### Specialist (8)

| Agent | Categories | Tools |
|---|---|---|
| Data Discovery & Access | 1 | 8 |
| Data Quality & Preprocessing | 2 | 8 |
| Consumption, Demand & Load Analytics | 3, 4 | 9 |
| Building Performance & Peer Benchmarking | 5, 7 | 7 |
| Weather, Climate & Spatio-Temporal | 6, 8 | 8 |
| Advanced AI Analytics | 9, 10, 11 | 18 |
| Decision Support & Reporting | 12 | 5 |
| **Knowledge & Semantic Reasoning** | cross-cutting | 6 |

Category 12 has **no dedicated reasoning agent**. It fans out across Performance
(EUI, baseload), Weather (normalised excess), Benchmarking (peer percentile) and AI
(persistent anomalies), then converges on Decision Support and the Verifier. That
fan-out is the multi-agent argument the paper rests on — and with the knowledge
layer warm, much of it becomes a single graph join instead.

### Mode toggle

```
mode = multi    Orchestration routes to specialists; each agent is shown
                only its own tool subset (8, 8, 9, 7, 8, 18, 5, 6)

mode = single   One agent is shown all 69 runtime tools; the plan DAG is
                still emitted and still scored
```

Same registry, same tools, same traces. The only difference is which subset of the
registry each agent sees — which is what makes the comparison clean.

---

## 4. Tools & services (Layer 3)

**69 runtime tools + 4 onboarding.** Consolidated by parameterising rather
than proliferating: `calculate_load_profile(profile=...)`
replaces four tools, `create_visualization(type=...)` replaces nine,
`forecast_timeseries(model=...)` replaces one-tool-per-model.

| Agent | Tools |
|---|---|
| Data Discovery & Access (8) | `check_task_eligibility` · `fetch_contextual_data` · `fetch_electricity` · `fetch_weather` · `find_buildings` · `get_building_metadata` · `get_data_availability` · `search_catalog` |
| Data Quality & Preprocessing (8) | `align_timeseries` · `clean_timeseries` · `convert_timezone` · `detect_data_issues` · `impute_missing_values` · `profile_data_quality` · `profile_metadata_completeness` · `resample_timeseries` |
| Consumption & Load Analytics (9) | `analyze_consumption_trend` · `analyze_operating_pattern` · `calculate_baseload` · `calculate_consumption_statistics` · `calculate_demand_metrics` · `calculate_load_profile` · `compare_temporal_profiles` · `count_step_changes` · `run_structured_query` |
| Performance & Peer Benchmarking (7) | `benchmark_building` · `calculate_normalized_performance` · `compare_building_groups` · `construct_peer_group` · `construct_performance_index` · `identify_peer_outliers` · `rank_buildings` |
| Weather, Climate & Spatio-Temporal (8) | `aggregate_geographically` · `analyze_spatiotemporal_patterns` · `analyze_weather_relationship` · `calculate_weather_statistics` · `compare_geographies` · `fit_temperature_response` · `prepare_geospatial_analysis` · `weather_normalize_consumption` |
| Advanced AI Analytics (18) | `analyze_ai_model_results` · `backtest_forecast_models` · `cluster_timeseries` · `compare_cluster_stability` · `compare_clustering_methods` · `compare_similarity_measures` · `correlate_group_profiles` · `detect_anomalies` · `detect_breakouts` · `detect_change_points` · `evaluate_forecast` · `explain_consumption` · `find_similar_timeseries` · `forecast_timeseries` · `generate_timeseries_representations` · `rank_profile_divergence` · `sweep_anomaly_sensitivity` · `validate_anomaly_detection` |
| Decision Support & Reporting (5) | `combine_evidence` · `create_visualization` · `generate_decision_support` · `generate_report` · `prioritize_entities` |
| **Knowledge & Semantic Reasoning** (6) | `assert_findings` · `export_semantic_model` · `find_entity_paths` · `lookup_derived_facts` · `query_knowledge_graph` · `traverse_relations` |

**69 runtime tools**, plus 4 onboarding = 73.
| **BYOD onboarding** (4, offline) | `inspect_dataset` · `map_to_energyx_schema` · `validate_dataset` · `register_dataset` |

### The handle pattern

Tools never exchange frames. A fetch returns a reference:

```json
{"status": "OK", "handle": "res_7f3a", "rows": 1541760,
 "schema": ["timestamp_utc", "building_id", "value", "quality_flag"],
 "preview": [["2017-07-01T00:00:00Z", "Fox_office_Ana", 412.6, "OK"]],
 "provenance": {"snapshot": "EnergyX-BDG2-v1", "tool": "fetch_contextual_data"}}
```

The frame lives in `runs/<run_id>/`; only this summary reaches the LLM. Without
this, a single fetch of 176 buildings × 8760 hours would blow the context window.

### Knowledge Curator

A **deterministic service, not an agent.** It validates, pins, deduplicates and
asserts findings into the knowledge layer after an analysis completes. It must not
reason — reasoning happens in Layer 2, and letting a write-back path reason is how
speculative conclusions become permanent facts.

---

## 5. Failure and budget

| Outcome | Meaning | Handling |
|---|---|---|
| `OK` | result produced | continue |
| `EMPTY` | valid query, no matching entities | routine — may be the correct answer |
| `PARTIAL` | some entities resolved, some did not | continue, record shortfall in evidence |
| `INSUFFICIENT_DATA` | eligibility predicate failed | replan, or refuse with the predicate named |
| `ERROR` | genuine fault | bounded retry, then abort with trace |

Budget caps on steps, tokens and wall-clock guarantee termination: an
unbounded run is a run that never reports.

**An honest refusal is a first-class answer.** *Status / error message* is a
valid output format, so for some requests refusing correctly, and naming the
failing predicate, is the right answer rather than a failure to produce one.

---

## 6. What is deliberately not here

- No knowledge graph is required at *ingestion* — the graph is derived
- No agent performs arithmetic that reaches an answer
- No unrestricted `run_sql(...)` or Python execution exposed to any agent
- No cross-session memory by default — one session never sees another's
- No causal claims: "associated with", never "caused by"

---

See also: `DATA_MODEL.md` · `KNOWLEDGE_GRAPH.md` · `QUERY_ROUTING.md`

