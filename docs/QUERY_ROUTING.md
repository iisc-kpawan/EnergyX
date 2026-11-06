# EnergyX — Query Routing

How one natural-language question is resolved across three stores, and who decides
what.

---

## 1. Two decisions, deliberately separated

| Decision | Where | Nature |
|---|---|---|
| **Strategy** — "do I need to compute at all?" | Layer 2, **Planning** | Reasoning, informed by capability metadata and fact freshness |
| **Engine** — "which store answers this sub-query?" | Layer 4, **Router** | Mechanical dispatch |

Conflating these buries the interesting decision inside the data layer.
Planning annotates every step of the plan DAG with a strategy; the Router then
picks the store. Planning is a decision worth scoring; the Router is not — it is
plumbing.

---

## 2. The four strategies

```
graph_only    Answerable by traversal over asserted structure.
              No measurement scan. Milliseconds.
              → "Which cities have more than 20 office buildings?"

cached_fact   A derived fact already exists, pinned to the current snapshot
              with a compatible method. Read, do not recompute.
              → "What is Building X's cooling change-point?"  (computed last week)

hybrid        Graph resolves the entity set and supplies prior facts;
              tools compute over raw measurements for the rest.
              → "Which offices in London used unusually more electricity
                 during hot weather than their peers?"

compute       Nothing cacheable. Straight to measurements.
              → "Forecast Building X's next 24 hours."
```

### How Planning chooses

```
for each step in the plan DAG:

    required = what this step needs
    ├─ structural / metadata only? ─────────────────► graph_only
    │
    ├─ a derived fact, and lookup_derived_facts finds it
    │  with matching snapshot + compatible method + acceptable age?
    │                                              ─► cached_fact
    │
    ├─ entity resolution or prior facts + new computation over measurements?
    │                                              ─► hybrid
    │
    └─ otherwise                                   ─► compute
```

Freshness is not a timestamp heuristic — a fact is usable when its
`snapshotVersion` matches the active snapshot and its `method`/`params` are
compatible with what the step asks for. Facts on a superseded snapshot are never
served.

---

## 3. Routing by task category

| Cat | Category | Dominant strategy | What the graph contributes |
|---|---|---|---|
| 1 | Data Discovery & Availability | **graph_only** | Entire answer — pure traversal |
| 2 | Data Quality & Preprocessing | **graph_only** | Coverage, missingness, quality status as node properties |
| 3 | Consumption & Demand Statistics | compute | Resolves the eligible building set |
| 4 | Temporal & Load-Profile | compute | Resolves set; caches inferred operating schedules |
| 5 | Building Performance & Normalization | hybrid | Caches EUI and normalised metrics as `PerformanceScore` |
| 6 | Weather & Climate | hybrid | Caches change-points and weather sensitivity |
| 7 | Comparative & Peer Benchmarking | **hybrid, graph-strong** | Peer groups are nodes; percentiles are weighted edges |
| 8 | Spatial & Spatio-Temporal | **graph_only / hybrid** | site → city → region → country traversal |
| 9 | Forecasting & Predictive | compute | Resolves set only — nothing cacheable |
| 10 | Anomaly & Change Detection | hybrid | Findings cached and queryable as `AnomalyEvent` |
| 11 | Clustering, Similarity & Representation | **graph-strongest** | Similarity *is* a weighted edge; clusters *are* membership |
| 12 | Advanced Inference & Decision Support | **graph-strongest** | Evidence combination is a join over cached facts |

Categories 11 and 12 are where the knowledge layer earns most of its keep. Category
12 was defined as combining EUI + baseload + peer percentile + weather-normalised
excess + persistent anomalies — five specialist outputs. Cold, that means running
four specialists over raw data. Warm, it is one query, with specialists invoked
only for what is missing.

---

## 4. Worked examples

### A · `graph_only` — Category 1

> "Which cities have more than 20 office buildings with at least one year of
> electricity data and weather?"

```
Planning:   one step, resolution = graph_only
Router:     Knowledge layer
```

```sparql
PREFIX ekg: <http://energyx.org/ns#>
SELECT ?city (COUNT(?b) AS ?n) WHERE {
  ?b  a ekg:Building ;
      ekg:hasBuildingType ekgd:buildingtype/Office ;
      ekg:atSite ?site ;
      ekg:canWeatherAnalysis true ;
      ekg:hasStream ?s .
  ?s  ekg:coverageFraction ?cov .
  ?site ekg:inCity ?city .
  FILTER(?cov >= 0.9)
} GROUP BY ?city HAVING (COUNT(?b) > 20)
```

**Zero measurement rows scanned.** Under the old design this joined the catalog
and still worked — but the graph version also composes with traversal the catalog
cannot express without a purpose-built view.

### B · `hybrid` — Category 7, the nine-call chain

> "Which office buildings in London used unusually more electricity during hot
> weather in July 2017 compared with similar offices?"

```
step 1  graph_only    find_buildings(city=London, type=Office)        → 26 candidates
step 2  graph_only    check_task_eligibility(task=weather_analysis)   → 16 eligible
step 3  cached_fact   lookup_derived_facts(peer_group)          → hit: PeerGroup_017
step 4  compute       fetch_contextual_data(...)                → handle res_7f3a
step 5  compute       weather_normalize_consumption(res_7f3a)   → handle res_9d12
step 6  compute       benchmark_building(res_9d12, PeerGroup_017)
step 7  compute       identify_peer_outliers(...)
step 8  compute       combine_evidence(...)
step 9  compute       create_visualization(type=bar)
        ────────────────────────────────────────────────────────
        write-back    assert_findings(peer percentiles, outlier flags)
```

Steps 1–3 touch no measurements. Step 3 is a cache hit that would otherwise have
been a full peer-construction pass over the portfolio. Step 4 is the only large
fetch, and it returns a *handle* — the rows themselves never enter the LLM
context.

### C · `cached_fact` then `compute` — Category 6

> "What is Building Fox_office_Ana's cooling change-point, and has it shifted
> between 2016 and 2017?"

```
step 1  cached_fact   lookup_derived_facts(ChangePoint, whole period)  → 18.2 °C
step 2  compute       fit_temperature_response(2016)   ─┐ not cached — the
step 3  compute       fit_temperature_response(2017)   ─┘ sub-periods are new
step 4  compute       compare + report
        write-back    two new ChangePoint findings, each with validFrom/validTo
```

The bi-temporal model matters here: three change-point facts coexist for one
building over different validity windows, and none contradicts the others.

### D · `compute` — Category 9

> "Forecast Building X's electricity demand for the next 24 hours."

Straight to measurements. Nothing about a forecast is reusable, and caching one
would be wrong.

### E · Honest refusal — Category 5

> "What is the per-occupant electricity consumption of Building Y?"

```
step 1  graph_only    check_task_eligibility(task=occupancy_normalisation)
                      → INSUFFICIENT_DATA: ekg:canOccupancyNormalise = false
```

EnergyX refuses and names the failing predicate: `occupants` is null for this
building. Only **230 of 1,636** buildings (14%) carry occupancy, so this is a
common and correct outcome: for such a question the correct answer *is* the
refusal.

---

## 5. Verification of graph answers

The graph is a projection, so wherever it overlaps the catalog the two must agree.
The Verifier spot-checks:

```
graph answer ──┐
               ├──► compare ──► agree?  ─► accept
catalog SQL ───┘                       └─► reject, fall back to catalog,
                                           flag projection defect
```

Because the catalog is authoritative, a disagreement is a **graph bug**,
never a catalog bug. This check is cheap and catches broken projections early,
which matters most right after re-ingestion.

NL→SPARQL is validated before execution: generation is constrained to the EKG
ontology, and any query referencing an unknown class or property is rejected
without running.

---

## 6. Cost model

Indicative, for the EnergyX-BDG2-v1 snapshot (1,578 streams × 17,544 hours ≈ 27.7M
cells, 26.4M of them stored):

| Strategy | Touches | Typical cost |
|---|---|---|
| `graph_only` | triples only | milliseconds |
| `cached_fact` | triples only | milliseconds |
| `hybrid` | triples + a filtered measurement slice | proportional to the *resolved* entity set, not the portfolio |
| `compute` | measurement scan | proportional to buildings × period |

The leverage is in step ordering: **resolve the entity set in the graph first, then
fetch.** Fetching the 16 eligible London offices rather than scanning all 1,578
streams is close to a 100× reduction before any analysis begins.

---

## 7. Reportable metrics this design enables

Because the plan DAG is a first-class artifact and strategy is annotated
per step:

- **Planning accuracy** — plan vs the analytical specification the question calls for
- **Agent-selection accuracy** — did the right specialist get the step
- **Tool-selection accuracy** — did the right tool get called
- **Strategy accuracy** — did the planner route to the graph when it could have
- **Graph-query accuracy** — NL→SPARQL correctness, comparable with BuildingQA
- **Knowledge saturation** — cost and latency per question as facts accumulate
- **Warm vs cold delta** — the value of the knowledge layer, isolated

The last two are only measurable because write-back exists and because the same
questions can be run both ways.

---

See also: `ARCHITECTURE.md` §2 execution flow · `KNOWLEDGE_GRAPH.md` for the
ontology
