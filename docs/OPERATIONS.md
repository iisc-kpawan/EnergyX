# EnergyX — Operations and Maintenance Manual

Snapshot `EnergyX-BDG2-v1` · 73 tools · 12 agents · decisions …
Figures measured 28 September 2026 from the running service and from DuckDB directly.

Everything needed to operate this system without reading its source: what is on disk,
what each command does, how a question becomes an answer, how a new dataset enters, and
how to tell whether any of it still works.

---

## 1. What this is

A multi-agent system that answers natural-language questions about building electricity
data. Agents decide *what to do*; deterministic tools compute the numbers. Nothing a
model produces is trusted as a figure.

### Start it

```bash
# from the repository root
ENERGYX_LLM_PROVIDER=ollama ENERGYX_LLM_MODEL=qwen3.5-9b \
  ./.venv/bin/python scripts/serve.py
#   EnergyX -> http://127.0.0.1:8321
```

Reach it from a laptop — it binds loopback only:

```bash
ssh -L 8321:localhost:8321 <user>@<host>
# then open http://localhost:8321
```

> **The one rule that matters most.** The service has **no authentication**. It must stay
> on `127.0.0.1`. `serve.py` warns if you bind anything else; the warning is not
> decorative.

### Three ways in, one engine

| Interface | Use it for | Entry point |
|---|---|---|
| Chat page | People asking questions | `http://127.0.0.1:8321` |
| HTTP API | Scripts, notebooks, other services | `POST /ask` |
| Python | Analysis, batch runs, testing | `EnergyXAgent(mode="single")` |
| MCP | An external agent using the tools | `python -m energyx.mcp.server` |

---

## 2. Repository layout

Code and documents are tracked. Source data, the built snapshot and run artifacts are
not — they are rebuilt or regenerated.

```
EnergyX/
├── energyx/              the package
│   ├── config.py           paths, snapshot version, scope constants
│   ├── ecdm.py             canonical model: ids, vocabularies, 8 table schemas
│   ├── ingest/           source data -> snapshot
│   │   ├── bdg2.py          build the BDG2 snapshot
│   │   ├── byod.py          onboard any further dataset
│   │   ├── registry.py      load and validate dataset manifests
│   │   ├── aggregates.py    daily / monthly / yearly / profiles
│   │   └── tz.py            constant per-site UTC offsets
│   ├── access/           DuckDB session, views, the UTC invariant
│   ├── tools/            69 runtime + 4 onboarding tools
│   ├── agents/           intent, planning, orchestration, verification
│   ├── knowledge/        ontology, projector, store, curator, cache
│   ├── service/          FastAPI app + the chat page
│   └── mcp/              JSON-RPC server over stdio
├── scripts/              every operation you will run — see §10
├── datasets/             one folder per dataset, each with energyx.yaml
├── docs/
│   ├── OPERATIONS.md       this manual
│   ├── ARCHITECTURE.md     layers, agents, execution flow
│   ├── DATA_MODEL.md       ECDM, conventions, snapshot spec
│   ├── KNOWLEDGE_GRAPH.md  ontology, derived facts, standards export
│   ├── QUERY_ROUTING.md    how a question reaches a store
│   └── figures/            the architecture figure, PNG and PDF
├── tests/                unit tests for the structured layer
├── BDG2/                 source release — gitignored, obtained separately
├── data/                 the built snapshot — gitignored, rebuildable
└── runs/                 traces and spilled frames — gitignored
```

**Why data is not tracked.** The repository is a few megabytes against a working
directory of more than a gigabyte.
`BDG2/` is third-party, `data/` is derived from it, and `runs/` is disposable. Clone the
repo, place the BDG2 release, run the build in §9 — you get a byte-identical snapshot.

---

## 3. The snapshot on disk

One frozen directory, built once, never written to by a query. Everything the system
answers from lives here.

```
data/EnergyX-BDG2-v1/
├── MANIFEST.json     provenance: source dir, scope, counts, reconciliation
├── catalog/          8 Parquet tables — the metadata spine
│     datasets · sites · buildings · electricity_streams
│     capabilities · data_quality · building_attributes · weather_sources
├── electricity/      37 partitions
│   └── dataset=bdg2/site_id=<site>/year=<local year>/part.parquet
│         sorted by (building_id, timestamp_utc)
├── weather/          site-level hourly, same partitioning
├── aggregates/
│   ├── daily/      1,101,288 rows        ├── yearly/    3,095 rows
│   ├── monthly/    36,668 rows           └── profiles/  37,824 daily · 264,710 weekly
├── graph/            RDF quad store (Oxigraph / RocksDB)
└── vectors.parquet   load-shape representations
```

### What the catalog holds

| Table | Rows | Answers |
|---|---:|---|
| `datasets` | 1 | Name, sector, licence, citation, source |
| `sites` | 19 | City, country, timezone, constant UTC offset |
| `buildings` | 1,636 | Type, floor area, year built, occupants |
| `electricity_streams` | 1,578 | Unit, resolution, first/last reading, coverage |
| `capabilities` | 1,636 | Which analyses each building can support |
| `data_quality` | 1,578 | Gaps, duplicates, masked zeros, status |
| `building_attributes` | 4,281 | Entity–attribute–value, for anything non-core |
| `weather_sources` | 19 | Per-site weather channel completeness |

> **The invariant — never remove it.** Every timestamp is stored **UTC**; local calendar
> boundaries come from a constant per-site offset. `connect` pins
> `SET TimeZone='UTC'`. Without it the host's own timezone shifts every hour-of-day
> extraction and silently corrupts every aggregate; `verify_snapshot.py` catches it by
> asserting the yearly table's row count.

> **Two counts, both correct.** 1,636 buildings have metadata; 1,578 have an electricity
> meter. The 58-building gap is BDG2's own. Answers state which one they used — if a
> number looks wrong, check that first.

---

## 4. Running the service

| Variable | Default | Effect |
|---|---|---|
| `ENERGYX_LLM_PROVIDER` | `null` | `null` uses rule-based planning only. Set `ollama` for the real product. |
| `ENERGYX_LLM_MODEL` | — | `qwen3.5-9b` (fast) or `qwen3.5:27b` (slower, stronger) |
| `ENERGYX_WRITE_BACK` | `1` | Derived facts accumulate in the graph. Set `0` for a frozen run. |
| `OLLAMA_HOST` | `localhost:11434` | Point at a private Ollama pinned to an idle GPU |

### Endpoints

```
POST /ask                            {question, mode?} -> answer, payload, plan, evidence
GET  /health                         snapshot, counts, graph size, live planner
GET  /tools                          the 69 runtime tools, filterable by agent
GET  /catalog?entity=sites           datasets|countries|cities|sites|building_types
GET  /runs/{id}/trace                the full step trace for one run
GET  /runs/{id}/artifacts/{h}.csv    a result table in full
GET  /runs/{id}/artifacts/{h}.html   the same table, sortable, in a page
GET  /runs/{id}/figures/{name}.png   a rendered chart
```

Check it is alive, and which planner is *really* running:

```bash
curl -s http://127.0.0.1:8321/health | python -m json.tool
#   "llm": {"provider": "ollama", "model": "qwen3.5-9b", "planner": "llm"}
```

**The worker restarts itself.** `serve.py` runs uvicorn as a child and restarts it if it
dies from a signal — a native crash in a C extension cannot be caught in Python. Capped
at five restarts in ten minutes, so a genuinely broken server fails loudly instead of
looping. `--no-supervisor` disables it for debugging.

**One question at a time.** The DuckDB connection and the graph handle are process-wide
and not concurrency-safe, so requests are serialised behind a lock. Two people asking at
once queue. A portfolio-wide question takes one to two minutes and everyone else waits
behind it.

---

## 5. How a query is answered

Eight stages. Three of them can end the run before any tool touches data — that is
deliberate, and it is where most wrong answers were being produced.

1. **Is it a question?** *(input gate)* — Text carrying a status code, a predicate name
   or a stack trace is refused — such text parses as a query and would otherwise be
   answered as one.
2. **Intent** *(regex + catalog vocabulary)* — Entities resolved against the real
   catalog, plus period, metric, limit, the **subject** the answer must be about, the
   **shape** it must take, and any capability requirements the question states.
3. **Scope and capability gates** *(before planning, 0.0 s)* — Gas, water, steam, indoor
   conditions, cost and emissions are refused by scope. Sub-hourly resolution,
   sub-building metering, a second dataset, or a span longer than the snapshot holds are
   refused against *limits read from the catalog*, each naming the value that does exist.
4. **Planning** *(the model, or rules)* — The model sees one compact line per tool —
   names, types and *enums* — plus grounding stating which countries, cities, types and
   years exist. It emits a typed JSON plan. A mislabelled routing hint is repaired; an
   invented tool name earns one corrective retry with the errors quoted back.
5. **Validation** *(deterministic)* — Unknown tools, duplicate ids, missing dependencies,
   cycles, unresolvable references. Failure falls back to rule-based planning **for that
   question only** — never for the agent.
6. **Orchestration** *(topological order)* — References resolved, arguments type-checked,
   a filter the question stated but the plan omitted re-applied, analyst-only arguments
   dropped. Each tool runs under a **deadline** and reports how much of its cohort it
   covered.
7. **Verification** *(10 checks)* — Nine ask whether the machinery worked. The tenth asks
   whether the answer is *about the question* — without it, "indoor air temperature" was
   answered with electricity and passed every other check.
8. **Composition** — One of `scalar · boolean · table · ranking · set · refusal`, with the
   count, the filters applied, any chart drawn, and a downloadable full table.

> **Where the time goes.** Over 50 questions planned by `qwen3.5-9b`, the median
> question spends **6.5 seconds** planning and **0.16 seconds** in tool execution.
> Reading 26 million rows is not the expensive part of this system; deciding what to
> read is. The exception is the heavy AI tools — a clustering or backtest step can
> run for minutes, which is what the wall-clock budget in §5 exists to bound.

---

## 6. The tools — what to use when

69 runtime tools across eight specialists, plus four onboarding tools. Every tool returns
a typed outcome and **never raises**.

| Specialist | Tools | Reach for it when |
|---|---:|---|
| `data_discovery` | 8 | Which buildings/sites/cities exist; fetching raw series |
| `ai_analytics` | 18 | Forecasting, anomalies, change points, clustering, similarity |
| `consumption` | 9 | Totals, demand, load profiles, baseload, trends, year-on-year |
| `data_quality` | 8 | Coverage, gaps, duplicates, resampling, imputation |
| `weather_spatial` | 8 | Weather values, sensitivity, normalisation, geography |
| `performance` | 7 | EUI, peer groups, benchmarking, ranking, outliers |
| `knowledge` | 6 | Graph queries, recorded findings, standards export |
| `reporting` | 5 | Charts, combined evidence, prioritisation, reports |

### Typed outcomes

| Status | Means | What to do |
|---|---|---|
| `OK` | Complete result | Use it |
| `PARTIAL` | Ran out of time; covered *n* of *m* | Narrow the cohort, or accept the sample — the counts are reported |
| `EMPTY` | Nothing matched the filters | For a count this is **0**; for a yes/no it is **No** |
| `INSUFFICIENT_DATA` | Refused, with a named predicate | Read the predicate — it says what is missing |
| `ERROR` | A bug, converted from an exception | Check the trace at `/runs/{id}/trace` |

> **If you add a tool.** Type every fixed-vocabulary parameter as a `Literal`. A
> `list[str]` shows the planner `param:[string]` and it will invent values, including
> values it reads out of the tool's own prose. Write the summary as the question the tool
> answers, not as prose about it, and let the type carry the vocabulary.

---

## 7. The knowledge graph

An RDF store that starts as a projection of the catalog and then accumulates what the
system works out.

**Two kinds of content:**

- **Projected triples** — buildings, sites, streams, types, countries, derived from the
  catalog. Regenerable at any time.
- **Derived facts** — performance scores, change points, anomaly events, asserted by
  tools as they compute. *Not regenerable*: they exist only because someone asked a
  question.

**Bi-temporal, and never overwritten.** `validFrom`/`validTo` say when a fact was true of
the world; `computedAt` says when the system worked it out. Recomputing does not
overwrite — the old fact gains a `supersededBy` link and stays.

```bash
curl -s http://127.0.0.1:8321/health | python -m json.tool
#   "graph_triples": 61568,
#   "derived_facts": {
#     "current_facts": {}, "superseded": 0, "relations": 0 }
```

`graph_triples` counts the projection from the catalog, which a rebuild always
reproduces. The `derived_facts` counters start empty and grow only as runs assert
findings, so they are zero on a freshly projected graph and on any instance run
with `ENERGYX_WRITE_BACK=0`.

> ### Rebuilding the graph destroys derived facts
> `build_graph.py` is **idempotent and destructive by design**: it deletes the store and
> re-projects from the catalog. That makes it the reset for anything a run asserted —
> which is what you want after a bad run, and emphatically not what you want otherwise.
>
> There is **no incremental mode**. If the accumulated facts matter, copy
> `data/EnergyX-BDG2-v1/graph/` before rebuilding.

**Honest scope.** Six of 73 tools are graph-backed, and catalog questions go to DuckDB,
not the graph. Its benefit is on *repeat analytical work* — a cache hit returns a
byte-identical answer without recomputing it — and on facts a later question reuses.
It does nothing for "how many sites are in this dataset".

---

## 8. Adding a dataset

A new dataset enters through a manifest, not through code. The canonical model absorbs it
and every existing tool queries it without knowing.

### Step 1 — write the manifest

Create `datasets/<id>/energyx.yaml`. It declares how the files are organised; the files
themselves hold the records.

```yaml
dataset:
  id: myset
  name: ...          sector: commercial | residential
  source_url: ...    citation: ...    license: ...

timestamps:
  convention: local_standard_time     # or utc
  dst_observed: false
  offset_source: timezone_column      # or utc_offset_hours: -5
  stored_as: utc                      # always

electricity:
  path: readings.parquet
  layout: wide | long | wide_appliance
  timestamp: timestamp
  unit: kWh | W | W_min               # converted on ingest
  resolution_minutes: 60
  stream_level: whole_building | appliance
```

### Step 2 — dry run, then register

```bash
./.venv/bin/python -c "
from energyx.ingest.registry import load, validate
m = load('datasets/myset/energyx.yaml')
print(validate(m, 'datasets/myset'))"        # [] means it is valid

./.venv/bin/python -c "
from energyx.ingest.byod import register
print(register(source_dir='/path/to/files',
               manifest_path='datasets/myset/energyx.yaml'))"
```

### Step 3 — rebuild what depends on the catalog

```bash
./.venv/bin/python -c "
from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"

./.venv/bin/python scripts/build_graph.py     # see the warning in §7
./.venv/bin/python scripts/verify_snapshot.py
```

**Registration appends; it does not replace.** New rows are appended to `datasets`,
`sites`, `buildings`, `electricity_streams` and the quality tables. Measurements land in
their own `dataset=<id>` partition. BDG2 is untouched — `verify_snapshot.py` asserts its
count is still 1,636.

> **Record where it came from.** `MANIFEST.json` currently carries `source_dir` for the
> original build only. A dataset registered without its provenance recorded cannot be
> reproduced on another machine, and there is no way to recover it afterwards. Write the
> source path and a checksum into the manifest folder as you go.

**What changes automatically, and what does not.**

- *Automatic:* every catalog answer, every tool, the capability predicates, the planner's
  grounding vocabulary.
- *Manual:* aggregates, the graph projection, and the capability gate in §5 — its limits
  are read from the catalog at query time, so a finer-resolution dataset stops the
  refusals on its own, but confirm it with `scripts/oracle_check.py`.

---

## 9. Rebuilding from source

The whole snapshot regenerates from `BDG2/` in about two minutes. Do this after changing
ingestion, or on a new machine. **Stop the service first — it holds the graph lock.**

```bash
# 1 · measurements + catalog       ~55 s
./.venv/bin/python -c "
from energyx.ingest.bdg2 import build_snapshot; print(build_snapshot)"

# 2 · aggregates                    ~26 s
./.venv/bin/python -c "
from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"

# 3 · graph projection + vectors    ~40 s
./.venv/bin/python scripts/build_graph.py

# 4 · prove it
./.venv/bin/python scripts/verify_snapshot.py
#   58/58 checks passed, 2 not applicable to this snapshot
```

**Reading the result.** `reconciliation_exact: True` in `MANIFEST.json` means every source
reading is accounted for. The two "not applicable" checks are federation and
appliance-level tests that a single whole-building dataset cannot exercise — skipped, not
failed.

---

## 10. Testing — which harness, and when

Six harnesses. They answer different questions and cost different amounts of time;
running the wrong one wastes an hour.

| Run | Answers | Cost | When |
|---|---|---:|---|
| `verify_snapshot.py` | Is the data itself sound? | ~20 s | After any rebuild |
| `smoke_tools.py` | Does every tool still run? | ~3 min | After touching any tool |
| `demo_agent.py` | Does the agent loop hold together? | ~1 min | After touching agents |
| `oracle_check.py` | Are catalog answers right, against SQL? | ~2 min | After intent or planning changes |

```bash
# the standard post-change sweep
./.venv/bin/python scripts/demo_agent.py             # the agent loop, no model
./.venv/bin/python scripts/oracle_check.py           # graded against SQL
```

> ### The rule that keeps this honest
> A fix that passes the questions you were looking at is not yet a fix to the
> system. `oracle_check.py` draws its buildings, sites and periods from the
> catalog and computes every expected answer independently in SQL, so it cannot
> be satisfied by tuning the answers to a list you already have.

> **Test through the service, not only the library.** The rule-based planner and the model
> planner take different paths, so a fix verified against the library has not been
> verified in the browser, where the model plans. If a change affects answers, check it with
> `curl -X POST http://127.0.0.1:8321/ask`.

---

## 11. Proving correctness

Two harnesses generate their own questions and their own answers, so neither can
be satisfied by tuning against a fixed list.

**`oracle_check.py`** asks catalog questions whose truth it computes in SQL first,
then compares the agent's answer to it. Counts, memberships and rankings only --
the class of question where "close" is wrong.

Its questions are built from buildings, sites and periods taken from the
catalog rather than from a fixed list, which is what distinguishes a fix to the
system from a fix to one question.

```bash
./.venv/bin/python scripts/oracle_check.py
```

It prints, per question, the expected value, the returned value, the tool that
produced it and whether the model's own plan was used or the deterministic
planner took over. **Read that last column before reading the score** -- a model
can answer well while being planned around, and a score alone hides it.

To evaluate against your own answer key, run questions through `POST /ask` and
compare `payload.value` (typed: scalar, boolean, table, ranking, set, refusal)
rather than the prose answer. The payload is the contract; the text is a
rendering of it.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A tool errors with `IO error: While lock file … graph/LOCK` | Two processes on the graph; RocksDB admits one writer | Stop the service before running a script that writes facts |
| Browser shows an old interface | Cached page | Hard reload. The page is served `no-store`, so this should not recur |
| A question hangs for minutes | Portfolio-wide analysis under the global lock | Expected — it returns `PARTIAL` at the deadline with its coverage stated |
| Answers stop using the model | The verifier rejected that plan | Check `metrics.planner`; the fallback is per question, never per agent |
| A count looks too small | A coverage floor or a `limit` | The answer states both — read the bracketed evidence line |
| The service died | Native crash in a C extension | The supervisor restarts it; check the log for `worker died … restarting` |

```bash
# diagnose any single answer
curl -s -X POST http://127.0.0.1:8321/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many buildings are in London?"}' | python -m json.tool

# then read the full trace, including why a planner fell back
curl -s http://127.0.0.1:8321/runs/<run_id>/trace | python -m json.tool
```

---

## 13. Known limits

Stated plainly so nobody rediscovers them under deadline.

### Data

- **Electricity is the only meter channel.** BDG2's gas, chilled water, steam, hot water,
  water, irrigation and solar channels — about 180 MB of cleaned series — sit in `BDG2/`
  unread, by an early scope decision worth revisiting. Site weather and building metadata
  are ingested as context and are fully queryable.
- **Hourly only, 2016–2017.** Questions needing finer resolution or a longer span are
  refused, with the real limit cited.
- **Whole-building only.** No sub-metering, so no end-use breakdown.

### System

- **A native crash in the graph layer is contained, not prevented.** A SPARQL filter
  naming several hundred buildings can crash the embedded store from native code, which
  Python cannot catch. That query shape is guarded and the worker restarts, but treat the
  containment as a guard rail rather than a fix.
- **One question at a time.** The global lock means a slow question blocks everyone.
- **Correctness is verified, not scored.** `oracle_check.py` proves the catalog
  and analytical paths against independently computed SQL truth. It is not a
  published accuracy figure, and should not be quoted as one.
- **Cloud adapters are untested.** OpenAI-compatible and Anthropic providers are wired
  and report unavailable without an API key; only the local path is exercised.

### Answer shapes still open

- "What years does this snapshot cover?" returns the dataset row, which carries first and
  last reading but not a stated span.

---

Every design decision, with the failure that prompted it and the alternative rejected, is
recorded as …in — read that before changing behaviour,
because most of the surprising code is load-bearing.
