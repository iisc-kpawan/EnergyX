# EnergyX

**A multi-agent framework for conversational energy data analytics**

EnergyX answers natural-language questions about real building electricity data by
coordinating specialist agents over deterministic analytical tools, a canonical data
model, and a derived knowledge graph.

> Agents reason and decide what to do next. Deterministic tools compute the numbers.

---

## What it does

Ask in English; get an answer with the number, the filters that produced it, the
rows behind it, and a chart when the question deserves one.

```
How many buildings are in London?
  -> 68 buildings  [dataset=bdg2, city=London]

Rank the ten most weather-sensitive offices in Arizona
  -> a ranked table + the fitted sensitivity for each

What is the indoor air temperature in building X?
  -> refused: this snapshot meters electricity; indoor conditions are not recorded
```

The last one matters as much as the first two. A question the data cannot support
returns a refusal that names the missing predicate, not a plausible number.

## Scope

Electricity consumption and demand from commercial and residential buildings, with
building metadata and weather as context. Gas, thermal, water and solar channels,
equipment diagnostics, comfort, control and simulation are out of scope and are
refused explicitly rather than approximated.

Current snapshot — **EnergyX-BDG2-v1**, the Building Data Genome Project 2:
1,636 non-residential buildings, of which **1,578 carry electricity**, at 19 sites
in 4 countries, hourly, **26,372,337 measurements** spanning 2016–2017, with
site-level weather and building metadata throughout.

Onboarding for further datasets exists (`energyx/ingest/byod.py` — three layouts,
unit harmonisation, manifest-driven), so the canonical model is not BDG2-shaped.

## Architecture

```
1. User, Applications & Workflows
2. Agentic Energy Data Intelligence   4 core + 8 specialist agents, shared memory
3. Tools & Services                   69 runtime + 4 onboarding tools
4. Semantic, Knowledge & Data Access  knowledge graph · catalog · measurements
5. Data Ingestion & Onboarding        native datasets + BYOD
   cross-cutting: runtime environment · observability
```

Layer 4 carries three stores behind one router. Planning decides the *strategy*
(`graph_only` / `cached_fact` / `hybrid` / `compute`); the router picks the *engine*.

Findings flow back: analyses assert pinned facts into the knowledge graph, so
repeated questions stop being recomputed and the system gets cheaper with use.

## Design commitments

- **No agent performs arithmetic that reaches an answer.** Models plan; tools compute.
- **No unrestricted SQL or Python is exposed to any agent.** Every tool is typed.
- **Tools never raise.** They return `OK · PARTIAL · EMPTY · INSUFFICIENT_DATA · ERROR`.
- **Refusal is a first-class answer**, and names the predicate that failed.
- **Every timestamp is stored UTC**; local calendar boundaries come from a constant
  per-site offset.
- **No causal claims** — "associated with", never "caused by".

## Quick start

Requires Python 3.11+ and the BDG2 release placed at `BDG2/`. A GPU is not
required; a local [Ollama](https://ollama.com) model is one of three ways to
plan, and the other two need no GPU at all.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements/base.txt

# build the snapshot from source (~2 minutes)
./.venv/bin/python -c "from energyx.ingest.bdg2 import build_snapshot; print(build_snapshot)"
./.venv/bin/python -c "from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"
./.venv/bin/python scripts/build_graph.py
./.venv/bin/python scripts/verify_snapshot.py

# run it
ENERGYX_LLM_PROVIDER=ollama ENERGYX_LLM_MODEL=qwen3.5-9b \
  ./.venv/bin/python scripts/serve.py
#   EnergyX -> http://127.0.0.1:8321
```

The service binds loopback only and has no authentication. Keep it that way, or
put a reverse proxy with auth in front of it.

**[`SETUP.md`](SETUP.md) is the full version**: prerequisites, every environment
variable with its default, dataset acquisition, how to verify the setup worked,
and troubleshooting for the failures that actually occur.

## Reproducing the evaluation

The reported figures come from 50 questions with an independently computed
answer for each. Both, and the grader that compares them, are in
[`EnergyX-Eval/`](EnergyX-Eval/).

```bash
./.venv/bin/python EnergyX-Eval/scripts/run_evaluation.py --service http://127.0.0.1:8321
```

Five planner models on this set scored between 25/50 and 38/50. The scoring
path needs no GPU, no model and no dataset: a CSV of answers collected anywhere
can be scored offline with `--answers`.

## Repository layout

```
energyx/            the application: agents, tools, knowledge, service, MCP
  agents/             intent, planning, orchestration, verification
  tools/              the 69 runtime tools, by specialist
  knowledge/          ontology, projector, store, curator
  service/            HTTP API and the chat page
  ingest/             building a snapshot from a source release
EnergyX-Eval/       the 50 questions, the answers, the grader, the results
EnergyX-Data/       where the dataset comes from and how to build it
EnergyX-Gold/       the question set's pinned definitions and reference
                      implementations, as released by their author
EnergyX-Max/        a generated extension: several thousand questions with
                      program-computed answers, not used in the evaluation
requirements/       dependencies, split by what you actually need
scripts/            build, serve and check entry points
datasets/           dataset manifests (schema registry entries)
docs/               architecture, data model, knowledge graph, routing, deployment
tests/              the test suite
```

## Status

| | |
|---|---|
| **Working** | The ten analytical categories end to end — planning, tool execution, verification, and a composed answer carrying its evidence. Best planner model: **38 of 50**. |
| **Partial** | Questions combining several measures into one judgement are the least reliable. Multi-agent mode is implemented but answers less correctly than single-agent on this model, and is kept for comparison. Heavy forecasting and clustering tools can run for minutes, bounded at 180 s and 12 plan steps. |
| **Not supported** | Conversational follow-up — each question is answered independently, with no history. Concurrent users. Meters other than electricity. Any period outside 2016–2017. |

Known limitations are listed on the application's landing page and in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Using it

| Interface | For | Entry point |
|---|---|---|
| Chat page | People asking questions | `http://127.0.0.1:8321` |
| HTTP API | Scripts, notebooks, other services | `POST /ask` |
| Python | Analysis and batch work | `EnergyXAgent(mode="single")` |
| MCP | An external agent using the tools | `python -m energyx.mcp.server` |

```bash
curl -s -X POST http://127.0.0.1:8321/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many sites are in this dataset?"}'
```

## Documentation

| Document | Covers |
|---|---|
| [`SETUP.md`](SETUP.md) | **Start here.** Clean machine to a reproduced evaluation |
| [`EnergyX-Eval/README.md`](EnergyX-Eval/README.md) | How answers are judged, and how to reproduce the scores |
| [`EnergyX-Data/README.md`](EnergyX-Data/README.md) | The dataset: obtaining it, building the snapshot |
| [`EnergyX-Gold/README.md`](EnergyX-Gold/README.md) | How each question is pinned, and its reference implementation |
| [`EnergyX-Max/README.md`](EnergyX-Max/README.md) | The generated question set and how it was produced |
| [`COMMANDS.md`](COMMANDS.md) | Every command, grouped by what you are doing |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | What the demonstration runs on, and what you need instead |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Every folder, command and failure mode |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The five layers, the agents, the execution flow |
| [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) | The canonical model, conventions, snapshot spec |
| [`docs/KNOWLEDGE_GRAPH.md`](docs/KNOWLEDGE_GRAPH.md) | Ontology, derived facts, standards export |
| [`docs/QUERY_ROUTING.md`](docs/QUERY_ROUTING.md) | How a question reaches a store |

## Data

The Building Data Genome Project 2 is third-party and obtained separately; it is
not redistributed here. Place the release at `BDG2/` and build. See
[`EnergyX-Data/README.md`](EnergyX-Data/README.md) for acquisition and
preparation, `datasets/bdg2/energyx.yaml` for the manifest, and
[`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) for provenance and licensing.

## Future work

**A planned refactor onto ADK.** The implementation in this repository is the
working baseline and is what every reported figure was measured on. A migration
to ADK is planned and not started; when it happens it may change the code
structure, the APIs, the configuration and how agents are implemented and
executed. Nothing in this repository depends on it, and no claim here should be
read as describing it.

## Licence

MIT — see [`LICENSE`](LICENSE). The licence covers the source code in this
repository. The BDG2 dataset is third-party and governed by its own terms at
the upstream repository; model weights obtained through Ollama or a hosted API
carry their own licences.
