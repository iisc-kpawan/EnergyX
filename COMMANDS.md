# Commands

Every command needed to build, run, query and check EnergyX, in the order you
would first need them. All of them assume the repository root as the working
directory and the virtualenv at `.venv`.

---

## 1. Install

```bash
python -m venv .venv
./.venv/bin/pip install -e .
```

---

## 2. Build a snapshot

EnergyX reads a frozen, versioned snapshot rather than the source files. Build
it once; everything afterwards reads it.

```bash
# Measurements and catalog, from the source release in BDG2/
./.venv/bin/python -c "from energyx.ingest.bdg2 import build_snapshot; print(build_snapshot)"

# Pre-computed daily, monthly and yearly rollups and typical load profiles
./.venv/bin/python -c "from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"

# Project the catalog into the RDF store
./.venv/bin/python scripts/build_graph.py

# Confirm counts, coverage and provenance reconcile
./.venv/bin/python scripts/verify_snapshot.py
```

`build_snapshot` writes `data/<snapshot>/` with the partitioned measurements,
the catalog tables and `MANIFEST.json`. `verify_snapshot.py` is the one to run
after any rebuild: it fails loudly rather than reporting a partial snapshot as
a whole one.

---

## 3. Run the service

```bash
ENERGYX_LLM_PROVIDER=ollama ENERGYX_LLM_MODEL=qwen3.5-9b \
  ./.venv/bin/python scripts/serve.py
```

| flag | default | |
|---|---|---|
| `--host` | `127.0.0.1` | loopback only unless you change it |
| `--port` | `8321` | |
| `--reload` | off | reload on source change, for development |
| `--no-supervisor` | off | run in the foreground without the restart wrapper |

The service binds loopback and carries no authentication. Putting it on a
routable interface exposes an unauthenticated endpoint that will answer
anything, so do that only behind something that authenticates.

### With a local model server

```bash
# Start the model server once, pinned to one GPU
CUDA_VISIBLE_DEVICES=0 OLLAMA_KEEP_ALIVE=24h ollama serve

# Point EnergyX at it
export ENERGYX_LLM_PROVIDER=ollama ENERGYX_LLM_MODEL=qwen3.5-9b
./.venv/bin/python scripts/serve.py
```

Ollama listens on `127.0.0.1:11434` by default and EnergyX looks there, so
neither side needs `OLLAMA_HOST` set. Set it on both when something else
already holds that port -- another user's server on a shared machine, most
often -- and give the two the same value:

```bash
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_KEEP_ALIVE=24h ollama serve
export OLLAMA_HOST=127.0.0.1:11435
```

Start the model server before the service. EnergyX resolves the provider once,
when an agent is first built, so a service started against a model server that
is not yet up keeps answering `503` until it is restarted.

---

## 4. Ask it something

```bash
# Health, and which planner each agent is actually holding
curl -s http://127.0.0.1:8321/health | python -m json.tool

# Ask a question
curl -s -X POST http://127.0.0.1:8321/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many buildings are located in London?"}' | python -m json.tool

# See how a question was read, before it runs
curl -s -X POST http://127.0.0.1:8321/interpret \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which office buildings had the highest EUI in 2017?"}' | python -m json.tool

# The full execution trace for a run: plan, tool steps, timings, tokens
curl -s http://127.0.0.1:8321/runs/<run_id>/trace | python -m json.tool
```

Every endpoint:

| method | path | |
|---|---|---|
| `GET` | `/` | the web client |
| `GET` | `/health` | snapshot version, tool counts, live planner per agent |
| `POST` | `/ask` | a question; returns the answer, plan, evidence and run id |
| `POST` | `/interpret` | how a question was read, without executing it |
| `GET` | `/tools` | the tool catalogue, optionally `?agent=<name>` |
| `GET` | `/catalog` | datasets, sites, buildings, coverage |
| `GET` | `/browse` | the entity tree |
| `GET` | `/runs/{run_id}/trace` | the execution trace |
| `GET` | `/runs/{run_id}/artifacts/{handle}.csv` | a result table as CSV |
| `GET` | `/runs/{run_id}/artifacts/{handle}.html` | the same, rendered |
| `GET` | `/runs/{run_id}/figures/{name}.png` | a generated figure |

---

## 5. Use the tools without the agent

The tool registry is reachable over MCP (JSON-RPC 2.0 on stdio), so another
agent, an IDE or a script can call the same tools without EnergyX's agent
layer.

```bash
# Run the server
./.venv/bin/python -m energyx.mcp.server

# Verify it end to end with an external client
./.venv/bin/python scripts/mcp_client_check.py
```

Or in-process:

```bash
./.venv/bin/python -c "
from energyx.access import SnapshotDB
from energyx.config import SNAPSHOT_VERSION
from energyx.tools import ArtifactStore, ToolContext, load_all, registry
load_all
ctx = ToolContext(db=SnapshotDB, store=ArtifactStore.for_run,
                  snapshot_version=SNAPSHOT_VERSION, options={})
result = registry.get('find_buildings').fn(ctx, city='London')
print(result.status, result.evidence)
"
```

---

## 6. Checks

```bash
./.venv/bin/python scripts/smoke_tools.py           # every tool is callable and typed
./.venv/bin/python scripts/demo_agent.py            # the agent layer end to end, no LLM
./.venv/bin/python scripts/oracle_check.py          # graded against SQL
./.venv/bin/python scripts/mcp_client_check.py      # the MCP surface
```

`demo_agent.py` needs no model: it exercises planning, execution, verification
and refusal with the deterministic planner, so it is the fastest way to tell
whether a change broke the agent layer, and the only one of these that runs on
a machine with no GPU.

`oracle_check.py` is the one to run if you want evidence rather than a smoke
test. It builds its questions from the catalog at runtime and pairs each with
the SQL that answers it, so nothing is graded against a stored key: the
service is asked a question, the database is asked the same question, and the
two are compared. It takes `--mode single|multi|both`, `--per-kind N`,
`--timeout N` and `--out PATH`, and needs the service already running.

---

## 7. Reaching a running instance

**A local instance** — clone, build the snapshot (§2), run it (§3). Everything
needed is in this repository apart from the source release and the model
weights.

---

## 8. Environment

| variable | default | |
|---|---|---|
| `ENERGYX_ROOT` | repository root | where `data/` and `runs/` live |
| `ENERGYX_LLM_PROVIDER` | — | `ollama`, `openai`, `gemini`, `anthropic` |
| `ENERGYX_LLM_MODEL` | — | the model name for that provider |
| `ENERGYX_REQUIRE_LLM` | `0` | `1` plans with the model or refuses, with no keyword fallback |
| `ENERGYX_PLANNING_ENGINE` | `legacy` | `legacy`, `structured` or `shadow` |
| `ENERGYX_MAX_STEPS` | tool default | step budget per question |
| `ENERGYX_MAX_SECONDS` | tool default | time budget per question |
| `ENERGYX_WRITE_BACK` | `0` | `1` lets a run assert derived findings into the graph |
| `ENERGYX_SCOPE_RULES` | — | override the capability gate's scope rules |
| `ENERGYX_BACKTEST_WORKERS` | CPU-derived | parallelism for multi-model forecasting |
| `OLLAMA_HOST` | `127.0.0.1:11434` | where the local model server listens |
| `OPENAI_BASE_URL` | provider default | for an OpenAI-compatible endpoint |
| `GEMINI_BASE_URL` | provider default | |

Provider credentials are read from the environment
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) and are never read
from a file in the repository.

`ENERGYX_REQUIRE_LLM=1` is worth setting deliberately. Left off, a model that
fails mid-question falls back to keyword routing and answers anyway, which is
indistinguishable in the result from a question the model answered well.
