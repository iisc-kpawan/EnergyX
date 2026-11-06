# Setup and reproduction

From a clean machine to a running application and a reproduced evaluation.

Every command here is copy-paste ready. Where a command depends on something
you choose — a path, a model — the choice is named before the command, not
buried in it. Nothing assumes knowledge of how the project was developed.

**Contents**

1. [Prerequisites](#1-prerequisites)
2. [Clone](#2-clone)
3. [Environment](#3-environment)
4. [Dependencies](#4-dependencies)
5. [Configuration](#5-configuration)
6. [Dataset](#6-dataset)
7. [Build the snapshot](#7-build-the-snapshot)
8. [The planner model](#8-the-planner-model)
9. [Start the application](#9-start-the-application)
10. [Verify the setup](#10-verify-the-setup)
11. [Run the tests](#11-run-the-tests)
12. [Run the evaluation](#12-run-the-evaluation)
13. [Reading the results](#13-reading-the-results)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [Deployment architecture](#16-deployment-architecture)
17. [Future work](#17-future-work)

---

## 1. Prerequisites

| | Minimum | Notes |
|---|---|---|
| OS | Linux, macOS, or Windows via WSL2 | developed on Ubuntu 22.04 |
| Python | 3.11 or newer | 3.13 is what the reported runs used |
| RAM | 16 GB | |
| Disk | ~5 GB | source release, snapshot and environment |
| GPU | none required | only accelerates the planner model |

A GPU is optional. Section 8 gives three ways to plan, one of which needs no
local model at all. The evaluation *scoring* needs neither GPU nor dataset.

Check what you have:

```bash
python3 --version        # need 3.11+
free -g                  # Linux; `vm_stat` on macOS
df -h .
```

---

## 2. Clone

```bash
git clone <anonymised-repository-url>
cd EnergyX
```

Every path from here is relative to this directory.

---

## 3. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

The commands below write `./.venv/bin/python` explicitly, so they work whether
or not the environment is activated.

---

## 4. Dependencies

```bash
./.venv/bin/pip install -r requirements/base.txt
```

That is enough to start the application and answer most questions.

Two optional additions:

```bash
# the heavier forecasting and clustering models (~2 GB, includes torch)
./.venv/bin/pip install -r requirements/models.txt

# the Gemini planner back-end; not needed for a local model
./.venv/bin/pip install -r requirements/llm.txt
```

On a machine with no GPU, install the CPU build of torch first so pip does not
pull CUDA wheels:

```bash
./.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

`requirements/lock.txt` is the full package closure of the environment the
reported evaluation ran in. Use it to reproduce a result exactly; `base.txt` is
the better starting point otherwise.

---

## 5. Configuration

There are no configuration files to edit and no secrets to place. Everything is
an environment variable, and every one has a working default.

| Variable | Default | What it does |
|---|---|---|
| `ENERGYX_LLM_PROVIDER` | none | `ollama` or `gemini`; unset means the rule-based planner |
| `ENERGYX_LLM_MODEL` | none | model name, e.g. `qwen3.5-9b` |
| `ENERGYX_MODELS` | none | comma-separated models a visitor may choose between; a planner dropdown appears when two or more are named |
| `ENERGYX_NARRATE` | `0` | add a plain-English reading of each answer, written by the model from the completed result. The answer itself, the payload and the evidence are unchanged, and the evaluation never requests one |
| `ENERGYX_REQUIRE_LLM` | `1` | refuse to answer if the model is unreachable, rather than silently falling back |
| `ENERGYX_MAX_SECONDS` | `7200` | wall-clock budget for one question |
| `ENERGYX_MAX_STEPS` | `20` | plan steps allowed for one question |
| `ENERGYX_WRITE_BACK` | `1` | whether a run may assert findings into the knowledge graph |
| `ENERGYX_ROOT` | the repository | where `data/` is looked for |
| `OLLAMA_HOST` | `127.0.0.1:11434` | where the local model server listens |
| `GEMINI_API_KEY` | none | required only for the Gemini back-end |

`ENERGYX_REQUIRE_LLM=1` is worth setting deliberately. Left off, a model that
is unreachable causes a silent fall back to the rule-based planner, and the run
reports a success rate for a question the model was never asked.

**Secrets are never committed.** `api_key.txt` and `*.key` are in
`.gitignore`. If you use the Gemini back-end, export the key in your shell or
put it in a file you do not commit.

---

## 6. Dataset

The dataset is third-party and is not redistributed here. Obtain the Building
Data Genome Project 2 release from its authors:

**https://github.com/buds-lab/building-data-genome-project-2**

Place four files in a directory named `BDG2/` at the repository root:

```
BDG2/
├── electricity_cleaned.csv
├── electricity.csv
├── BDG2_metadata.csv
└── weather.csv
```

The release is about 713 MB in total; these four are what EnergyX reads.
Nothing is downloaded at build or run time — if `BDG2/` is missing, the build
stops and says so.

See [`EnergyX-Data/README.md`](../EnergyX-Data/README.md) for the full dataset
description.

---

## 7. Build the snapshot

EnergyX reads a frozen, versioned snapshot rather than the source CSVs. Build
it once.

```bash
# measurements, catalog tables and MANIFEST.json  (a few minutes)
./.venv/bin/python -c "from energyx.ingest.bdg2 import build_snapshot; print(build_snapshot)"

# daily, monthly and yearly rollups, and typical load profiles
./.venv/bin/python -c "from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"

# project the catalog into the RDF store
./.venv/bin/python scripts/build_graph.py

# acceptance checks
./.venv/bin/python scripts/verify_snapshot.py
```

`verify_snapshot.py` is the one that matters: it fails loudly rather than
reporting a partial snapshot as a whole one. Expect it to confirm 1,636
buildings, 1,578 electricity streams, 19 sites and 26,372,337 measurements.

---

## 8. The planner model

Three options. Pick one.

### a. A local model through Ollama — what the reported runs used

Install Ollama from https://ollama.com, then:

```bash
ollama pull qwen3.5-9b
CUDA_VISIBLE_DEVICES=0 OLLAMA_KEEP_ALIVE=24h ollama serve
```

In another shell:

```bash
export ENERGYX_LLM_PROVIDER=ollama
export ENERGYX_LLM_MODEL=qwen3.5-9b
```

If something else already holds port 11434 — a shared machine, most often —
give both sides the same alternative:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve      # in the server's shell
export OLLAMA_HOST=127.0.0.1:11435            # in the application's shell
```

### b. A hosted model — no GPU needed

```bash
./.venv/bin/pip install -r requirements/llm.txt
export ENERGYX_LLM_PROVIDER=gemini
export ENERGYX_LLM_MODEL=gemini-3.8-flash
export GEMINI_API_KEY=...                     # your own key
```

### c. No model at all

Leave `ENERGYX_LLM_PROVIDER` unset. The rule-based planner answers a useful
subset. This is a way to check the data path works, not a way to reproduce the
reported figures.

---

## 9. Start the application

```bash
export ENERGYX_REQUIRE_LLM=1
./.venv/bin/python scripts/serve.py --port 8321
```

| flag | default | |
|---|---|---|
| `--host` | `127.0.0.1` | loopback only |
| `--port` | `8321` | |
| `--reload` | off | reload on source change |

The service binds loopback and carries **no authentication**. Putting it on a
routable interface exposes an endpoint that will answer anything, so do that
only behind something that authenticates.

Open http://127.0.0.1:8321 in a browser, or use the API:

```bash
curl -s -X POST http://127.0.0.1:8321/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many buildings are located in London?"}' | python -m json.tool
```

---

## 10. Verify the setup

```bash
curl -s http://127.0.0.1:8321/health | python -m json.tool
```

A correct setup reports:

```json
{
  "status": "ok",
  "snapshot": "EnergyX-BDG2-v1",
  "catalog": {"datasets": 1, "buildings": 1636, "streams": 1578, "sites": 19},
  "graph_triples": 61568,
  "tools": {"runtime": 69, "onboarding": 4},
  "llm": {"provider": "ollama", "model": "qwen3.5-9b",
          "available": true, "planner": "llm"}
}
```

Three things to check in it:

- `catalog` matches the counts in section 7 — if not, the snapshot is partial.
- `llm.available` is `true` and `llm.planner` is `"llm"`. If it says `"rules"`,
  the model is unreachable and every answer will come from the rule-based
  planner.
- `graph_triples` is non-zero — otherwise `build_graph.py` did not run.

Then ask one question end to end:

```bash
./.venv/bin/python EnergyX-Eval/scripts/run_evaluation.py --ids Q01
```

Expect `Q01 ok` and `1/1 correct`.

---

## 11. Run the tests

```bash
./.venv/bin/pip install pytest
./.venv/bin/python -m pytest tests/ -q
```

With a snapshot built, expect **5 passed, 2 xfailed**. The two expected
failures are the structured planning engine, which is unfinished; `legacy` is
the default and is what every reported figure was measured on.

Before the snapshot is built the same command reports **2 passed, 5 skipped** —
the skipped ones need data and say so.

## 12. Run the evaluation

```bash
./.venv/bin/python EnergyX-Eval/scripts/run_evaluation.py --service http://127.0.0.1:8321
```

This asks all 50 questions in order, scores each answer against its golden
answer, and writes two files into `EnergyX-Eval/results/`. Allow 8–20 minutes
depending on the planner model.

To score answers collected elsewhere — if the application runs somewhere the
script cannot reach — put them in a CSV with `id` and `answer` columns:

```bash
./.venv/bin/python EnergyX-Eval/scripts/run_evaluation.py --answers my_answers.csv
```

That path needs no GPU, no model and no dataset.

---

## 13. Reading the results

`evaluation_summary.md` gives overall success and a breakdown by difficulty
level. `evaluation_per_question.csv` has one row per question, with the
grader's verdict, why, the timing, the golden answer and the application's
answer side by side.

Success is the count of correct answers over the number asked. There is no
partial credit, and a question the application fails to answer counts as
incorrect rather than being dropped.

For comparison, five planner models on this same set scored between 25/50 and
38/50 — see `EnergyX-Eval/results/five_model_summary.md`.

[`EnergyX-Eval/README.md`](../EnergyX-Eval/README.md) explains the tolerance
rules the grader applies and why each exists.

---

## 14. Troubleshooting

**`llm.planner` says `"rules"`, or answers look shallow.**
The model is unreachable. Check the model server is running and on the port
both sides agree on: `curl -s http://127.0.0.1:11434/api/tags`. Set
`ENERGYX_REQUIRE_LLM=1` so this fails loudly instead of quietly.

**The service was started before the model server.**
Availability is read at startup. Start the model server first, then restart the
service.

**`Address already in use` on 8321.**
Something else holds the port. `ss -ltnp | grep 8321` names it; either stop it
or pass `--port` a different number.

**The build stops saying `BDG2/` is missing.**
Section 6 — the four source files must be in place first. Nothing downloads
them for you.

**`verify_snapshot.py` reports wrong counts.**
The snapshot is partial. Rebuild from section 7; a half-written snapshot is not
repaired by running later steps.

**Every hour-of-day figure looks shifted.**
The DuckDB session is not in UTC. The build sets `SET TimeZone='UTC'`; without
it the host's own timezone corrupts every aggregate.
`verify_snapshot.py` catches this.

**`ModuleNotFoundError` for torch, sktime, lightgbm or chronos.**
Those are in `requirements/models.txt`, which is optional. Install it, or avoid
the questions that need those tools — the affected tools report a dependency
error rather than failing silently.

**Out of memory loading the model.**
Use a smaller or quantised planner model, or switch to the hosted back-end in
section 8b. The analytical tools themselves are pandas and DuckDB and are not
the memory constraint.

**A question times out.**
The default budget is generous, but the heavy clustering and backtesting tools
can run for minutes. Raise `ENERGYX_MAX_SECONDS`, or accept the bound — it
exists so a run always terminates.

---

## 15. Known limitations

- No answer is guaranteed correct. The published evaluation is the measure:
  between 25 and 38 of 50 depending on the planner model.
- Each question is answered independently. There is no conversation memory, so
  a follow-up such as *"and for 2016?"* will not resolve.
- Electricity is the only meter channel, 2016–2017, hourly; site weather and
  building metadata are carried as context. Questions outside that are refused
  with the missing predicate named rather than answered approximately.
- One question at a time. Simultaneous callers queue.
- Two building counts are both correct: 1,636 have metadata, 1,578 have a
  meter. Answers state which they used.
- 400 buildings carry no geography and are excluded from spatial questions by
  an explicit predicate. Six streams flagged as having electricity carry no
  data and are excluded by name.

## 16. Deployment architecture

What the demonstration instance runs on, what a reviewer needs instead, the
ports and endpoints, and the resource constraints are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). The short version: it is one
workstation behind an HTTPS tunnel, and reproducing it needs far less — 16 GB
and no GPU.

## 17. Future work

A refactor onto ADK is planned and not started. The implementation here is the
working baseline and is what every reported figure was measured on; the
migration may change the code structure, the APIs, the configuration and how
agents are implemented and executed. Nothing in this repository depends on it.
