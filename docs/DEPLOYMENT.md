# Deployment

Where the demonstration instance runs, what it runs on, and what a reviewer
needs to run their own. These are two different things, and the difference
matters: the machine below is what we had, not what the system requires.

---

## 1. The current demonstration instance

A single workstation, reached through an HTTPS tunnel. There is no cloud
instance, no load balancer and no container orchestrator.

### Hardware

| | |
|---|---|
| CPU | Intel Core i9-13900K, 24 cores / 32 threads |
| RAM | 125 GB |
| GPU | 2 × NVIDIA GeForce RTX 4090, 24 GB each |
| GPU used by EnergyX | **GPU 0 only** |

The second GPU belongs to another project on the same machine and is left
alone. The planner is pinned to GPU 0 with `CUDA_VISIBLE_DEVICES=0`.

### Software

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0 |
| Python | 3.13.13, in a virtualenv at `.venv/` |
| NVIDIA driver | 595.91.07 |
| CUDA (driver / torch build) | 13.2 / 13.0 |
| Model runtime | Ollama, private instance on port **11435** |
| Planner model | `qwen3.5-9b` (default), with `qwen3.5:27b` selectable |

The Ollama instance is deliberately separate from the machine's shared one on
port 11434, which belongs to other users.

### Services and ports

| Port | Bound to | What |
|---|---|---|
| 8321 | `127.0.0.1` | the EnergyX FastAPI service and chat page |
| 11435 | `127.0.0.1` | the private Ollama instance the planner calls |
| 4040 | `127.0.0.1` | the tunnel agent's local inspector |

Nothing binds a public interface. The service is reachable from outside only
through the tunnel, which terminates TLS at the provider and forwards to
`127.0.0.1:8321`.

### Endpoints

| Route | Purpose |
|---|---|
| `POST /ask` | ask a question; returns answer, payload, plan, evidence |
| `GET /health` | snapshot, counts, graph size, live planner |
| `GET /tools` | the 69 runtime tools, filterable by agent |
| `GET /catalog?entity=sites` | datasets, countries, cities, sites, building types |
| `GET /runs/{id}/trace` | the full step trace for one run |
| `GET /runs/{id}/artifacts/{h}.csv` | a result table in full |

### External dependencies

Only two, and neither is required to run the system:

- **The HTTPS tunnel provider**, which exposes the local port. A reviewer
  running locally needs none of this.
- **The Gemini API**, used only when the planner back-end is set to Gemini.
  The default back-end is a local model and reaches no external service.

The dataset is obtained once from its publisher and read locally thereafter.
Nothing is fetched at run time.

---

## 2. Resource constraints

> **The demonstration instance serves one question at a time.**

This is a verified property of the deployment, not a caution. A single global
lock serialises question handling, so simultaneous visitors queue rather than
run concurrently. What that means in practice:

- The first question after an idle period takes about a minute while the model
  weights load. Subsequent questions take a few seconds to a few minutes
  depending on the analysis.
- A second visitor arriving mid-question waits for the first to finish.
- A question is bounded at **180 seconds** and **12 plan steps**. Beyond
  either, it is stopped and reported, rather than running unbounded.

Concurrent-user support has not been tested and is not claimed. The limit is
resource contention on one GPU, not a design ceiling.

Write-back is **disabled** on the demonstration instance
(`ENERGYX_WRITE_BACK=0`), so no visitor can assert facts into the knowledge
graph. The graph a visitor queries is the one the snapshot shipped.

---

## 3. What a reviewer actually needs

Substantially less than the machine above.

| | Minimum | Why |
|---|---|---|
| CPU | any x86-64, 4 cores | the analytical tools are the cheap part |
| RAM | 16 GB | the largest fetch holds a filtered slice, not the portfolio |
| Disk | ~5 GB | ~713 MB source release, ~130 MB snapshot, plus the environment |
| GPU | **none required** | a GPU only accelerates the planner model |
| OS | Linux, macOS, or WSL2 | nothing platform-specific |
| Python | 3.11 or newer | |

Three ways to run it, in descending order of hardware need:

1. **Local model on a GPU** — what we ran. A 9B planner fits comfortably in
   24 GB; smaller quantised models fit in 8 GB.
2. **Local model on CPU** — works, and is slower at planning. The tools
   themselves are unaffected, since they are pandas and DuckDB.
3. **A hosted model API** — no GPU at all. Set the planner back-end to Gemini
   and supply a key. Planning then costs an API call instead of local compute.

The evaluation scoring needs none of this: `EnergyX-Eval` can score a CSV of
answers collected anywhere, with no GPU, no model and no dataset.

---

## 4. Reproducing the deployment

Start the three pieces in this order — the service caches model availability at
startup, so a service started before Ollama will answer with the rule-based
planner instead of the model:

```bash
# 1. the model runtime
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_KEEP_ALIVE=24h \
  setsid nohup ollama serve > /tmp/ollama.log 2>&1 &

# 2. the application
export ENERGYX_LLM_PROVIDER=ollama ENERGYX_LLM_MODEL=qwen3.5-9b
export ENERGYX_MODELS=qwen3.5-9b,qwen3.5:27b   # offers the planner dropdown
export ENERGYX_NARRATE=1                       # plain-English reading per answer
export ENERGYX_WRITE_BACK=0        # visitors cannot assert into the graph
export ENERGYX_MAX_SECONDS=180     # one question cannot hold the lock forever
export ENERGYX_MAX_STEPS=12
setsid nohup ./.venv/bin/python scripts/serve.py --port 8321 > /tmp/energyx.log 2>&1 &

# 3. confirm
curl -s http://127.0.0.1:8321/health | python -m json.tool
```

`setsid` detaches each process from the terminal, so closing the session does
not stop them. None of the three restarts itself after a reboot; if that
matters, install them as systemd user units.

Exposing the service publicly is a separate, optional step and is not required
to evaluate anything.
