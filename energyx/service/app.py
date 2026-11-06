"""HTTP service over the EnergyX agent (architecture Layer 1).

Everything below is a thin shell around :class:`EnergyXAgent`. No analysis lives
here: the service resolves a question to an agent call and returns the answer
with its plan, evidence and trace id, so that what a browser sees is exactly
what a script or the MCP server would get.

Two operational choices worth stating:

* **Agent calls are serialised.** The DuckDB connection and the Oxigraph handle
  are shared process-wide and are not safe for concurrent use. A lock is the
  honest fix for a service of this size; a pool would be premature.
* **Write-back is off by default.** A browsing user should not silently mutate
  the knowledge layer; it is opt-in per request.
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pandas as pd

from ..access import SnapshotDB
from ..agents import Budget, EnergyXAgent, PlannerRefused
from ..agents.runtime import apply_intent_overrides
from ..agents.llm import make_llm
from ..config import PATHS, SNAPSHOT_VERSION
from ..knowledge.curator import KnowledgeCurator
from ..knowledge.store import KnowledgeStore
from ..tools import ensure_loaded, registry
from ..tools.base import jsonify

STATIC = Path(__file__).parent / "static"

#: Origins allowed to call this service from a browser, comma separated. Empty
#: means same-origin only, which is what a local instance wants.
ALLOWED_ORIGINS = tuple(
    o.strip() for o in os.environ.get("ENERGYX_ALLOWED_ORIGINS", "").split(",")
    if o.strip())

app = FastAPI(title="EnergyX",
              description="Conversational analytics over building electricity data",
              version="0.1.0")

log = logging.getLogger("energyx.service")
_lock = threading.Lock()
_agents: dict[str, EnergyXAgent] = {}
_db: SnapshotDB | None = None


def _llm(model: str | None = None):
    """Configured from the environment so the service needs no code change."""
    provider = os.environ.get("ENERGYX_LLM_PROVIDER", "null")
    model = model or os.environ.get("ENERGYX_LLM_MODEL", "")
    return make_llm(provider, **({"model": model} if model else {}))


#: Models a caller may choose between. An allowlist, because `model` arrives
#: from the browser. Empty means no choice is offered.
SELECTABLE_MODELS = tuple(
    m.strip() for m in os.environ.get("ENERGYX_MODELS", "").split(",") if m.strip())

#: Whether an answer carries a plain-English reading by default.
NARRATE = os.environ.get("ENERGYX_NARRATE", "0") not in ("", "0", "false", "no")

NARRATION_SYSTEM = (
    "You restate a completed analysis in plain English for a reader who has "
    "the full result in front of them.\n"
    "Rules, all of them absolute:\n"
    "- Use only figures that appear in the result. Never compute, round, "
    "convert or infer a number.\n"
    "- Never say one thing caused another. Say 'associated with'.\n"
    "- If the result is a refusal or found nothing, say so plainly.\n"
    "- Write about the finding, not about the table: never name a column.\n"
    "- Two to four sentences. No headings, no bullets, no preamble.")


def _narrate(question: str, answer: str, payload: dict | None,
             model: str | None) -> str | None:
    """A plain-English reading of an answer that is already complete.

    Additive: it cannot change the answer, the payload or the evidence, and a
    failure returns nothing.
    """
    try:
        llm = _llm(model)
        if not getattr(llm, "available", False):
            return None
        value = (payload or {}).get("value")
        prompt = (f"Question: {question}\n\n"
                  f"Result:\n{answer[:4000]}\n\n"
                  + (f"Headline value: {value}\n" if value is not None else "")
                  + "\nRestate this for the reader.")
        reply = llm.complete(prompt, system=NARRATION_SYSTEM)
        text = (getattr(reply, "text", "") or "").strip()
        return text or None
    except Exception:                                          # noqa: BLE001
        return None


# Interactive use is exactly what the knowledge layer was built for: a derived
# finding computed once is reused by every later question. It was off here, so
# eleven fact-producing tools never wrote anything, the cache never hit, and the
# planner's "already computed" list was always empty. A measured run should
# still execute against a frozen read-only graph -- that is an independence
# requirement, and it is set per run rather than by this default.
WRITE_BACK = os.environ.get("ENERGYX_WRITE_BACK", "1") not in ("0", "false", "no")


# How long one question may run. Backtests and portfolio scans are legitimately
# long analyses -- a seasonal ARIMA fit is tens of seconds and a rolling
# backtest wants dozens of them -- so the ceiling is configurable rather than
# a constant someone has to edit the source to change. The default stays at
# three minutes because that is about as long as anyone will watch a browser.
# A budget exists to stop a runaway, not to decide what the answer is.
# At 180s a cohort backtest silently became a one-origin estimate, and
# the estimate read like the answer. Long enough that correctness is
# never the thing traded away; the origin-limited warning remains for
# the cases where even this is not enough.
MAX_SECONDS = float(os.environ.get("ENERGYX_MAX_SECONDS", "7200"))
MAX_STEPS = int(os.environ.get("ENERGYX_MAX_STEPS", "20"))
# Every plan comes from the model. The rule-based planner stays in the process
# as the thing the model is measured against, but it no longer answers on the
# model's behalf. A transient model failure must not return a rule-routed
# answer with no tokens spent, which is indistinguishable in the result from a
# question the model was never asked. Set ENERGYX_REQUIRE_LLM=0 to allow the
# fallback.
REQUIRE_LLM = os.environ.get("ENERGYX_REQUIRE_LLM", "1") not in ("", "0")


def agent_for(mode: str, model: str | None = None) -> EnergyXAgent:
    """One agent per (mode, model), built on first use and kept.

    Keyed on the model too, because an agent holds its planner and the planner
    holds the model.
    """
    key = (mode, model or "")
    if key not in _agents:
        llm = _llm(model)
        agent = EnergyXAgent(
            mode=mode, llm=llm, stateless=True, write_back=WRITE_BACK,
            require_llm=REQUIRE_LLM,
            budget=Budget(max_steps=MAX_STEPS, max_seconds=MAX_SECONDS))
        print(f"  [agent] {mode}/{model or 'default'}: llm={type(llm).__name__} "
              f"available={llm.available} planner={type(agent.planner).__name__} "
              f"require_llm={REQUIRE_LLM}", flush=True)
        _agents[key] = agent
    return _agents[key]


def db() -> SnapshotDB:
    global _db
    if _db is None:
        _db = SnapshotDB()
    return _db


# --- models -----------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: Literal["multi", "single"] = "single"      # 
    # Corrections the reader made to the resolved intent before running. See
    # /interpret: the chips it returns are exactly these fields, and a run
    # that carries any of them says so in its trace.
    intent_overrides: dict | None = None
    write_back: bool | None = None      # None = the service default
    # Which planner model answers. None uses the service default; anything
    # else has to be named in SELECTABLE_MODELS.
    model: str | None = None
    # Ask the model to read the composed answer back in plain English.
    narrate: bool | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str
    mode: str
    run_id: str
    plan: list[dict]
    notes: list[str] = []
    evidence: list[dict] = []
    verification: dict = {}
    metrics: dict = {}
    # The final step's rows as data rather than as formatted text, so the page
    # can draw a table and a chart instead of re-parsing its own answer string.
    table: dict | None = None
    figure: str | None = None       # URL of a rendered chart, when one was drawn
    # The typed answer: kind (scalar/boolean/table/ranking/set/refusal) and its
    # value. The agent has produced this since and the HTTP layer dropped
    # it, so no caller could read the verdict or the single number -- only the
    # formatted text.
    payload: dict | None = None
    # A plain-English reading of the answer above. Never the answer itself.
    narration: str | None = None
    seconds: float

HANDLE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _artifact_frame(run_id: str, handle: str) -> "pd.DataFrame":
    """Resolve a run-scoped artifact, refusing anything outside runs/."""
    if not HANDLE_RE.match(run_id) or not HANDLE_RE.match(handle):
        raise HTTPException(400, "malformed run id or handle")
    path = (PATHS.runs / run_id / "artifacts" / f"{handle}.parquet").resolve()
    if not path.is_relative_to(PATHS.runs.resolve()) or not path.exists():
        raise HTTPException(404, f"no artifact {handle!r} in run {run_id!r}")
    return pd.read_parquet(path)


def figure_url(answer) -> str | None:
    """URL of a chart this run drew, if any.

    Independent of the table builder: a plotting question often returns a
    figure and no rows, and the figure was being dropped with them.
    """
    for payload in list(answer.results.values())[::-1]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict) and data.get("figure_path"):
            return (f"/runs/{answer.run_id}/figures/"
                    f"{Path(data['figure_path']).stem}.png")
    return None


CHART_ROW_CAP = 400          # enough for an hourly profile or a year of days


def final_table(agent: EnergyXAgent, answer) -> dict | None:
    """Rows behind the final answer, read from the artifact when it spilled.

    The text answer shows five rows because that is what a reader scans. A chart
    needs the rest, so this reads the handle rather than charting the preview and
    calling it the shape of the data.
    """
    steps = answer.plan.get("steps") or []
    if not steps:
        return None
    payload = answer.results.get(str(steps[-1].get("id")))
    if not payload or payload.get("status") not in ("OK", "PARTIAL"):
        return None

    rows = payload.get("data") if isinstance(payload.get("data"), list) else None
    truncated = False
    handle = payload.get("handle")
    if handle:
        # Read the artifact off disk by run id, the same way the download
        # endpoint does, rather than through `agent.ctx.store`: EnergyXAgent
        # has no `ctx` -- only its Orchestrator does -- and the AttributeError
        # that follows is swallowed, leaving every result over the 50-row
        # inline limit showing a five-row preview instead.
        try:
            frame = _artifact_frame(answer.run_id, handle)
        except HTTPException:
            frame = None
        if frame is not None and (rows is None or len(frame) > len(rows)):
            truncated = len(frame) > CHART_ROW_CAP
            # `to_json` was used here and renders a datetime column as epoch
            # milliseconds, so every timestamp in the page's table and chart
            # arrived as 1500940800000 -- an unreadable integer where a reader
            # expected an hour. The tool layer already renders frames for
            # transport, including the wall-clock rule for local timestamps, so
            # the same function does it here and the two cannot disagree.
            rows = jsonify(frame.head(CHART_ROW_CAP)).to_dict(orient="records")
    if rows is None:
        rows = payload.get("preview") or []
        truncated = bool(payload.get("rows") and payload["rows"] > len(rows))
    if not rows:
        return None
    return {"tool": payload.get("provenance", {}).get("tool"),
            "handle": handle, "run_id": answer.run_id,
            "columns": payload.get("columns") or list(rows[0]),
            "rows": rows,
            "total_rows": payload.get("rows") or len(rows),
            "truncated": truncated,
            "evidence": payload.get("evidence") or {}}


# --- endpoints --------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    ensure_loaded()
    connection = db()
    graph = KnowledgeStore()
    counts = {
        "datasets": connection.value("SELECT COUNT(*) FROM datasets"),
        "buildings": connection.value("SELECT COUNT(*) FROM buildings"),
        "streams": connection.value("SELECT COUNT(*) FROM electricity_streams"),
        "sites": connection.value("SELECT COUNT(*) FROM sites"),
    }
    llm = _llm()
    return {
        "status": "ok",
        "snapshot": SNAPSHOT_VERSION,
        "catalog": counts,
        "graph_triples": graph.count() if graph.exists() else 0,
        "derived_facts": KnowledgeCurator(graph, SNAPSHOT_VERSION).stats()
        if graph.exists() else {},
        "tools": {"runtime": len(registry.runtime_tools()),
                  "onboarding": len(registry.for_agent("onboarding"))},
        # The planner each live agent is actually holding, rather than the
        # configured provider: only the former shows whether a cached agent is
        # really planning with a model.
        "agents": {mode: type(a.planner).__name__ for mode, a in _agents.items()},
        "write_back": WRITE_BACK,
        "llm": {"provider": llm.provider, "model": llm.model,
                "available": llm.available,
                "planner": "deterministic" if llm.provider == "null" else "llm"},
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    ensure_loaded()
    if request.model and request.model not in SELECTABLE_MODELS:
        raise HTTPException(
            400, f"model {request.model!r} is not offered by this service; "
                 f"available: {', '.join(SELECTABLE_MODELS) or 'none'}")
    began = time.time()
    with _lock:
        agent = agent_for(request.mode, request.model)
        agent.write_back = (WRITE_BACK if request.write_back is None
                            else request.write_back)
        try:
            answer = agent.ask(request.question,
                               intent_overrides=request.intent_overrides)
        except PlannerRefused as exc:
            # Strict planning: the model was asked, retried, and still produced
            # nothing usable. Saying so is the point of the setting -- the
            # alternative is a rule-routed answer presented as the model's.
            raise HTTPException(
                503, f"The planner model could not produce a usable plan for "
                     f"this question ({exc}). The request was retried; it was "
                     f"not answered by a fallback route.") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        finally:
            agent.write_back = WRITE_BACK
    wants = NARRATE if request.narrate is None else request.narrate
    narration = (_narrate(answer.question, answer.text, answer.payload,
                          request.model) if wants else None)
    return AskResponse(
        question=answer.question, answer=answer.text, status=answer.status,
        mode=answer.mode, run_id=answer.run_id,
        plan=answer.plan.get("steps", []), notes=answer.plan.get("notes", []),
        evidence=answer.evidence[:20], verification=answer.verification,
        metrics=answer.metrics, table=final_table(agent, answer),
        payload=answer.payload,
        figure=figure_url(answer),
        narration=narration,
        seconds=round(time.time() - began, 3))


class InterpretRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    intent_overrides: dict | None = None


# The categories and shapes a reader may choose between, named as they are
# named everywhere else in the system.
CATEGORY_NAMES = {
    "1": "Data discovery & availability", "2": "Data quality & preprocessing",
    "3": "Consumption & demand statistics", "4": "Temporal & load-profile",
    "5": "Performance & normalization", "6": "Weather & climate",
    "7": "Comparative & peer benchmarking", "8": "Spatial & spatio-temporal",
    "9": "Forecasting & prediction", "10": "Anomaly & change detection",
    "11": "Clustering, similarity & representation",
    "12": "Advanced inference & decision support",
}
OUTPUT_FORMATS = ("number", "text", "boolean", "list", "table", "ranking",
                  "figure", "map", "report")
SUBJECTS = ("buildings", "sites", "cities", "countries", "building_types",
            "datasets", "streams")


def _cohort_chip(intent, own_db=None) -> dict:
    """Who the answer is about, and how many of them there are.

    The count is the part a reader acts on: "site Fox" is agreement, "site Fox
    -- 137 buildings" is a fact they can recognise as right or wrong.
    """
    entities = intent.entities or {}
    named = entities.get("building_ids") or []
    parts, count = [], None
    if named:
        parts.append(", ".join(named[:3])
                     + (f" and {len(named) - 3} more" if len(named) > 3 else ""))
        count = len(named)
    for key, word in (("site_id", "site"), ("city", "city"),
                      ("country", "country"), ("building_type", "type")):
        if entities.get(key):
            parts.append(f"{word} {entities[key]}")
    if not parts:
        parts.append("the whole portfolio")
    detail = None
    try:
        where, params = [], []
        if named:
            where.append("building_id IN (" + ",".join("?" * len(named)) + ")")
            params += list(named)
        for column, key in (("site_id", "site_id"), ("city", "city"),
                            ("country", "country"),
                            ("building_type_canonical", "building_type")):
            if entities.get(key):
                where.append(f"{column} = ?"); params.append(entities[key])
        rows = (own_db or db()).df(
            "SELECT COUNT(*) AS n FROM buildings"
            + (" WHERE " + " AND ".join(where) if where else ""),
            params or None)
        count = int(rows["n"].iloc[0])
        detail = f"{count:,} building" + ("" if count == 1 else "s")
    except Exception:
        detail = f"{count:,} named" if count else None
    return {"key": "cohort", "label": "Cohort", "value": "; ".join(parts),
            "detail": detail, "editable": True, "field": "entities",
            "inputs": [
                {"name": "building_ids", "label": "buildings",
                 "kind": "list", "value": named},
                {"name": "site_id", "label": "site", "kind": "text",
                 "value": entities.get("site_id")},
                {"name": "city", "label": "city", "kind": "text",
                 "value": entities.get("city")},
                {"name": "building_type", "label": "building type",
                 "kind": "text", "value": entities.get("building_type")},
            ]}


def _period_chip(intent) -> dict:
    window = intent.time_range or {}
    label = window.get("label")
    if not label:
        label = ("the whole record" if not window.get("start")
                 else f"{window['start']} to {window.get('end') or 'the end'}")
    return {"key": "period", "label": "Period", "value": label,
            "detail": (window.get("assumed") or
                       ("end is exclusive" if window.get("end") else None)),
            "editable": True, "field": "time_range",
            "inputs": [
                {"name": "start", "label": "from", "kind": "date",
                 "value": window.get("start")},
                {"name": "end", "label": "to (exclusive)", "kind": "date",
                 "value": window.get("end")},
            ]}


def interpret_chips(intent, own_db=None) -> dict:
    """The resolved intent as the few facts a reader can check at a glance."""
    category = str(intent.task_category or "")
    return {
        "question": intent.question,
        "chips": [
            _cohort_chip(intent, own_db),
            _period_chip(intent),
            {"key": "metric", "label": "Measure", "value": intent.metric or "—",
             "detail": "what the answer measures", "editable": True,
             "field": "metric", "inputs": [
                 {"name": "metric", "label": "measure", "kind": "text",
                  "value": intent.metric}]},
            {"key": "output_format", "label": "Answer as",
             "value": intent.output_format or "table",
             "detail": "the shape the answer comes back in", "editable": True,
             "field": "output_format", "inputs": [
                 {"name": "output_format", "label": "shape", "kind": "choice",
                  "value": intent.output_format, "options": list(OUTPUT_FORMATS)}]},
            {"key": "subject", "label": "About",
             "value": intent.subject or "—",
             "detail": "what each row of the answer is", "editable": True,
             "field": "subject", "inputs": [
                 {"name": "subject", "label": "rows are", "kind": "choice",
                  "value": intent.subject, "options": list(SUBJECTS)}]},
            {"key": "task_category", "label": "Category",
             "value": (f"{category} — {CATEGORY_NAMES[category]}"
                       if category in CATEGORY_NAMES else "unclassified"),
             "detail": "which analysis this routes to", "editable": True,
             "field": "task_category", "inputs": [
                 {"name": "task_category", "label": "category", "kind": "choice",
                  "value": category,
                  "options": [f"{k} — {v}" for k, v in CATEGORY_NAMES.items()]}]},
        ],
        # Whatever the resolver could not settle. These are the reader's to
        # answer, and are the reason the panel exists rather than a rewrite:
        # the system knows exactly which choices it made under-determined.
        "ambiguities": list(intent.ambiguities or []),
        "out_of_scope": intent.out_of_scope,
        "group_by": intent.group_by,
        "recognised_shape": bool(intent.recognised_shape),
    }


# The reading panel gets its own resolver and its own database handle, and
# never touches the agent lock. Two reasons, both found the hard way.
#
# `/ask` holds that lock for the whole run -- minutes, on a loaded machine --
# so Check queued behind whatever question was in flight and appeared to do
# nothing. A pre-flight check that cannot run while a question is running is
# not a pre-flight check.
#
# And a DuckDB connection is not thread-safe. The cohort chip counts buildings,
# and sharing the service's connection with a request executing on another
# thread is a hang waiting to happen.
_interpret_lock = threading.Lock()
_interpreter: dict = {}


def _intent_agent():
    """A resolver for the reading panel alone, built once."""
    if "agent" not in _interpreter:
        from ..agents.planner import IntentAgent, Vocabulary
        own_db = SnapshotDB()
        _interpreter["db"] = own_db
        _interpreter["agent"] = IntentAgent(Vocabulary.from_db(own_db))
    return _interpreter["agent"], _interpreter["db"]


@app.post("/interpret")
def interpret(request: InterpretRequest) -> dict:
    """What the system understood, before it plans anything.

    No model is called: the intent resolver is rules over the catalog, so this
    answers in milliseconds and costs nothing. That is the point -- a reader
    can check the reading of every question, not only the ones they suspect.
    """
    ensure_loaded()
    # The lock here guards only this endpoint's own resolver and connection,
    # so two readers cannot race on one DuckDB handle; it is never held by a
    # running question.
    with _interpret_lock:
        agent, own_db = _intent_agent()
        intent = agent.resolve(request.question, carried={})
        applied = apply_intent_overrides(intent, request.intent_overrides)
        payload = interpret_chips(intent, own_db)
    payload["applied_overrides"] = applied
    return payload


@app.get("/tools")
def tools(agent: str | None = None) -> dict:
    ensure_loaded()
    specs = registry.for_agent(agent) if agent else registry.runtime_tools()
    by_agent: dict[str, int] = {}
    for spec in specs:                      # counted over the same set as `count`
        by_agent[spec.agent] = by_agent.get(spec.agent, 0) + 1
    return {"count": len(specs),
            "by_agent": dict(sorted(by_agent.items())),
            "tools": [s.describe() for s in sorted(specs, key=lambda x: x.name)]}


#: The evaluation set, read from the file the grader scores against.
EVAL_QUESTIONS = (Path(__file__).resolve().parents[2]
                  / "EnergyX-Eval" / "questions" / "final50_questions.csv")


@app.get("/models")
def models() -> dict:
    """Planner models a caller may choose between, and the default."""
    return {"default": os.environ.get("ENERGYX_LLM_MODEL", ""),
            "provider": os.environ.get("ENERGYX_LLM_PROVIDER", "null"),
            "models": list(SELECTABLE_MODELS)}


@app.get("/eval/questions")
def eval_questions() -> dict:
    """The 50 evaluation questions, for the question list in the interface."""
    if not EVAL_QUESTIONS.exists():
        return {"count": 0, "questions": [],
                "note": "EnergyX-Eval/questions/final50_questions.csv not found"}
    import csv
    with EVAL_QUESTIONS.open() as fh:
        rows = [{"id": r["id"], "difficulty": r.get("difficulty", ""),
                 "category": r.get("category", ""), "question": r["question"]}
                for r in csv.DictReader(fh)]
    categories = sorted({r["category"] for r in rows if r["category"]})
    return {"count": len(rows), "categories": categories, "questions": rows}


class GradeRequest(BaseModel):
    id: str
    answer: str


@app.post("/eval/grade")
def eval_grade(request: GradeRequest) -> dict:
    """Score one answer against its golden answer, with the published grader."""
    scripts = EVAL_QUESTIONS.parents[1] / "scripts"
    if not (scripts / "grade_gold50.py").exists():
        return {"id": request.id, "correct": None, "detail": "grader not present"}
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        from grade_gold50 import grade                          # noqa: PLC0415
        correct, detail, _evidence = grade(request.id, request.answer)
    except Exception as exc:                                    # noqa: BLE001
        return {"id": request.id, "correct": None, "detail": type(exc).__name__}
    return {"id": request.id, "correct": correct, "detail": detail}


@app.get("/runs/{run_id}/artifacts/{handle}.csv")
def artifact_csv(run_id: str, handle: str):
    """Download a result table as CSV, in full rather than as previewed.

    Artifact stores are run-scoped (``runs/<run_id>/artifacts``), so the run id
    is part of the address: a later request cannot reach an earlier run's store
    through the live agent. Both path parts are pattern-checked before they touch
    the filesystem, because both arrive from a URL and the store addresses files
    by name.
    """
    frame = _artifact_frame(run_id, handle)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return Response(
        content=buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{handle}.csv"',
                 "X-EnergyX-Rows": str(len(frame))})


def _html_escape(value) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


@app.get("/runs/{run_id}/artifacts/{handle}.html", response_class=HTMLResponse)
def artifact_html(run_id: str, handle: str) -> HTMLResponse:
    """The whole result as a standalone page: every row, sortable, printable."""
    frame = _artifact_frame(run_id, handle)
    numeric = {c: pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns}
    head = "".join(
        f'<th class="{"num" if numeric[c] else ""}" data-i="{i}">{_html_escape(c)}</th>'
        for i, c in enumerate(frame.columns))
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if numeric[c] else ""}">'
            + ("&mdash;" if value is None or (isinstance(value, float) and value != value)
               else _html_escape(value)) + "</td>"
            for c, value in row.items()) + "</tr>"
        for row in frame.to_dict(orient="records"))
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html_escape(handle)} · EnergyX</title><style>
:root{{color-scheme:light dark;--bg:#fbfbf9;--fg:#15150f;--mut:#87867d;--line:#e6e5de;
  --head:#f2f2ee;--zebra:#faf9f6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#121211;--fg:#f3f3ef;--mut:#88877f;
  --line:#2f2e2b;--head:#222220;--zebra:#191917}}}}
body{{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
header{{padding:18px 22px;border-bottom:1px solid var(--line);
  display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;background:var(--bg)}}
h1{{font-size:15px;margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.mut{{color:var(--mut);font-size:12.5px}}
a{{color:inherit}}
.wrap{{padding:0 22px 40px;overflow:auto;max-height:calc(100dvh - 96px)}}
table{{border-collapse:collapse;font-size:13px;width:100%}}
th{{position:sticky;top:0;z-index:3;background:var(--head);text-align:left;padding:8px 12px;
  border-bottom:1px solid var(--line);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--mut);white-space:nowrap;cursor:pointer;user-select:none}}
th:hover{{color:var(--fg)}}
th.num,td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td{{padding:6px 12px;border-bottom:1px solid var(--line);white-space:nowrap}}
td.num{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}}
tbody tr:nth-child(even){{background:var(--zebra)}}
@media print{{header{{position:static}}th{{position:static}}}}
</style></head><body>
<header><h1>{_html_escape(handle)}</h1>
  <span class="mut" title="data rows only; the header row is not counted">
    {len(frame):,} data rows &times; {len(frame.columns)} columns</span>
  <span class="mut">run {_html_escape(run_id)}</span>
  <span class="mut">&middot; click a column to sort</span>
  <span style="margin-left:auto"><a href="./{_html_escape(handle)}.csv">Download CSV</a></span>
</header>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<script>
let dir = 1, last = -1;
document.querySelectorAll("th").forEach(th => th.onclick = () => {{
  const i = +th.dataset.i, tb = document.querySelector("tbody");
  dir = (i === last) ? -dir : 1; last = i;
  const rows = [...tb.rows].sort((a, b) => {{
    const x = a.cells[i].textContent.trim(), y = b.cells[i].textContent.trim();
    const nx = parseFloat(x.replace(/,/g, "")), ny = parseFloat(y.replace(/,/g, ""));
    const both = !isNaN(nx) && !isNaN(ny);
    return (both ? nx - ny : x.localeCompare(y)) * dir;
  }});
  rows.forEach(r => tb.appendChild(r));
}});
</script></body></html>""")


@app.get("/runs/{run_id}/figures/{name}.png")
def artifact_figure(run_id: str, name: str) -> FileResponse:
    """Serve a rendered chart from a run's artifact store.

    create_visualization writes a PNG to the run's store and returns its path,
    which only a process on this machine could open. This serves it over HTTP.
    """
    if not HANDLE_RE.match(run_id) or not HANDLE_RE.match(name):
        raise HTTPException(400, "malformed run id or figure name")
    path = (PATHS.runs / run_id / "artifacts" / f"{name}.png").resolve()
    if not path.is_relative_to(PATHS.runs.resolve()) or not path.exists():
        raise HTTPException(404, f"no figure {name!r} in run {run_id!r}")
    return FileResponse(path, media_type="image/png")


@app.get("/catalog")
def catalog(entity: Literal["datasets", "countries", "cities", "sites",
                            "building_types"] = "datasets") -> dict:
    queries = {
        "datasets": "SELECT dataset_id, dataset_name, building_sector FROM datasets",
        "countries": "SELECT country_canonical AS country, COUNT(*) AS sites "
                     "FROM sites WHERE country_canonical IS NOT NULL GROUP BY 1",
        "cities": "SELECT city, country_canonical AS country FROM sites "
                  "WHERE city IS NOT NULL ORDER BY 2, 1",
        "sites": "SELECT site_id, site_name, city, country_canonical AS country, "
                 "n_buildings FROM sites ORDER BY n_buildings DESC",
        "building_types": "SELECT building_type_canonical AS building_type, "
                          "COUNT(*) AS buildings FROM buildings "
                          "WHERE building_type_canonical IS NOT NULL "
                          "GROUP BY 1 ORDER BY 2 DESC",
    }
    return {"entity": entity,
            "rows": db().df(queries[entity]).to_dict(orient="records")}


@app.get("/browse")
def browse(country: str | None = None, site_id: str | None = None,
           building_id: str | None = None) -> dict:
    """Walk the catalog one level at a time: country, site, building, year.

    Navigation only. The snapshot is partitioned dataset/site/year and stays
    that way: a country level adds nothing to pruning that site does not
    already give, and a building level would turn 37 files into 3,156 of
    thirty kilobytes each. This reads the catalog instead, so browsing costs
    nothing and the layout that makes scans fast is left alone.

    400 buildings sit at sites with no country recorded. They are their own
    branch rather than hidden, because a drill-down that silently omits a
    quarter of the portfolio is worse than one that names the gap.
    """
    params: list = []
    if country is None:
        rows = db().df("""
            SELECT COALESCE(s.country_canonical, '(country not recorded)') AS name,
                   COUNT(DISTINCT s.site_id) AS sites,
                   COUNT(b.building_id)      AS buildings
            FROM sites s LEFT JOIN buildings b ON b.site_id = s.site_id
            GROUP BY 1 ORDER BY buildings DESC""")
        return {"level": "country", "parent": None,
                "rows": rows.to_dict(orient="records")}

    if site_id is None:
        clause = ("s.country_canonical IS NULL"
                  if country.startswith("(") else "s.country_canonical = ?")
        if not country.startswith("("):
            params.append(country)
        rows = db().df(f"""
            SELECT s.site_id AS name, s.site_name, s.city, s.timezone,
                   COUNT(b.building_id) AS buildings
            FROM sites s LEFT JOIN buildings b ON b.site_id = s.site_id
            WHERE {clause}
            GROUP BY 1,2,3,4 ORDER BY buildings DESC""", params or None)
        return {"level": "site", "parent": country,
                "rows": rows.to_dict(orient="records")}

    if building_id is None:
        rows = db().df("""
            SELECT b.building_id AS name, b.building_type_canonical AS building_type,
                   b.floor_area_m2, b.year_built,
                   ROUND(q.coverage_fraction, 3) AS coverage,
                   q.quality_status
            FROM buildings b
            LEFT JOIN electricity_streams st ON st.building_id = b.building_id
            LEFT JOIN data_quality q ON q.stream_id = st.stream_id
            WHERE b.site_id = ?
            ORDER BY b.building_id""", [site_id])
        return {"level": "building", "parent": site_id,
                "rows": rows.to_dict(orient="records")}

    rows = db().df("""
        SELECT CAST(YEAR(period_start) AS INTEGER) AS name,
               SUM(n_observations)      AS readings,
               ROUND(SUM(total_kwh), 1) AS total_kwh,
               ROUND(AVG(coverage_fraction), 3) AS coverage
        FROM agg_yearly WHERE building_id = ?
        GROUP BY 1 ORDER BY 1""", [building_id])
    return {"level": "year", "parent": building_id,
            "rows": rows.to_dict(orient="records")}


@app.get("/runs/{run_id}/trace")
def trace(run_id: str) -> JSONResponse:
    # Path is confined to the runs directory: a run id is an identifier, not a
    # path fragment, and must not be able to escape it.
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(400, "invalid run id")
    path = Path(PATHS.runs) / run_id / "trace.json"
    if not path.exists():
        raise HTTPException(404, f"no trace for run {run_id}")
    return JSONResponse(content=__import__("json").loads(path.read_text()))


@app.get("/")
def index() -> FileResponse:
    # No caching for the page itself. A browser holding yesterday's index.html
    # shows an interface that no longer matches the API it is talking to.
    return FileResponse(STATIC / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


if ALLOWED_ORIGINS:
    # Named origins only. The service carries no authentication, so "*" would
    # let any page on the internet spend this machine's GPU.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware, allow_origins=list(ALLOWED_ORIGINS),
        allow_methods=["GET", "POST"], allow_headers=["*"])
    print(f"  [cors] browser calls allowed from: {', '.join(ALLOWED_ORIGINS)}",
          flush=True)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
