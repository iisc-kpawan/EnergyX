"""Run traces.

Every run emits a bundle: intent, plan, agent and tool calls with arguments,
evidence, provenance, timings, token accounting and the verification verdict.

This is not logging. It is the artifact an evaluator scores against the
specification a question calls for, and the thing that makes a failure replayable
rather than merely reported.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PATHS
from ..tools.base import ToolResult


@dataclass
class StepTrace:
    step_id: str
    agent: str | None
    tool: str
    resolution: str
    args: dict
    status: str
    started_at: float
    duration_s: float
    handle: str | None = None
    rows: int | None = None
    message: str | None = None
    predicate: str | None = None
    evidence: dict = field(default_factory=dict)
    attempt: int = 1
    engine: str | None = None
    strategy_consistent: bool | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Trace:
    run_id: str
    question: str
    mode: str
    snapshot: str
    started_at: float = field(default_factory=time.time)
    intent: dict | None = None
    # What the caller corrected before planning, if anything. A run whose
    # intent was edited is not the same evidence as one that resolved it
    # unaided, and a results table that cannot tell them apart is measuring
    # two systems at once.
    intent_overrides: dict = field(default_factory=dict)
    # What the question assumed, and whether the data bore it out.
    premises: dict = field(default_factory=dict)
    analysis_spec: dict | None = None
    analysis_spec_problems: list[str] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    verification: dict | None = None
    answer: str | None = None
    answer_payload: dict | None = None
    status: str = "running"
    budget: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    llm_calls: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # "llm" | "hierarchical" | "deterministic" -- what produced the plan that
    # actually ran, including a silent fall back from a model plan.
    planner_used: str = ""

    # -- recording ---------------------------------------------------------
    def record_plan(self, plan: Any, attempt: int = 1, problems: list[str] | None = None):
        self.plans.append({"attempt": attempt, "plan": plan.to_dict(),
                           "problems": problems or []})

    def record_step(self, step, result: ToolResult, started: float,
                    args: dict, attempt: int = 1, routing=None) -> StepTrace:
        entry = StepTrace(
            step_id=step.id, agent=step.agent, tool=step.tool,
            resolution=step.resolution, args=_safe(args),
            status=result.status.value, started_at=started,
            duration_s=round(time.time() - started, 4),
            handle=result.handle, rows=result.rows,
            message=result.message, predicate=result.predicate,
            evidence=_safe(result.evidence), attempt=attempt,
            engine=getattr(routing, "engine", None),
            strategy_consistent=getattr(routing, "consistent", None),
        )
        self.steps.append(entry)
        return entry

    def record_llm(self, provider: str, model: str, purpose: str,
                   prompt_tokens: int = 0, completion_tokens: int = 0,
                   duration_s: float = 0.0) -> None:
        self.tokens["prompt"] += prompt_tokens
        self.tokens["completion"] += completion_tokens
        self.llm_calls.append({"provider": provider, "model": model,
                               "purpose": purpose,
                               "prompt_tokens": prompt_tokens,
                               "completion_tokens": completion_tokens,
                               "duration_s": round(duration_s, 4)})

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -- metrics the harness reads ----------------------------------------
    def metrics(self) -> dict:
        by_status: dict[str, int] = {}
        for step in self.steps:
            by_status[step.status] = by_status.get(step.status, 0) + 1
        resolutions: dict[str, int] = {}
        engines: dict[str, int] = {}
        for step in self.steps:
            resolutions[step.resolution] = resolutions.get(step.resolution, 0) + 1
            if step.engine:
                engines[step.engine] = engines.get(step.engine, 0) + 1
        routed = [s for s in self.steps if s.strategy_consistent is not None]
        return {
            "steps_executed": len(self.steps),
            # Which planner actually produced the plan, not which one was
            # configured. A cached agent that silently degraded to the
            # deterministic planner reported "planner: llm" at /health while
            # never calling a model, and no answer revealed it.
            "planner": self.planner_used or "unknown",
            "plans_emitted": len(self.plans),
            "replans": max(0, len(self.plans) - 1),
            "status_counts": by_status,
            "resolution_counts": resolutions,
            "engine_counts": engines,
            "strategy_accuracy": (
                sum(1 for s in routed if s.strategy_consistent) / len(routed)
                if routed else None),
            "measurement_steps": engines.get("measurement", 0) + engines.get("model", 0),
            "tools_called": [s.tool for s in self.steps],
            "agents_used": sorted({s.agent for s in self.steps if s.agent}),
            "wall_clock_s": round(time.time() - self.started_at, 3),
            "tokens": self.tokens,
            "llm_calls": len(self.llm_calls),
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "question": self.question, "mode": self.mode,
            "snapshot": self.snapshot, "status": self.status,
            "started_at": self.started_at, "budget": self.budget,
            "intent": self.intent,
            "analysis_spec": self.analysis_spec,
            "analysis_spec_problems": self.analysis_spec_problems,
            "plans": self.plans,
            "steps": [s.to_dict() for s in self.steps],
            "verification": self.verification,
            "answer": self.answer,
            "answer_payload": self.answer_payload,
            "metrics": self.metrics(),
            "llm_calls": self.llm_calls,
            "notes": self.notes,
        }

    def write(self, root: Path | None = None) -> Path:
        directory = Path(root or PATHS.runs) / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trace.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def _safe(value: Any) -> Any:
    """Make a value JSON-serialisable, truncating long id lists."""
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_safe(v) for v in value[:20]]
        if len(value) > 20:
            items.append(f"... {len(value) - 20} more")
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
