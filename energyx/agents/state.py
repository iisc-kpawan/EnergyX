"""Shared state / memory for one conversation.

Contents are exactly what the architecture specifies: user intent, conversation
state, the plan, handles to intermediate results, tool outputs, an evidence
ledger, provenance and execution history.

Scope is a *session*. Cross-session memory is deliberately off: a run that could
read another run's state would not be independent of it.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..tools.base import ToolResult


@dataclass
class Intent:
    """What Interaction & Intent resolved from the question."""

    question: str
    task_category: str | None = None
    entities: dict = field(default_factory=dict)
    time_range: dict = field(default_factory=dict)
    metric: str | None = None
    output_format: str | None = None
    ambiguities: list[str] = field(default_factory=list)
    # What the answer must be about, and what the snapshot cannot answer at all.
    # Both were previously resolved and then discarded: the intent shaped the
    # plan and nothing ever checked the answer against it, so a question about
    # buildings could be answered with the datasets catalog and pass every
    # verification check.
    subject: str | None = None          # buildings | sites | cities | ...
    out_of_scope: str | None = None     # the quantity asked for, if unavailable
    # Capability predicates the cohort must satisfy, and the dimension the
    # answer should carry one row per.
    requirements: list = field(default_factory=list)
    group_by: str | None = None
    # Whether a question-shape pattern matched, rather than the default tag
    # being assumed. It is what the hybrid planner asks before deciding
    # whether a rule-based route exists for this question at all.
    recognised_shape: bool = False

    def to_dict(self) -> dict:
        return {"question": self.question, "task_category": self.task_category,
                "entities": self.entities, "time_range": self.time_range,
                "metric": self.metric, "output_format": self.output_format,
                "subject": self.subject, "out_of_scope": self.out_of_scope,
                "requirements": self.requirements, "group_by": self.group_by,
                "recognised_shape": self.recognised_shape,
                "ambiguities": self.ambiguities}


@dataclass
class Turn:
    """One question and everything produced while answering it."""

    question: str
    intent: Intent | None = None
    analysis_spec: Any = None
    plan: Any = None                                   # agents.plan.Plan
    results: dict[str, ToolResult] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    answer: str | None = None
    verification: dict | None = None
    started_at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """Shared memory, readable and writable by every agent in the session."""

    session_id: str = field(default_factory=lambda: f"s_{uuid.uuid4().hex[:10]}")
    mode: str = "multi"
    turns: list[Turn] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)   # execution history
    entities: dict = field(default_factory=dict)        # carried across turns

    # -- turn lifecycle ----------------------------------------------------
    def begin_turn(self, question: str) -> Turn:
        turn = Turn(question=question)
        self.turns.append(turn)
        return turn

    @property
    def current(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    # -- results and evidence ---------------------------------------------
    def record(self, step_id: str, result: ToolResult) -> None:
        turn = self.current
        if turn is None:
            raise RuntimeError("no active turn")
        turn.results[step_id] = result
        if result.evidence:
            turn.evidence.append({"step": step_id,
                                  "tool": result.provenance.get("tool"),
                                  **result.evidence})

    def results(self) -> dict[str, ToolResult]:
        return self.current.results if self.current else {}

    def evidence_ledger(self) -> list[dict]:
        return self.current.evidence if self.current else []

    def note(self, event: str, **fields) -> None:
        self.history.append({"t": time.time(), "event": event, **fields})

    # -- follow-up support -------------------------------------------------
    def remember_entities(self, entities: dict) -> None:
        """Carry resolved entities forward so follow-ups can elide them."""
        for key, value in entities.items():
            if value:
                self.entities[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.entities.get(key, default)
