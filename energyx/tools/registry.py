"""The tool registry.

One registry, two surfaces. Agents call functions directly in-process; an MCP
wrapper can expose the same specs unchanged. The multi/single agent
toggle is a *filter over this registry* -- in ``multi`` each specialist sees only
its own subset, in ``single`` one agent sees them all -- which is what keeps the
two evaluation arms honest.

Parameter schemas are derived from type hints, so a tool's signature is its
contract and the two cannot drift.
"""
from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from .artifacts import ArtifactStore
from .base import (ToolResult, UnknownBuildings, error,
                   insufficient)

AGENTS = (
    "data_discovery", "data_quality", "consumption", "performance",
    "weather_spatial", "ai_analytics", "reporting", "knowledge", "onboarding",
)

RUNTIME_AGENTS = (
    "data_discovery", "data_quality", "consumption", "performance",
    "weather_spatial", "ai_analytics", "reporting", "knowledge",
)


@dataclass
class ToolContext:
    """Everything a tool needs that is not an argument."""

    db: Any                                   # energyx.access.SnapshotDB
    store: ArtifactStore
    snapshot_version: str = ""
    min_coverage: float = 0.90
    # When the run must be finished by. The budget was only checked *between*
    # steps, so one heavy step ran unbounded: a portfolio-wide weather
    # regression took over eight minutes, holding the service lock the whole
    # time and showing the user nothing.
    deadline: float | None = None
    options: dict = field(default_factory=dict)
    _graph: Any = None

    @property
    def out_of_time(self) -> bool:
        """Whether the remaining budget is spent."""
        import time
        return self.deadline is not None and time.time() >= self.deadline

    @property
    def graph(self):
        """The knowledge store, opened on first use.

        Lazy because a run that never asks a graph question should not pay to
        open it, and because the store is a rebuildable projection -- a missing
        one is a recoverable condition, not a failed run.
        """
        if self._graph is None:
            from ..knowledge.store import KnowledgeStore
            self._graph = KnowledgeStore(snapshot_version=self.snapshot_version)
        return self._graph

    @property
    def write_back_enabled(self) -> bool:
        """Derived facts may only be asserted when explicitly allowed."""
        return bool(self.options.get("write_back", False))


@dataclass
class ToolSpec:
    name: str
    agent: str
    fn: Callable[..., ToolResult]
    summary: str
    parameters: dict
    # Parameters a human analyst may pass through the Python API but a planner
    # may not: arguments that silently change which rows come back and that no
    # model has evidence to set. They are absent from `parameters`, so
    # validate_args rejects them in a plan.
    analyst_only: tuple = ()

    def describe(self) -> dict:
        """JSON-schema-shaped description, as handed to an LLM."""
        return {"name": self.name, "agent": self.agent,
                "description": self.summary, "parameters": self.parameters}


REGISTRY: dict[str, ToolSpec] = {}

_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _unwrap_optional(annotation):
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return (args[0] if len(args) == 1 else annotation), True
    return annotation, False


def _json_type(annotation) -> dict:
    annotation, _ = _unwrap_optional(annotation)
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    origin = get_origin(annotation)
    if origin is Literal:
        options = list(get_args(annotation))
        return {"type": "string", "enum": [str(o) for o in options]}
    if origin in (list, set, tuple):
        args = get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    # A union that mixes a Literal with another type -- `Literal["mean",
    # "minmax", "none"] | bool` -- fell through to a bare string here, and with
    # it went the enum. Nothing then validated the argument and the planner
    # never saw the valid values: one call arrived with normalize="string",
    # the JSON type name copied out of the schema, and the tool raised.
    if origin in (types.UnionType, typing.Union):
        options: list[str] = []
        has_bool = False
        for part in get_args(annotation):
            if part is type(None):
                continue
            if part is bool:
                has_bool = True
            elif get_origin(part) is Literal:
                options.extend(str(o) for o in get_args(part))
        if options:
            if has_bool:
                options.extend(["true", "false"])
            return {"type": "string", "enum": options}
    return {"type": "string"}


def _build_parameters(fn: Callable, analyst_only: tuple = ()) -> dict:
    hints = typing.get_type_hints(fn)
    signature = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if name == "ctx" or name in analyst_only:
            continue
        schema = _json_type(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            schema["default"] = param.default
        properties[name] = schema
    return {"type": "object", "properties": properties, "required": required}


def tool(agent: str, name: str | None = None,
         analyst_only: tuple = ()) -> Callable:
    """Register a function as an EnergyX tool.

    The wrapper converts any unexpected exception into ``Status.ERROR`` so the
    no-raise contract holds even for bugs.
    """
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")

    def decorate(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        tool_name = name or fn.__name__
        summary = (fn.__doc__ or "").strip().split("\n\n")[0].strip()

        def wrapper(ctx: ToolContext, *args, **kwargs) -> ToolResult:
            try:
                result = fn(ctx, *args, **kwargs)
            except UnknownBuildings as exc:
                # A cohort naming no building is the caller's mistake and the
                # caller can fix it, so it is a refusal with a predicate rather
                # than a crash: ERROR reads as "the tool broke" and loses the
                # one thing worth saying, which is that those ids are not
                # buildings.
                return insufficient("building_ids", str(exc))
            except Exception as exc:  # noqa: BLE001 - deliberate boundary
                return error(f"{tool_name}: {type(exc).__name__}: {exc}")
            if not isinstance(result, ToolResult):
                return error(f"{tool_name}: returned {type(result).__name__}, "
                             "expected ToolResult")
            result.provenance.setdefault("tool", tool_name)
            result.provenance.setdefault("agent", agent)
            if ctx.snapshot_version:
                result.provenance.setdefault("snapshot", ctx.snapshot_version)
            return result

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__wrapped__ = fn
        signature = inspect.signature(fn)
        for hidden in analyst_only:
            if hidden not in signature.parameters:
                raise ValueError(
                    f"{tool_name}: analyst_only names {hidden!r}, which is not "
                    f"a parameter of the function")
        REGISTRY[tool_name] = ToolSpec(
            name=tool_name, agent=agent, fn=wrapper, summary=summary,
            parameters=_build_parameters(fn, analyst_only),
            analyst_only=tuple(analyst_only),
        )
        return wrapper

    return decorate


# --- lookup -----------------------------------------------------------------

def get(name: str) -> ToolSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool: {name}")
    return REGISTRY[name]


def for_agent(agent: str) -> list[ToolSpec]:
    return [s for s in REGISTRY.values() if s.agent == agent]


def runtime_tools() -> list[ToolSpec]:
    return [s for s in REGISTRY.values() if s.agent in RUNTIME_AGENTS]


def visible_to(mode: str, agent: str | None = None) -> list[ToolSpec]:
    """The toggle: ``single`` sees every runtime tool, ``multi`` a subset."""
    if mode == "single":
        return runtime_tools()
    if agent is None:
        raise ValueError("multi mode requires an agent")
    return for_agent(agent)


# Parameters expressed as a fraction of one. A model that writes 90 for "90%"
# otherwise passes type checking and silently matches every row.
ALTERNATIVE_PARAM = {"min_coverage": "min_months_of_data"}

FRACTION_PARAMS = {
    "min_coverage", "max_coverage", "coverage_fraction", "clip_quantile",
    "percentile", "contamination", "area_tolerance", "min_shift_pct",
}


#: Names a planner writes for a parameter a tool spells differently. A rename
#: is applied only when the tool does not have the written name and does have
#: exactly one of the candidates, so a tool that genuinely offers both is never
#: touched. These are synonyms, not guesses: "group_by" and "across" name the
#: same grouping, and a call refused over the spelling loses the question while
#: having read it correctly.
PARAM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "group_by": ("across",),
    "groupby": ("across",),
    "by": ("across",),
    "metric": ("metrics",),
    "metrics": ("metric",),
    "limit": ("k", "top_n"),
    "top_n": ("limit", "k"),
    "k": ("limit",),
    "normalise": ("normalize",),
    "normalize": ("normalise",),
    "season": ("season_length",),
    "horizon": ("horizon_hours",),
    "window": ("window_days",),
}


def canonical_args(name: str, args: dict) -> tuple[dict, list[str]]:
    """Rename parameters a tool spells differently, and report what moved.

    Returns the arguments the tool can accept and a note per rename. A name
    with no unambiguous home is left alone for validate_args to reject, which
    keeps a genuine mistake visible instead of quietly dropping it.
    """
    spec = get(name)
    properties = spec.parameters.get("properties", {})
    if not properties:
        return args, []
    out, notes = {}, []
    for key, value in args.items():
        if key in properties:
            fixed = _fit_shape(properties[key], value)
            if fixed is not value:
                notes.append(f"{name}: {key} unwrapped from a nested list")
            out[key] = fixed
            continue
        landing = [c for c in PARAM_SYNONYMS.get(key, ())
                   if c in properties and c not in args]
        if len(landing) != 1:
            out[key] = value
            continue
        target = landing[0]
        # A tool that takes a list should not be handed the bare string the
        # singular name invited: metric="eui" becomes metrics=["eui"].
        if (properties[target].get("type") == "array"
                and not isinstance(value, (list, tuple))):
            value = [value]
        elif (properties[target].get("type") != "array"
              and isinstance(value, (list, tuple)) and len(value) == 1):
            value = value[0]
        out[target] = _fit_shape(properties[target], value)
        notes.append(f"{name}: {key} read as {target}")
    return out, notes


def _fit_shape(schema: dict, value):
    """Unwrap a list of scalars that arrived wrapped in another list."""
    if schema.get("type") != "array" or not isinstance(value, (list, tuple)):
        return value
    items = schema.get("items") or {}
    if items.get("type") == "array":
        return value
    if value and all(isinstance(v, (list, tuple)) for v in value):
        return [inner for v in value for inner in v]
    return value


def validate_args(name: str, args: dict) -> list[str]:
    """Type- and enum-check resolved arguments before a tool runs.

    Catches the failure that is otherwise invisible: a scalar handed to a list
    parameter matches nothing and returns EMPTY, which looks exactly like a
    genuinely empty result.
    """
    spec = get(name)
    properties = spec.parameters.get("properties", {})
    problems: list[str] = []
    for key, value in args.items():
        # A ".handle" reference names a whole table. Passed to a scalar the
        # resolver hands over the handle *string*, which matches no site, no
        # building and no city -- an EMPTY result indistinguishable from a
        # genuinely empty one. Only a parameter that asks for a handle may
        # receive one.
        if (isinstance(value, str) and value.startswith("$")
                and value.endswith(".handle")
                and key != "handle" and not key.endswith("_handle")):
            problems.append(
                f"{name}.{key} is a scalar and cannot take {value!r}; a handle "
                f"names a whole table. Use $step.column.<name> for one column, "
                f"or a literal value.")
            continue
        if key not in properties:
            problems.append(f"{name}: unknown parameter {key!r}; "
                            f"valid: {sorted(properties)}")
            continue
        schema = properties[key]
        if value is None:
            continue
        expected = schema.get("type")
        # A ".column." reference names a whole column -- many values. Bound to
        # a scalar parameter the resolver hands over the column, and the error
        # surfaces far downstream as a database cast failure ("VARCHAR ...
        # can't be cast to VARCHAR[]") that names neither the tool nor the
        # argument at fault. Same reasoning as the .handle guard above: refuse
        # here, where the message can say which parameter and what to do.
        if (isinstance(value, str) and value.startswith("$")
                and ".column." in value and expected != "array"):
            problems.append(
                f"{name}.{key} takes one value and cannot take {value!r}, "
                f"which names a whole column. Name a single value, or use a "
                f"tool that accepts a cohort.")
            continue
        # References are resolved before validation, so the column arrives as
        # a list. Unchecked it reached the database as `WHERE building_id = ?`
        # bound to 1,636 ids and failed with a cast error naming neither the
        # tool nor the argument. One value is bound through silently, because
        # a cohort of one is a legitimate way to reach a single-subject tool.
        if expected in ("string", "number", "integer") and isinstance(value, (list, tuple)):
            if len(value) == 1:
                args[key] = value = value[0]
            elif key in ("building_id", "site_id"):
                # The question named no subject, so the planner handed over
                # every candidate. Refusing tells the reader nothing; the
                # reference solver simply chose one. This chooses the first in
                # the cohort's own order, which is stable, and the tools that
                # take a single subject all name it in their answer -- so the
                # choice is visible rather than silent.
                args[key] = value = value[0]
            else:
                problems.append(
                    f"{name}.{key} takes one value but was given {len(value)}"
                    + (f" ({value[0]!r}, {value[1]!r}, ...)" if len(value) > 1 else "")
                    + ". Name a single subject, or use a tool that takes a cohort.")
                continue
        # A model writing JSON by hand quotes its booleans about as often as
        # not: "normalize": "true" reached the tool as the string and raised
        # `unknown normalize: 'true'`, which reads as a bad argument rather
        # than a quoting slip. The two spellings mean the same thing and only
        # one of them is valid, so the obvious one is accepted.
        if isinstance(value, str) and value.lower() in ("true", "false"):
            truth = value.lower() == "true"
            if expected == "boolean":
                args[key] = value = truth
            else:
                # A union like `Literal["mean", "minmax", "none"] | bool`
                # declares no single type; accept the boolean only where the
                # literal set does not already spell the word out.
                allowed = schema.get("enum") or []
                if not allowed or value not in allowed:
                    if any(isinstance(a, bool) for a in allowed) or not allowed:
                        args[key] = value = truth
        if expected == "array" and not isinstance(value, (list, tuple)):
            problems.append(f"{name}.{key} expects a list, got "
                            f"{type(value).__name__} {str(value)[:40]!r}")
        elif expected in ("number", "integer") and isinstance(value, str):
            try:
                float(value)
            except ValueError:
                problems.append(f"{name}.{key} expects a number, got {value!r}")
        elif schema.get("enum") and isinstance(value, str) \
                and value not in schema["enum"]:
            problems.append(f"{name}.{key}={value!r} is not one of "
                            f"{schema['enum']}")
        if key in FRACTION_PARAMS and key != "min_shift_pct":
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = None
            if number is not None and not 0.0 <= number <= 1.0:
                # A number out of range is usually the wrong parameter rather
                # than the wrong scale: "at least 12 months" became
                # min_coverage=12, and suggesting 0.12 sends the next attempt
                # somewhere just as wrong. Name the parameter that expresses
                # what a whole number here most likely meant.
                instead = ALTERNATIVE_PARAM.get(key)
                hint = ""
                if instead and instead in spec.parameters.get("properties", {}):
                    hint = (f" (for a length of record use {instead}; "
                            f"for a percentage use {number / 100:g})")
                elif number <= 100:
                    hint = f" (did you mean {number / 100:g}?)"
                problems.append(
                    f"{name}.{key}={value} must be a fraction between 0 and 1"
                    + hint)
    for required in spec.parameters.get("required", []):
        if required not in args or args[required] is None:
            problems.append(f"{name}: required parameter {required!r} is missing")
    return problems


def call(name: str, ctx: ToolContext, **kwargs) -> ToolResult:
    return get(name).fn(ctx, **kwargs)


def counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in REGISTRY.values():
        out[spec.agent] = out.get(spec.agent, 0) + 1
    return out
