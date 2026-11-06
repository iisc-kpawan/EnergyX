"""The plan DAG.

Planning emits a typed, inspectable plan; Orchestration executes it.
Making the plan a first-class artifact is the point: it can be diffed against the
specification a question calls for, which turns planning accuracy, agent-selection
accuracy and tool-selection accuracy into three separately measurable numbers,
and it makes every failure replayable.

Steps refer to earlier results through ``$`` references rather than by passing
data around, which keeps intermediate frames out of the plan and out of any LLM
context:

    "$s1.handle"              the artifact handle produced by step s1
    "$s1.data"                s1's inline payload
    "$s1.data.eligible"       a key inside that payload
    "$s1.column.building_id"  one column, as a list
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..tools import registry

# Resolution strategy assigned by Planning. The Layer 4 router then
# picks an engine; these say whether computation is needed at all.
RESOLUTIONS = ("graph_only", "cached_fact", "hybrid", "compute")

REFERENCE = re.compile(
    r"^\$([A-Za-z0-9_]+)\.(handle|data|column|evidence)(?:\.(.+))?$")
# A planner writing "$1.building_id" means the building_id column of step 1.
# Accepting that is not laxity: the alternative is the string surviving as a
# literal and quietly matching nothing, which is indistinguishable from an
# empty result and was the single largest source of LLM plan failures.
SHORTHAND = re.compile(r"^\$([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)$")
# Any "$..." at all, with or without a field. The dot is optional: a bare "$2"
# must match too, or it passes through as the literal string "$2" -- the
# unresolved-reference failure the comment above describes.
class EmptyUpstream(KeyError):
    """A reference into a step that found nothing.

    Distinct from a wrong column name: nothing is misspelled and no retry will
    help, because the honest answer is that the filter matched no rows.
    """


ANY_REFERENCE = re.compile(r"^\$[A-Za-z0-9_]+")
# "$2" alone, which a planner writes when it means that step's handle.
BARE_REFERENCE = re.compile(r"^\$([A-Za-z0-9_]+)$")


def _distinct(values):
    """The column's values, each once, in the order they first appear.

    A column reference reads a whole column, and a frame is not always one row
    per entity: fetch_contextual_data returns one row per building per hour, so
    `$step.column.building_id` handed a cohort tool the same 77 ids seventeen
    thousand times over. The tool then joined on a cohort of 1.2 million and
    spent ten minutes producing a number that was seventy times too large.
    Order is kept because a ranking step upstream means it.
    """
    seen, out = set(), []
    for value in values:
        key = value if isinstance(value, (str, int, float, bool)) else repr(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


@dataclass
class PlanStep:
    id: str
    tool: str = ""                      # empty while a subgoal is unexpanded
    args: dict = field(default_factory=dict)
    agent: str | None = None            # filled from the registry when omitted
    depends_on: list[str] = field(default_factory=list)
    resolution: str = "compute"
    purpose: str = ""
    # Set in multi-agent mode: the objective handed to a specialist, which
    # expands it into concrete calls from its own tools.
    subgoal: str | None = None

    def __post_init__(self) -> None:
        # A dependency written as a string is one dependency, not a list of its
        # letters. A plan that said depends_on="1_1_find" was reported as
        # depending on missing steps "1", "_", "f" and so on, and was discarded
        # for it -- the plan was sound and the shape of one field was not.
        #
        # This has to be the only place that decides it. Four callers were
        # doing `[str(d) for d in raw["depends_on"]]` on the way in, which
        # splits the string before this runs and reintroduces the same bug in a
        # place that looks like careful type coercion. They now hand the raw
        # value over.
        if isinstance(self.depends_on, str):
            self.depends_on = [self.depends_on] if self.depends_on else []
        elif self.depends_on is None:
            self.depends_on = []
        else:
            # A dependency may arrive as an object rather than an id --
            # {"step": 1, "condition": "success"} -- and str() on that yields
            # the whole dict as a step name, which matches nothing. The id is
            # inside it, so take it here: this is the one place the shape of
            # the field is decided.
            self.depends_on = [
                str(d.get("step", d.get("id", d))) if isinstance(d, dict)
                else str(d)
                for d in self.depends_on]
        # A dependency names a step; a reference names a value inside one.
        # The plan language spells the second "$1.building_id", and a model
        # that has just written that in an argument writes "$1" here too.
        # Read strictly it is a step that does not exist, and six plans were
        # discarded whole for a sigil: "depends on missing step $1". The id
        # is what is left when the reference syntax is taken off.
        self.depends_on = [d[1:].split(".", 1)[0] if d.startswith("$") else d
                           for d in self.depends_on]

    @property
    def expanded(self) -> bool:
        return bool(self.tool)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    # Which rule-based route produced this plan, where one did: a named route
    # for a known question shape, "tag:<name>" for the generic branch of a
    # recognised shape, "unrecognised" where no pattern matched, and empty for
    # a plan a model wrote.
    route: str = ""
    question: str = ""
    notes: list[str] = field(default_factory=list)
    answer_step: str | None = None
    answer_field: str | None = None

    def add(self, step: PlanStep) -> PlanStep:
        """Append a step, giving it its owning agent.

        fill_agents() runs once when the plan is built, but repairs append
        steps afterwards and nothing re-filled them: every repair-added step
        was traced with no agent, so the agent split reported
        "(unassigned)" for work that belongs to reporting, performance or
        consumption. Assigning here means it cannot be missed, whoever adds
        the step and whenever.
        """
        if step.agent is None and step.tool:
            from ..tools import ensure_loaded
            ensure_loaded()
            if step.tool in registry.REGISTRY:
                step.agent = registry.get(step.tool).agent
        self.steps.append(step)
        return step

    def by_id(self, step_id: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    # -- validation --------------------------------------------------------
    def normalise(self) -> list[str]:
        """Repair what is safely repairable, and say what was repaired.

        ``resolution`` is a routing *hint*: which engine Planning expects to
        answer the step. A model that writes a sentence there has mislabelled a
        hint, not asked for something wrong, and the safe reading is "compute" --
        the engine that touches the data directly. Failing the whole plan over it
        sent every question to the deterministic planner instead.
        """
        from ..tools import ensure_loaded
        ensure_loaded()
        repairs: list[str] = []
        # A tool named with the wrong verb -- calculate_weather_relationship
        # for analyze_weather_relationship -- is a slip, not a request for
        # something that does not exist. One unambiguous near match is taken;
        # two candidates are left alone, because guessing between them would
        # be choosing the analysis on the model's behalf.
        for step in self.steps:
            if not step.tool or step.tool in registry.REGISTRY:
                continue
            import difflib
            close = difflib.get_close_matches(step.tool, registry.REGISTRY, n=2,
                                              cutoff=0.82)
            if len(close) == 1 or (len(close) == 2 and
                                   difflib.SequenceMatcher(None, step.tool, close[0]).ratio()
                                   - difflib.SequenceMatcher(None, step.tool, close[1]).ratio()
                                   > 0.08):
                repairs.append(f"{step.id}: no tool named {step.tool!r}; "
                               f"read as {close[0]!r}")
                step.tool = close[0]
        for step in self.steps:
            if step.resolution not in RESOLUTIONS:
                repairs.append(f"{step.id}: resolution {step.resolution!r} is not "
                               f"one of {sorted(RESOLUTIONS)}; read as 'compute'")
                step.resolution = "compute"
            # Two find_buildings steps in a plan are two cohorts -- offices
            # and education buildings, say -- not a cohort narrowed by another.
            # Given the option, the model chains the first step's ids into the
            # second and then asks for a different type, and the intersection
            # is empty: "how does EUI differ between offices and education"
            # came back as "nothing matched the question's filters".
            if (step.tool == "find_buildings" and step.args.get("building_ids")
                    and any(step.args.get(k) for k in
                            ("building_type", "building_subtype", "country",
                             "countries", "city", "site_id", "industry"))):
                carried = step.args["building_ids"]
                inherited = (isinstance(carried, str) and carried.startswith("$")) \
                    or (step.depends_on and isinstance(carried, list))
                if inherited:
                    step.args.pop("building_ids")
                    repairs.append(
                        f"{step.id}: dropped an inherited cohort -- it also "
                        f"names its own attributes, so it is a second cohort "
                        f"rather than a narrowing of the first")

            # A handle written into a parameter that does not take one, when
            # the same tool has a parameter that does, is a slip with one
            # sound reading: generate_report(evidence="$5.handle") meant
            # tables, because evidence is a dict of stated facts and tables is
            # the list of result handles. Refusing cost the whole plan --
            # three attempts, fifty-five seconds and no answer -- over an
            # argument that could be put where it belongs.
            spec = registry.REGISTRY.get(step.tool)
            if spec:
                properties = spec.parameters.get("properties", {})
                holders = [name for name in properties
                           if name == "handle" or name.endswith("_handle")
                           or name in ("tables", "handles")]
                for key, value in list(step.args.items()):
                    if (not isinstance(value, str) or not value.startswith("$")
                            or not value.endswith(".handle")):
                        continue
                    if key == "handle" or key.endswith("_handle") or key in ("tables", "handles"):
                        continue
                    # Prefer an empty holder, then a list one it can join, and
                    # failing both drop the argument: a report without one of
                    # its tables is worth more than no answer at all, which is
                    # what refusing cost.
                    free = next((h for h in holders if h not in step.args), None)
                    listy = next((h for h in holders
                                  if properties.get(h, {}).get("type") == "array"), None)
                    step.args.pop(key)
                    if free:
                        expects_list = properties.get(free, {}).get("type") == "array"
                        step.args[free] = [value] if expects_list else value
                        repairs.append(f"{step.id}: {key} does not take a handle; "
                                       f"moved {value!r} to {free}")
                    elif listy:
                        existing = step.args.get(listy)
                        if not isinstance(existing, list):
                            existing = [existing] if existing else []
                        if value not in existing:
                            existing.append(value)
                        step.args[listy] = existing
                        repairs.append(f"{step.id}: {key} does not take a handle; "
                                       f"added {value!r} to {listy}")
                    else:
                        repairs.append(f"{step.id}: {key} does not take a handle "
                                       f"and this tool has none; dropped {value!r}")

            # An empty "*_ids" list is a placeholder, and validate() knows the
            # one correct reading: every building when nothing produced a
            # cohort, the producing step's column when something did. Repaired
            # rather than refused, and the repair is reported.
            for key in [k for k, v in step.args.items()
                        if isinstance(v, list) and not v and k.endswith("_ids")]:
                if step.depends_on:
                    reference = f"${step.depends_on[0]}.{key[:-1]}"
                    step.args[key] = reference
                    repairs.append(f"{step.id}: {key} was an empty list; read as "
                                   f"{reference!r}, the cohort the previous step "
                                   f"produced")
                else:
                    step.args.pop(key)
                    repairs.append(f"{step.id}: {key} was an empty list; dropped, "
                                   f"which is how every building is asked for")
        return repairs

    def reseat_dangling(self) -> list[str]:
        """Resolve dependencies that name no step, where the data allows it.

        Two shapes turn up, and neither is a mistake about the analysis:

        A model writes a dependency as an object rather than an id --
        ``{"step": 1, "condition": "success"}`` instead of ``"1"``. The id is
        sitting inside it; the plan is discarded for the wrapper.

        A plan carries ordering dependencies on sibling sub-steps that were
        never expanded -- ``4_4_1`` waiting on ``3_1``, ``3_2``, ``3_3``.
        ``depends_on`` is ordering, not data: a step waiting on something that
        does not exist is waiting for nothing, and can run.

        A dangling ``$id.field`` reference is left alone. That one *is* about
        data: the step consumes an output no step produces, and running it
        anyway would answer from the wrong frame. validate() reports it.

        Returns notes describing what changed, empty when nothing did.
        """
        notes: list[str] = []
        for step in self.steps:
            unwrapped = []
            for dep in step.depends_on:
                if isinstance(dep, dict):
                    inner = dep.get("step", dep.get("id"))
                    if inner is not None:
                        unwrapped.append(str(inner))
                        notes.append(f"{step.id}: dependency {dep!r} read as "
                                     f"step {inner}")
                        continue
                unwrapped.append(dep)
            step.depends_on = unwrapped

        ids = {s.id for s in self.steps}
        for step in self.steps:
            needed = self.references(step)
            keep, dropped = [], []
            for dep in step.depends_on:
                if dep in ids or dep in needed:
                    keep.append(dep)
                else:
                    dropped.append(dep)
            if dropped:
                step.depends_on = keep
                notes.append(f"{step.id}: waiting on "
                             f"{', '.join(map(str, dropped))}, which no step "
                             f"produces; ordering dropped")
        return notes

    def validate(self) -> list[str]:
        """Structural problems. Empty list means the plan is executable."""
        from ..tools import ensure_loaded
        ensure_loaded()
        problems: list[str] = []
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                problems.append(f"duplicate step id: {step.id}")
            seen.add(step.id)
            if not step.tool:
                if not step.subgoal:
                    problems.append(f"{step.id}: neither a tool nor a subgoal")
                elif step.agent not in registry.AGENTS:
                    problems.append(f"{step.id}: subgoal needs a valid agent, "
                                    f"got {step.agent!r}")
            elif step.tool not in registry.REGISTRY:
                problems.append(f"{step.id}: unknown tool {step.tool!r}")
            elif step.agent and registry.get(step.tool).agent != step.agent:
                # A specialist naming a tool outside its own set is a report on
                # the routing, not a broken step: the router guessed the wrong
                # specialist and the specialist worked out the right tool
                # anyway. Rejecting the plan threw that away and fell back to
                # rules -- "quantify the impact of missing values on annual
                # EUI" was routed to data_quality, which asked for
                # calculate_normalized_performance and was refused for not
                # owning it. The step is reassigned to the agent that does own
                # the tool, which is the only agent that could have run it.
                step.agent = registry.get(step.tool).agent
            for dependency in step.depends_on:
                if dependency not in {s.id for s in self.steps}:
                    problems.append(f"{step.id}: depends on missing step {dependency}")
            for ref in self.references(step):
                if ref not in {s.id for s in self.steps}:
                    problems.append(f"{step.id}: references missing step {ref}")
            # An empty list where a cohort belongs is a placeholder the model
            # left behind, not a request to analyse nothing. It is read as an
            # unresolved reference rather than as an absent filter, because the
            # step that produced the cohort is sitting right above it, unused.
            for key, value in step.args.items():
                if isinstance(value, list) and not value and key.endswith("_ids"):
                    # Never intentional: omitting the argument means every
                    # building, so an empty list can only be a placeholder. It
                    # was caught before only when the step had a dependency,
                    # and a single-step plan passing it returned EMPTY as
                    # though nothing matched.
                    fix = (f'reference the step that produced them, as '
                           f'"${step.depends_on[0]}.{key[:-1]}"'
                           if step.depends_on else
                           "omit the argument to mean every building")
                    problems.append(f"{step.id}: {key} is an empty list; {fix}")
        if self.answer_step and self.by_id(self.answer_step) is None:
            problems.append(f"answer step {self.answer_step} does not exist")
        if not problems:
            try:
                self.order()
            except ValueError as exc:
                problems.append(str(exc))
        return problems

    @staticmethod
    def references(step: PlanStep) -> set[str]:
        found: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, str):
                match = REFERENCE.match(value) or SHORTHAND.match(value)
                if match:
                    found.add(match.group(1))
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(step.args)
        return found

    def order(self) -> list[PlanStep]:
        """Topological order; raises on a cycle."""
        pending = {s.id: set(s.depends_on) | self.references(s) for s in self.steps}
        resolved: list[PlanStep] = []
        while pending:
            ready = [sid for sid, deps in pending.items()
                     if not (deps - {s.id for s in resolved})]
            if not ready:
                raise ValueError(f"cycle or unreachable steps: {sorted(pending)}")
            for sid in ready:
                resolved.append(self.by_id(sid))
                pending.pop(sid)
        return resolved

    # -- serialisation -----------------------------------------------------
    def fill_agents(self) -> "Plan":
        """Attach each step's owning agent from the registry."""
        from ..tools import ensure_loaded
        ensure_loaded()
        for step in self.steps:
            if step.agent is None and step.tool and step.tool in registry.REGISTRY:
                step.agent = registry.get(step.tool).agent
        return self

    def to_dict(self) -> dict:
        return {"question": self.question, "notes": self.notes,
                "answer_step": self.answer_step, "answer_field": self.answer_field,
                "route": self.route,
                "steps": [s.to_dict() for s in self.steps]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        return cls(question=data.get("question", ""),
                   notes=list(data.get("notes", [])),
                   answer_step=data.get("answer_step"),
                   answer_field=data.get("answer_field"),
                   steps=[PlanStep(**s) for s in data.get("steps", [])])

    def summary(self) -> str:
        lines = []
        for step in self.steps:
            deps = f" <- {','.join(step.depends_on)}" if step.depends_on else ""
            label = step.tool or f"{step.agent}: {step.subgoal}"
            lines.append(f"{step.id}: [{step.resolution}] {label}{deps}"
                         + (f"  # {step.purpose}" if step.purpose else ""))
        return "\n".join(lines)


def resolve_args(args: Any, results: dict, store) -> Any:
    """Replace ``$`` references with values from completed steps."""
    if isinstance(args, str):
        match = REFERENCE.match(args)
        if not match:
            bare = BARE_REFERENCE.match(args)
            if bare:
                # "$2" means step 2's output. A handle is the only thing a bare
                # step reference can sensibly denote, and it is what a planner
                # means when it writes one.
                result = results.get(bare.group(1))
                if result is None:
                    raise KeyError(f"reference to step {bare.group(1)} which has "
                                   f"no result")
                if result.handle:
                    return result.handle
                raise KeyError(f"step {bare.group(1)} produced no handle to "
                               f"reference")

            shorthand = SHORTHAND.match(args)
            if shorthand:
                # "$1.building_id" -> the building_id column of step 1
                step_id, kind, path = shorthand.group(1), "column", shorthand.group(2)
                match = True
            elif ANY_REFERENCE.match(args):
                raise KeyError(
                    f"unresolvable reference {args!r}; use $step.handle, "
                    f"$step.data[.key] or $step.column.<name>")
            else:
                return args
        else:
            step_id, kind, path = match.group(1), match.group(2), match.group(3)
        result = results.get(step_id)
        if result is None:
            raise KeyError(f"reference to step {step_id} which has no result")
        if kind == "handle":
            return result.handle
        if kind == "evidence":
            # A tool that publishes a companion artifact names it in evidence,
            # and a later step can then chart or score that instead of the
            # summary. Without this, a comparison that computed two 24-hour
            # curves could only hand on the one row it reduced them to, and
            # the chart step had nothing with an hour axis to draw.
            value = (result.evidence or {}).get(path)
            if value is None:
                raise KeyError(f"step {step_id} declares no evidence {path!r}")
            return value
        if kind == "data":
            value = result.data
            for key in (path.split(".") if path else []):
                value = value[key] if isinstance(value, dict) else getattr(value, key)
            return value
        # kind == "column"
        # The stored frame first, and the inline data only when there is no
        # frame. `data` is a *preview* -- a handful of rows kept for a reader --
        # and reading a cohort out of it handed the next step those rows and no
        # more: a portfolio of 1,636 buildings arrived as 21, and every
        # statistic computed from it described them. The handle is the result;
        # the preview is a summary of it.
        loaded = store.get_or_none(result.handle)
        if loaded is not None and path in loaded.columns:
            return _distinct(loaded[path].dropna().tolist())
        if isinstance(result.data, list) and result.data:
            if path in result.data[0]:
                return _distinct([row[path] for row in result.data if path in row])
        if isinstance(result.data, dict) and path in result.data:
            return result.data[path]
        available = (list(loaded.columns) if loaded is not None
                     else sorted(result.data[0]) if isinstance(result.data, list)
                     and result.data else [])
        # The parameter's name written where the column's belongs: a cohort
        # goes in building_ids, and the column that fills it is building_id.
        # There is one column it could mean and it is in hand, so read it
        # rather than failing the step over the plural.
        if path.endswith("s") and path[:-1] in available:
            singular = path[:-1]
            if loaded is not None:
                return loaded[singular].dropna().tolist()
            return [row[singular] for row in result.data if singular in row]
        if not available:
            # The step has no columns because it matched nothing, not because
            # the reference is wrong. Saying "column not available; available:
            # []" sends the next attempt hunting for a column name when the
            # upstream simply found no rows -- which is an answer, and should
            # be reported as one.
            raise EmptyUpstream(
                f"step {step_id} matched nothing, so {path!r} has no values "
                f"to read; the answer is that nothing matched")
        raise KeyError(f"column {path!r} not available from step {step_id}; "
                       f"available: {available}")
    if isinstance(args, dict):
        return {k: resolve_args(v, results, store) for k, v in args.items()}
    if isinstance(args, list):
        return [resolve_args(v, results, store) for v in args]
    return args
