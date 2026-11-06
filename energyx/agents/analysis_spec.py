"""Validated semantic request passed from grounding to planning."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


TASKS = {
    "discovery", "relationship", "export", "aggregate", "quality", "profile",
    "normalize", "benchmark", "compare", "forecast", "anomaly",
    "change_point", "cluster", "similarity", "prioritize", "refusal",
}
RESOLUTIONS = {"graph_only", "cached_fact", "hybrid", "compute"}
OUTPUT_KINDS = {"scalar", "table", "set", "ranking", "forecast", "anomaly",
                "figure", "map", "refusal"}
GROUP_FIELDS = {"building_id", "building_type", "site_id", "city", "country",
                "dataset_id", "year", "month", "day", "hour", "weekday",
                "season"}


@dataclass
class AnalysisSpec:
    task: str
    entities: dict = field(default_factory=dict)
    period: dict = field(default_factory=dict)
    measure: str | None = None
    aggregation: str | None = None
    grain: str | None = None
    group_by: list[str] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    normalization: str | None = None
    comparison: dict | None = None
    ranking: dict | None = None
    method: str | None = None
    parameters: dict = field(default_factory=dict)
    output: dict = field(default_factory=lambda: {"kind": "table"})
    required_inputs: list[str] = field(default_factory=list)
    resolution: str = "compute"
    issues: list[str] = field(default_factory=list)

    @classmethod
    def from_intent(cls, intent, known_facts=None) -> "AnalysisSpec":
        """Translate catalog-grounded intent into a planner-safe contract."""
        text = intent.question.lower()
        tag = intent.metric or "consumption"
        task = {
            "discovery": "discovery", "known_facts": "discovery",
            "entity_path": "relationship", "semantic_export": "export",
            "quality": "quality", "forecast": "forecast",
            "anomaly": "anomaly", "change_point": "change_point",
            "cluster": "cluster", "similar": "similarity",
            "prioritize": "prioritize", "rank": "prioritize",
            "benchmark": "benchmark", "compare_geo": "compare",
            "compare_temporal": "compare", "weather_normalize": "normalize",
            "eui": "normalize", "per_occupant": "normalize",
            "consumption": "aggregate", "demand": "aggregate",
        }.get(tag, "profile")
        # Intent is first-match and "list" is deliberately broad. Preserve the
        # stronger ranking operator when both occur.
        if re.search(r"\b(top|bottom|highest|lowest|best|worst)\s+"
                     r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                     text):
            task = "prioritize"

        issues = []
        required_inputs = ["catalog"] if task in {
            "discovery", "relationship", "export"
        } else ["electricity"]
        if "weather" in text or tag in {
            "weather_normalize", "weather_relationship", "temperature_response"
        }:
            required_inputs.append("weather")
        if intent.out_of_scope:
            issues.append(f"unavailable data: {intent.out_of_scope}")
        if re.search(r"\b(?:building|site|dataset)\s+[xyzabc]\b", text):
            issues.append("unresolved entity placeholder")
        if re.search(r"\b(second|another) registered dataset\b", text):
            issues.append("second registered dataset is required")
        if re.search(r"\b(circuit[- ]level|submeter|15[- ]minute)\b", text):
            issues.append("requested electricity granularity is unavailable")

        group_by = []
        groups = {
            "site": "site_id", "city": "city", "country": "country",
            "building type": "building_type", "dataset": "dataset_id",
            "year": "year", "month": "month", "day": "day",
            "hour": "hour", "weekday": "weekday", "season": "season",
        }
        for label, field_name in groups.items():
            if re.search(rf"\b(?:by|per|each|across|between)\s+{label}s?\b", text):
                group_by.append(field_name)

        aggregation = None
        for pattern, name in (
            (r"\bhow many\b|\bcount\b", "count"),
            (r"\bmedian\b", "median"),
            (r"\baverage\b|\bmean\b", "mean"),
            (r"\btotal\b|\bsum(?:ming)?\b", "sum"),
            (r"\bmaximum\b|\bhighest\b|\bpeak\b", "max"),
            (r"\bminimum\b|\blowest\b", "min"),
        ):
            if re.search(pattern, text):
                aggregation = name
                break

        ranking = None
        if task == "prioritize":
            ranking = {
                "direction": "ascending" if re.search(r"\b(bottom|lowest|least)\b", text)
                else "descending",
                "limit": intent.entities.get("limit", 10),
            }

        parameters = {}
        horizon = re.search(r"\bnext\s+(\d+)\s*(hour|day)s?\b", text)
        if horizon:
            parameters["horizon"] = int(horizon.group(1)) * (
                24 if horizon.group(2) == "day" else 1)
        window = re.search(r"\brolling\s+(\d+)[ -]day\b", text)
        if window:
            parameters["window_days"] = int(window.group(1))
        base = re.search(r"\bbase(?: temperature)?(?: of| =)\s*(\d+(?:\.\d+)?)", text)
        if base:
            parameters["base_temperature_c"] = float(base.group(1))
        quantile = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*percentile\b", text)
        if quantile:
            parameters["quantile"] = float(quantile.group(1)) / 100
            aggregation = "quantile"
            if "baseload" not in text:
                task = "aggregate"
        missing = re.search(r"\b(?:less than|under|below|at most)\s*"
                            r"(\d+(?:\.\d+)?)\s*%\s*missing\b", text)
        if missing:
            parameters["max_missing_fraction"] = float(missing.group(1)) / 100
        if known_facts is not None:
            prefix = {
                "baseload": "baseload:", "weather_relationship": "weather:",
                "temperature_response": "temperature:", "benchmark": "benchmark:",
            }.get(tag)
            known = known_facts.available(prefix) if prefix else []
            if known:
                parameters["known_methods"] = known

        structural = task in {"discovery", "relationship", "export"}
        explicit = bool(intent.entities.get("building_ids"))
        cohort = bool(group_by or any(key in intent.entities for key in (
            "country", "city", "site_id", "building_type")))
        if structural:
            resolution = "graph_only"
        elif explicit:
            resolution = "compute"
        elif cohort or task in {"compare", "benchmark", "prioritize", "cluster"}:
            resolution = "hybrid"
        else:
            resolution = "compute"

        output_kind = {
            "number": "scalar", "ranking": "ranking", "figure": "figure",
            "map": "map", "table": "table",
        }.get(intent.output_format or "table", "table")
        if group_by and output_kind == "scalar":
            output_kind = "table"
        if task == "forecast":
            output_kind = "forecast"
        elif task == "anomaly":
            output_kind = "anomaly"
        if issues:
            task, output_kind = "refusal", "refusal"

        method = {
            "forecast": "forecast", "anomaly": "anomaly_detection",
            "change_point": "change_point_detection", "cluster": "clustering",
            "similarity": "profile_similarity", "normalize": tag,
            "quality": "data_quality",
        }.get(task, "descriptive")
        measure = ("electricity_consumption" if "electricity" in required_inputs
                   else None)
        return cls(
            task=task, entities=dict(intent.entities), period=dict(intent.time_range),
            measure=measure, aggregation=aggregation, group_by=group_by,
            grain="hourly" if re.search(r"\bhourly\b", text) else None,
            ranking=ranking, method=method, parameters=parameters,
            output={"kind": output_kind}, required_inputs=required_inputs,
            resolution=resolution, issues=issues,
        )

    def validate(self) -> list[str]:
        problems = []
        if self.task not in TASKS:
            problems.append(f"unknown task: {self.task or '<missing>'}")
        if self.resolution not in RESOLUTIONS:
            problems.append(f"unknown resolution: {self.resolution}")
        kind = self.output.get("kind")
        if kind not in OUTPUT_KINDS:
            problems.append(f"unknown output kind: {kind}")
        unknown_groups = sorted(set(self.group_by) - GROUP_FIELDS)
        if unknown_groups:
            problems.append(f"unknown group_by field(s): {', '.join(unknown_groups)}")
        start, end = self.period.get("start"), self.period.get("end")
        if start and end and end <= start:
            problems.append("period end must be after start")
        for name in ("quantile", "max_missing_fraction"):
            value = self.parameters.get(name)
            if value is not None and not 0 <= value <= 1:
                problems.append(f"{name} must be between 0 and 1")
        return problems

    def to_dict(self) -> dict:
        return asdict(self)
