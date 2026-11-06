"""Orchestration, Verification & QA, and the EnergyX entry point.

Execution follows the architecture exactly:

    Interaction & Intent -> Planning -> Orchestration -> specialists
                         -> Decision Support -> Verification & QA -> answer

Orchestration *selects and invokes*; tool bodies are Layer 3 services. The
Verifier runs before any answer is returned and may reject once, forcing a
bounded replan. Budgets on steps, tokens and wall-clock guarantee that
every run terminates, whatever a plan asks for.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date, timedelta
import time
from types import SimpleNamespace
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..access import SnapshotDB
from ..knowledge.curator import KnowledgeCurator
from ..knowledge.extract import extract
from ..knowledge.probe import KnownFacts, probe
from ..knowledge.router import CACHEABLE_TOOLS, QueryRouter
from ..config import PATHS, SNAPSHOT_VERSION, planning_engine as configured_engine
from ..tools import ArtifactStore, Status, ToolContext, load_all, registry
from ..tools.registry import ALTERNATIVE_PARAM, FRACTION_PARAMS
from ..tools.base import RECITABLE_ROWS, ToolResult
from .llm import LLM, NullLLM
from .analysis_spec import AnalysisSpec
from .capability_gate import (SnapshotLimits, looks_like_a_question,
                              unsupported)
from .premises import check as check_premises
from .plan import EmptyUpstream, Plan, PlanStep, resolve_args
from .query_compiler import compile_analysis
from .planner import (  # noqa: F401
    PlannerRefused,
    LATEST_FULL_YEAR, DeterministicPlanner, HierarchicalPlanner, IntentAgent,
    LLMPlanner, Vocabulary, COMPLETENESS_TOOLS, asks_about_completeness,
    asks_about_shape, asks_detector_validation, resolve_percentiles,
    asks_weather_relationship, WEATHER_TOOLS,
    asks_which_are_eligible, ELIGIBILITY_TOOLS, asks_for_conjunction,
    resolve_tail, CONJUNCTION_TOOLS, resolve_relative_threshold, resolve_top_n,
    resolve_self_baseline, resolve_threshold_sweep, resolve_before_after_metrics,
    resolve_named_metrics, resolve_breakout, asks_weekday_weekend_profile,
    asks_spearman, asks_day_count, asks_calendar_day, asks_daily_grain,
    resolve_cluster_methods, resolve_cluster_distance, resolve_n_clusters,
    asks_over_whole_population, resolve_unbroken_hours,
    resolve_weather_grain, resolve_grouped_count, resolve_zero_run,
    resolve_within_group_percentile, resolve_temperature_split,
    resolve_complete_year, resolve_group_statistic,
    resolve_named_statistic, asks_which_set,
    resolve_weather_controls,
    resolve_top_hours_share,
    asks_for_change_point, asks_sensitivity,
    asks_forecast_accuracy, resolve_accuracy_metrics,
    resolve_clock_window,
    resolve_cluster_method,
    resolve_anomaly_method, resolve_anomaly_threshold, resolve_day_type,
    resolve_forecast_models,
    resolve_forecast_model,
    resolve_horizon_hours, resolve_index_components, resolve_measure,
    resolve_moving_window_days, resolve_percentile, resolve_threshold,
    resolve_vintages, resolve_require_different, resolve_gap_hours,
    resolve_step_change, resolve_forecast_origin,
    resolve_season_length, ASKS_PEAK_HOUR, ASKS_CLUSTERING,
    asks_normalised, _calendar_year)
from .state import SessionState
from .trace import Trace


# What each analytic task category is answered *by*, so a plan that fetched
# the measurements and stopped can be told which tool it was missing. The
# arguments come from the question through the same resolvers the deterministic
# planner uses, so a repaired plan and a rule-planned one run the same model on
# the same window -- the repair is not a second, quieter set of rules.
#: Tools that take an eligibility argument, so a population question can ask
#: for the computable cohort rather than the rankable one.
POPULATION_TOOLS = ("calculate_normalized_performance",)

#: Question shapes that have one answering tool, paired with the tools that
#: answer them. Order matters only in that the first match wins, and the
#: shapes are disjoint in practice.
ANSWERING_TOOLS = (
    (asks_about_completeness, COMPLETENESS_TOOLS),
    (asks_weather_relationship, WEATHER_TOOLS),
    (asks_which_are_eligible, ELIGIBILITY_TOOLS),
    (asks_for_conjunction, CONJUNCTION_TOOLS),
)

#: Tools whose metric vocabulary partly belongs to another tool, as
#: ``asking tool -> (owning tool, the parameter that names the metric)``.
METRIC_OWNERS = {
    "calculate_consumption_statistics":
        ("calculate_normalized_performance", "metrics"),
}


#: How a model is written when a person reads it. The registry keys are
#: identifiers -- seasonal_naive, auto_arima -- and an answer that names the
#: winner as `seasonal_naive` has said the right thing in the wrong language.
#: The same slip as reporting a month as "1".
MODEL_DISPLAY_NAMES = {
    "seasonal_naive": "Seasonal Naive", "moving_average": "Moving Average",
    "auto_arima": "AutoARIMA", "arima": "ARIMA", "ets": "ETS",
    "theta": "Theta", "lightgbm": "LightGBM", "linear": "Linear",
    "gradient_boosting": "Gradient Boosting", "chronos": "Chronos",
    "tiny_time_mixer": "Tiny Time Mixer (TTM)", "knn_weather": "KNN (weather)",
    "weather_regression": "Weather Regression",
}


def _model_name(key: str) -> str:
    """A model identifier as a reader writes it."""
    return MODEL_DISPLAY_NAMES.get(str(key), str(key))


ANALYTIC_TOOLS: dict[str, dict] = {
    "9": {
        # A question about a forecast's *error* is answered by the backtest,
        # not by issuing one forecast: the models named have to be scored on
        # the same hours before "which is more accurate" means anything.
        "tool": lambda question: ("backtest_forecast_models"
                                  if asks_forecast_accuracy(question)
                                  else "forecast_timeseries"),
        "target": "building_id", "one": True,
        "tools": ("forecast_timeseries", "evaluate_forecast",
                  "backtest_forecast_models"),
        "what": "a forecast",
        "purpose": "forecast from a frozen cutoff (exclusive)",
        "args": lambda question, intent: (
            {"models": resolve_forecast_models(question) or ["seasonal_naive"],
             "start": (intent.time_range or {}).get("start"),
             "end": (intent.time_range or {}).get("end"),
             "horizon_hours": resolve_horizon_hours(question),
             "metrics": resolve_accuracy_metrics(question)}
            if asks_forecast_accuracy(question) else
            {"model": resolve_forecast_model(question)[0],
             "horizon_hours": resolve_horizon_hours(question),
             "cutoff": (intent.time_range or {}).get("start"),
             **({"window_days": resolve_moving_window_days(question)}
                if resolve_forecast_model(question)[0] == "moving_average"
                else {})}),
    },
    "10": {
        # An anomaly is one odd day; a change point is the date after which
        # the building is different. The question says which, and answering
        # the other is not a near miss -- a step change shows up as no
        # anomalies at all once the new level becomes the norm.
        # The repair table is also a whitelist: a tool absent from "tools" is
        # treated as a planner mistake and rewritten. validate_anomaly_detection
        # and sweep_anomaly_sensitivity were missing, so a correct plan for
        # "does adding change features improve LOF" was overruled back into a
        # flag count -- the runtime undoing the planner's right answer.
        "tool": lambda question: (
            "validate_anomaly_detection" if asks_detector_validation(question)
            else "sweep_anomaly_sensitivity" if asks_sensitivity(question)
            else "detect_change_points" if asks_for_change_point(question)
            else "detect_anomalies"),
        "target": "building_ids", "one": False,
        "tools": ("detect_anomalies", "detect_change_points",
                  "validate_anomaly_detection", "sweep_anomaly_sensitivity"),
        "what": "an anomaly scan or a change point",
        "purpose": "days, or dates, where the building departs from itself",
        "args": lambda question, intent: ({} if (asks_for_change_point(question)
                                                 or asks_detector_validation(question)
                                                 or asks_sensitivity(question))
                                          else {
            "method": resolve_anomaly_method(question)[0],
            "threshold": resolve_anomaly_threshold(
                question, resolve_anomaly_method(question)[0]),
            "start": (intent.time_range or {}).get("start"),
            "end": (intent.time_range or {}).get("end"),
        }),
    },
}


# Tags whose answer is not a quantity of electricity. A plan for one of these
# that ends in a fetch must not be completed with a consumption total: the
# total is a real number about the right building over the right period, and
# the wrong answer to the question.
# A stand-in for a tool the registry does not know, so the agent check below
# reads as "not data discovery" rather than raising on a missing spec.
NO_SPEC = SimpleNamespace(agent=None)

# Tools that publish, beside their summary, the curves the summary was reduced
# from -- and the evidence field naming the artifact. A chart step draws those
# rather than the one-row summary.
# Which parts of a resolved intent a caller may correct, and how each is
# written back. Deliberately a whitelist: an intent is the contract between
# what was asked and what is planned, and a caller that could set any field of
# it could ask for a plan the question does not support.
OVERRIDABLE = ("task_category", "metric", "output_format", "subject",
               "group_by", "entities", "time_range", "requirements")


def apply_intent_overrides(intent, overrides: dict | None) -> dict:
    """Write a caller's corrections onto a resolved intent; report what took.

    Entities and the period merge rather than replace, so correcting the year
    does not silently drop the building the question named. Everything else is
    a straight set. A key that is not overridable, or a value equal to what was
    already resolved, is not reported -- the trace should show what the reader
    changed, not what they confirmed.
    """
    if not overrides:
        return {}
    applied: dict = {}
    for field_name, value in overrides.items():
        if field_name not in OVERRIDABLE or value is None:
            continue
        current = getattr(intent, field_name, None)
        if field_name in ("entities", "time_range"):
            if not isinstance(value, dict):
                continue
            merged = {**(current or {}),
                      **{k: v for k, v in value.items() if v is not None}}
            # An empty string clears a field rather than setting it to "".
            for key, given in value.items():
                if given == "":
                    merged.pop(key, None)
            if field_name == "time_range" and ("start" in value or "end" in value):
                # A period is not just its two dates: `year`, `label`, `months`
                # and the "assumed" note are derived from them, and merging
                # left all four describing the period that was replaced.
                # Correcting the window to 2016 kept year=2017, and every
                # downstream reader that asks for the year -- which is most of
                # them, because an intensity is a per-year figure -- carried
                # on answering for 2017 while the chip claimed otherwise.
                for stale in ("label", "year", "assumed", "months",
                              "spans_all_years", "years", "explicit_dates"):
                    merged.pop(stale, None)
                start, end = merged.get("start"), merged.get("end")
                if start:
                    merged["year"] = int(str(start)[:4])
                    merged["label"] = (str(merged["year"]) if end
                                       and str(end)[:4] == str(merged["year"] + 1)
                                       and str(start)[5:] == "01-01"
                                       else f"{start} to {end or 'the end'}")
                    merged["explicit_dates"] = [d for d in (start, end) if d]
                merged["stated_by_caller"] = True
            if merged != (current or {}):
                setattr(intent, field_name, merged)
                applied[field_name] = merged
        elif value != current:
            setattr(intent, field_name, value)
            applied[field_name] = value
    if applied:
        # A corrected intent is no longer the one a question-shape pattern
        # recognised, and the hybrid planner asks that before choosing a rule
        # route. Saying otherwise would route a corrected question down a rule
        # written for the uncorrected one.
        intent.recognised_shape = False
    return applied


# Does the question draw a line of its own? A bound the question states is
# authoritative; one it does not state belongs to the tool that reasoned about
# its default, not to whichever planner wrote the step.
# "Top-decile", "top 10%", "bottom half" -- a band the question draws over a
# measure, which the measuring tool has to apply.
# "Less than 2% missing", "under 5% gaps" -- a threshold that selects.
MISSING_SHARE = re.compile(
    r"\b(?:less\s+than|under|below|at\s+most|no\s+more\s+than|fewer\s+than)"
    r"\s*(\d{1,2}(?:\.\d+)?)\s*%[^.?]{0,30}\b(?:missing|gaps?|absent|"
    r"incomplete)\b"
    r"|\b(?:missing|gaps?)\b[^.?]{0,24}\b(?:less\s+than|under|below)"
    r"\s*(\d{1,2}(?:\.\d+)?)\s*%", re.I)

TOP_BAND = re.compile(
    r"\b(?:top|highest|upper|bottom|lowest)[-\s]?(?:decile|tenth|quartile|"
    r"quintile|half|\d{1,2}\s*%)\b"
    r"|\b(?:top|bottom)\s+\d{1,2}\s*(?:per\s*cent|%)\b", re.I)

ANOMALY_BOUND = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:standard\s+deviations?|sigma|sds?|x|times)\b"
    r"|\b(?:twice|double|triple|thrice)\b", re.I)

PUBLISHES_PROFILE = {"compare_temporal_profiles": "profile_handle"}

MEASURED_ELSEWHERE = frozenset({
    "weather_relationship", "forecast", "anomaly", "change_point", "cluster",
    "similar", "quality", "entity_path", "semantic_export",
})


@dataclass
class Budget:
    """Hard caps. Exceeding one ends the run with what has been gathered."""

    max_steps: int = 25
    # No wall-clock ceiling by default. A budget that ends a calculation
    # early does not bound a runaway, it decides an answer: the deadline it
    # sets reaches the backtest, which then scores the origins it can afford
    # and reports a figure for a window it did not cover. Pass a finite value
    # only where a caller genuinely needs to stop -- an interactive page, say
    # -- and accept that the answer is then partial and says so.
    max_seconds: float = float("inf")
    max_tokens: int = 200_000
    max_replans: int = 1

    def exceeded(self, trace: Trace, started: float) -> str | None:
        if len(trace.steps) >= self.max_steps:
            return f"step budget exhausted ({self.max_steps})"
        if time.time() - started > self.max_seconds:
            return f"time budget exhausted ({self.max_seconds}s)"
        total = trace.tokens["prompt"] + trace.tokens["completion"]
        if total > self.max_tokens:
            return f"token budget exhausted ({self.max_tokens})"
        return None


@dataclass
class AnswerPayload:
    kind: str
    value: Any
    unit: str | None = None
    fields: list[str] = field(default_factory=list)
    method: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Answer:
    question: str
    text: str
    status: str
    run_id: str
    mode: str
    plan: dict
    evidence: list[dict] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    payload: dict | None = None
    trace_path: str | None = None

    @property
    def refused(self) -> bool:
        return self.status in ("INSUFFICIENT_DATA", "EMPTY")


# Which subject a column identifies. Used to check that an answer is about the
# entity the question named.
SUBJECT_OF_COLUMN = {
    "building_id": "buildings", "stream_id": "buildings",
    "site_id": "sites", "site_name": "sites",
    "city": "cities", "country": "countries",
    "building_type": "building_types", "dataset_id": "datasets",
    "dataset_name": "datasets", "year": "years",
}


# What shape of answer each declared output format needs. A "how many" question
# wants one number; a ranking wants ordered ids. Returning a table handle for
# either is not a wrong number, it is an unanswered question: the value is
# present but the reader has to dig it out.
SHAPE_OF_FORMAT = {"number": "scalar", "ranking": "ranking", "list": "set",
                   "figure": "table", "map": "table", "table": "table",
                   "boolean": "boolean", "entity": "entity"}

# Evidence keys that decide a yes/no question, and the column that carries the
# count behind the verdict. A verdict without its count is an assertion.
VERDICT_KEYS = ("duplicate_count", "dst_duplicate_hours", "anomalies_found",
                "issues_found", "n_events", "n_anomalies")

# Tools that return what they found, so an empty result means "no".
# The conventional cut-off for calling a building weather-sensitive. Stated in
# every answer that uses it; never applied silently.
SENSITIVITY_R = 0.5

# Tools that report on the data rather than answer a question about it. They
# are the right answer to "is this measurable" and never to "what does it show",
# so a plan ending in one is read past unless the question is itself about
# discovery or data quality.
NON_ANSWERING_TOOLS = frozenset({
    "get_data_availability", "profile_data_quality", "check_task_eligibility",
    "create_visualization", "generate_report",
})

# Reporting tools that *are* the answer, not a wrapper around one. The
# walk-back past the reporting agent exists to stop a generated report
# standing in for a count, and it was skipping these too: asked which
# buildings to investigate first, the plan ranked them, recommended ten with
# their drivers, and composition walked past both to hand back the 691-row
# fact table they were computed from -- most of its 24 columns empty, because
# a derived-fact table is sparse by nature. Prioritising *is* the analysis
# that question asks for.
ANSWERING_REPORTS = frozenset({
    "prioritize_entities", "generate_decision_support",
    # An intersection is a finding, not a fan-in. Asked which offices clear
    # both the EUI and the nighttime-baseload 90th percentile, the plan drew
    # each tail and intersected them, and composition walked back past the
    # intersection to one of the tails: 28 buildings reported where 22 clear
    # both. The step that combines the criteria is the one that answered.
    "combine_evidence",
})

# Measures that are per year by definition: an intensity without a year is
# not that intensity, it is a sum over whatever span the data happens to hold.
ANNUAL_METRICS = frozenset({"eui", "per_occupant", "peak_per_area",
                            "peak_power_density"})

# Metrics the performance tool computes, as the intent tag names them. The
# measure resolver reads the words a question uses; the tag reads what it is
# asking for, and for "consumption relative to occupants" only the tag is
# right.
PERFORMANCE_METRICS = frozenset({
    "eui", "per_occupant", "peak_per_area", "peak_power_density",
    "baseload_per_area", "load_factor", "eui_per_operating_hour",
    "peak_to_average_ratio"})

DETECTION_TOOLS = frozenset({
    "detect_anomalies", "detect_data_issues", "detect_change_points",
    "identify_peer_outliers", "find_similar_timeseries",
})


def _verdict_from(result: ToolResult, question: str = "",
                  subjects: list | None = None, store=None):
    """A yes/no answer with the number that decided it.

    Returns ``(answer, basis)`` or ``None``. The basis is always reported, so a
    "No" can be checked rather than trusted.

    ``subjects`` are the entities the question named. A question about one
    building can be answered from a table covering many, and which shape the
    plan happens to produce should not decide whether the reader gets a yes or
    a spreadsheet, so the row for the named subject is picked out first.
    """
    if subjects:
        rows_all = (result.data if isinstance(result.data, list)
                    else (result.preview or []))
        # preview is five rows. A cohort question about one building put that
        # building outside them -- 197 rows of correlations, the named one not
        # among the first five -- so narrowing found nothing, no verdict was
        # reached, and a yes/no question was answered with a table. The whole
        # frame is in the store; read it rather than guess from the sample.
        wanted_ids = {str(x) for x in subjects}
        if (store is not None and result.handle
                and not any(isinstance(r, dict)
                            and wanted_ids & {str(v) for v in r.values()}
                            for r in rows_all)):
            frame = store.get_or_none(result.handle)
            if frame is not None:
                rows_all = frame.to_dict("records")
        wanted = {str(x) for x in subjects}
        for column in ("building_id", "site_id", "stream_id"):
            picked = [r for r in rows_all
                      if isinstance(r, dict) and str(r.get(column)) in wanted]
            if picked and len(picked) < len(rows_all):
                result = replace(result, data=picked, preview=picked,
                                 rows=len(picked))
                break
    for key in VERDICT_KEYS:
        value = result.evidence.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # "No" is only checkable if it says what was looked for. A bare
            # "No -- 0 anomalies" hides whether the test was for spikes, for
            # dips, or at what threshold, which is the whole of the claim.
            basis = {key: value}
            for extra in ("direction", "method", "threshold"):
                if result.evidence.get(extra) is not None:
                    basis[extra] = result.evidence[extra]
            return bool(value), basis
    rows = result.data if isinstance(result.data, list) else (result.preview or [])
    for key in VERDICT_KEYS:
        if rows and key in rows[0]:
            total = sum(r.get(key) or 0 for r in rows)
            return bool(total), {key: total, "rows_examined": len(rows)}
    # "Did X use more in 2017 than in 2016" is decided by comparing the two
    # period columns the comparison produced. Both years are in the row; the
    # question names which is which.
    rows_cmp = result.data if isinstance(result.data, list) else (result.preview or [])
    if rows_cmp and len(rows_cmp) == 1:
        row = rows_cmp[0]
        periods = [c for c in row if re.fullmatch(r"(19|20)\d\d", str(c))
                   and isinstance(row[c], (int, float))]
        named = [p for p in periods if p in (question or "")]
        if len(named) == 2:
            # The later year is the subject: "more in 2017 than in 2016" asks
            # whether 2017 exceeds 2016, whatever order they were written in.
            first = question.index(named[0]) < question.index(named[1])
            subject, against = (named[0], named[1]) if first else (named[1], named[0])
            wants_more = bool(re.search(
                r"\b(more|higher|greater|larger|increase\w*|exceed\w*)\b",
                (question or "").lower()))
            bigger = row[subject] > row[against]
            return (bigger if wants_more else not bigger,
                    {subject: round(float(row[subject]), 2),
                     against: round(float(row[against]), 2),
                     "comparison": f"{subject} vs {against}"})

    # A correlation answers "is X sensitive to temperature", but only against a
    # threshold -- and a yes/no question demands one. State the convention in
    # the answer rather than applying it silently, so a reader who uses a
    # different cut-off can see it and disagree.
    rows_now = result.data if isinstance(result.data, list) else (result.preview or [])
    if rows_now and len(rows_now) == 1 and "correlation" in rows_now[0]:
        r_value = rows_now[0].get("correlation")
        if isinstance(r_value, (int, float)):
            return (abs(r_value) >= SENSITIVITY_R,
                    {"correlation": round(float(r_value), 3),
                     "threshold": f"|r| >= {SENSITIVITY_R}"})

    # Row count is only a verdict for a tool whose job is to *find* things. For
    # anything else it counts descriptions, not findings: a one-row data-quality
    # record for a building was read as "Yes, duplicates exist" when the true
    # duplicate count is zero. A wrong Yes is worse than no verdict.
    if result.provenance.get("tool") in DETECTION_TOOLS:
        count = result.rows if result.rows is not None else len(rows)
        return bool(count), {"matching_rows": count}
    return None

# Evidence keys that carry a true count, in order of preference: what matched
# the filter before any limit, then what the tool returned.
COUNT_KEYS = ("matched_filter", "matched", "n_buildings", "n_rows", "count")

# Columns a tool carries for context, never the answer to a measure question.
CONTEXT_COLUMNS = frozenset({
    "rank", "year", "n_observations", "expected_observations", "floor_area_m2",
    "occupants", "year_built", "utc_offset_hours", "resolution_minutes",
    "n_rows_present", "n_zero_masked",
})


def _measure_column(row: dict, columns, text: str,
                    require_named: bool = False) -> str | None:
    """The numeric column the question is asking about, if it names one."""
    literal = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    asked = literal
    for word, to in SYNONYMS:
        asked = re.sub(rf"\b{word}\b", to, asked)

    def spoken(column: str) -> str:
        return re.sub(r"[ _](m2|kwh|kw|pct|percent|frac(tion)?)$", "",
                      column.replace("_", " ")).strip()

    numeric = [c for c in (columns or list(row))
               if isinstance(row.get(c), (int, float))
               and not isinstance(row.get(c), bool)
               and c not in CONTEXT_COLUMNS and spoken(c)]
    named = [c for c in numeric if spoken(c) in literal or spoken(c) in asked]
    if named:
        return max(named, key=len)

    def canonical(phrase: str) -> set[str]:
        for word, to in SYNONYMS:
            phrase = re.sub(rf"\b{word}\b", to, phrase)
        return set(phrase.split())

    words = canonical(literal)
    matched = [(c, canonical(spoken(c))) for c in numeric]
    matched = [(c, t) for c, t in matched if t and t <= words]
    if matched:
        return max(matched, key=lambda ct: (len(ct[1]), len(ct[0])))[0]
    if require_named:
        return None
    return numeric[-1] if len(numeric) == 1 else None


# Which way a superlative points. "Highest" wants the largest row, "lowest" the
# smallest; a question with neither is not asking for an extreme at all.
SUPERLATIVE_MAX = re.compile(
    r"\b(highest|largest|greatest|most|maximum|max|top|biggest|peak|"
    r"strongest|longest)\b")
SUPERLATIVE_MIN = re.compile(
    r"\b(lowest|smallest|least|minimum|min|bottom|weakest|shortest)\b")


# Words a question and a column can use for the same statistic. Applied to both
# sides, so "maximum demand" and peak_demand meet in the middle.
SYNONYMS = (("average", "mean"), ("avg", "mean"), ("typical", "mean"),
            ("highest", "max"), ("largest", "max"), ("greatest", "max"),
            ("maximum", "max"), ("peak", "max"), ("lowest", "min"),
            ("smallest", "min"), ("minimum", "min"), ("middle", "median"))


def _scalar_from(result: ToolResult, text: str, subject: str | None = None,
                 counting: bool = True):
    """The single number a counting question is asking for.

    Prefers the count the tool recorded over the rows it returned, because a
    limit truncates rows but not the answer.
    """
    # A single row carrying a count of the thing asked about *is* the answer.
    # "How many total buildings are located at site Bear" returned one site row
    # whose `buildings` column reads 92, and the row count -- 1 -- was reported
    # instead.
    rows0 = result.data if isinstance(result.data, list) else (result.preview or [])
    if subject and len(rows0) == 1:
        for column, value in rows0[0].items():
            if (column.endswith(subject) or column == subject) and isinstance(
                    value, (int, float)) and not isinstance(value, bool):
                return value

    # A count key answers "how many". It must not answer "what is the peak
    # demand": that question returned 1 -- the number of buildings examined --
    # where the peak is 83.0 kW.
    if counting:
        for key in COUNT_KEYS:
            value = result.evidence.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    if isinstance(result.data, dict):
        numbers = [v for v in result.data.values()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numbers) == 1:
            return numbers[0]
    # A single-row frame is a scalar answer -- but which column? The tool
    # records the measure it computed, so prefer that over guessing among the
    # context columns it carried along (floor_area, occupants, rank).
    rows = result.data if isinstance(result.data, list) else result.preview
    if rows and len(rows) == 1:
        row = rows[0]
        measure = result.evidence.get("metric") or result.evidence.get("measure")
        if measure and isinstance(row.get(measure), (int, float)):
            return row[measure]
        # The question names the measure it wants. "What is the peak demand of
        # X" against a row carrying peak_demand, mean_demand, load_factor and
        # peak_to_average_ratio has four numbers and no evidence key to choose
        # between them; the column the question actually named is the answer
        #. Longest match first, so "peak demand" is not answered by
        # "demand".
        # The question and the column use different words for the same
        # statistic: "average air temperature" against mean_air_temperature,
        # "highest wind speed" against max_wind_speed.
        # Match against both the question's own words and a normalised form:
        # "peak demand" is a column name, while "peak" also means max elsewhere,
        # so replacing it outright broke peak_demand.
        literal = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        asked = literal
        for word, to in SYNONYMS:
            asked = re.sub(rf"\b{word}\b", to, asked)
        def spoken(column: str) -> str:
            # floor_area_m2 is asked for as "floor area"; kwh/kw/pct suffixes
            # are units, not part of the name the question uses.
            return re.sub(r"[ _](m2|kwh|kw|pct|percent|frac(tion)?)$", "",
                          column.replace("_", " ")).strip()

        numeric = [c for c in row
                   if isinstance(row[c], (int, float))
                   and not isinstance(row[c], bool) and spoken(c)]
        named = [c for c in numeric
                 if spoken(c) in literal or spoken(c) in asked]
        if named:
            return row[max(named, key=lambda c: len(c))]

        # The question and the column agree on the statistic but not on the
        # wording or the word order: "maximum hourly demand" against
        # peak_demand normalises to "max ... demand" on one side and
        # "max demand" on the other, which no substring test can join. Compare
        # the words themselves, normalising both sides the same way, and take
        # the column whose words the question all used -- the most specific
        # first, so peak_demand wins over a bare demand column.
        def canonical(phrase: str) -> set[str]:
            for word, to in SYNONYMS:
                phrase = re.sub(rf"\b{word}\b", to, phrase)
            return set(phrase.split())
        words = canonical(literal)
        matched = [(c, canonical(spoken(c))) for c in numeric]
        matched = [(c, t) for c, t in matched if t and t <= words]
        if matched:
            best = max(matched, key=lambda ct: (len(ct[1]), len(ct[0])))
            return row[best[0]]
        numbers = [v for k, v in row.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in CONTEXT_COLUMNS]
        if len(numbers) == 1:
            return numbers[0]
    # The row count answers "how many", never "what is the peak demand": that
    # returned 1 -- one building's row -- where the peak is 83.0 kW.
    if counting and result.rows is not None:
        return result.rows
    return None


def wants_document_shape(intent) -> bool:
    """Whether the question asked to be handed a document rather than a figure.

    A chart or report question is answered by the artifact, so promoting one of
    its numbers to the answer would replace the deliverable with a detail.
    """
    return bool(intent is not None and intent.output_format in
                ("figure", "map", "report"))


def named_measure(result: ToolResult, text: str) -> tuple[str, float] | None:
    """The column of a one-row result that the question named, if it named one.

    `_scalar_from` ends in fallbacks -- the only number in the row, the row
    count -- which are right when a scalar is what the question asked for and
    wrong as evidence that a table *is* a scalar. This is the strict half: a
    column whose own name the question used, and nothing else.

    It exists because a question can ask for a table and still have one number
    as its answer. "Calculate the mean and median demand for all healthcare
    buildings" is listed as a table question, and the tool pools the cohort
    into a single row; handing that row back leaves the reader to find the mean
    among fifteen columns, which is the difference between an answer and a
    spreadsheet.
    """
    rows = result.data if isinstance(result.data, list) else result.preview
    if not rows or len(rows) != 1:
        return None
    # A question that asks for a breakdown or a comparison is asking for more
    # than one number, and one row is not a reason to give it one. "Compare
    # Bull_office_Anne's yearly electricity totals" names "totals" and wants
    # them year by year; promoting the single row it came back with would
    # state one year's figure as the answer to a question about two.
    if re.search(r"\beach\b|\bper\s+\w+|\bby\s+(year|month|week|day|site|city|"
                 r"country|building|type)\b|\byearly\b|\bmonthly\b|\bweekly\b"
                 r"|\bbreakdown\b|\bover\s+time\b|\btrend\w*|\bcompare[ds]?\b"
                 r"|\bversus\b|\bvs\.?\b|\bagainst\b",
                 (text or "").lower()):
        return None
    row = rows[0]
    literal = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    asked = literal
    for word, to in SYNONYMS:
        asked = re.sub(rf"\b{word}\b", to, asked)

    def spoken(column: str) -> str:
        return re.sub(r"[ _](m2|kwh|kw|pct|percent|frac(tion)?)$", "",
                      column.replace("_", " ")).strip()

    numeric = [c for c in row
               if isinstance(row[c], (int, float)) and not isinstance(row[c], bool)
               and spoken(c) and c not in CONTEXT_COLUMNS]
    named = [c for c in numeric if spoken(c) in literal or spoken(c) in asked]
    # Two names for the same measure are one measure: "the daily load factor"
    # matches both load_factor and daily_load_factor, and the specific one is
    # what was asked for. Two names for different measures are two questions:
    # "the mean and median demand" names both, and answering with either is
    # answering half while looking complete -- it picked the median where the
    # reference pins the mean. So: drop the columns another named column
    # spells out more fully, and promote only what is left alone.
    specific = [c for c in named
                if not any(other != c and spoken(c) in spoken(other)
                           for other in named)]
    if len(specific) != 1:
        return None
    return specific[0], row[specific[0]]


PERCENTAGE_ASKED = re.compile(r"\bpercentage\b|\bpercent\b|\bwhat\s+share\b|%")


def _as_percentage(value, result: ToolResult, question: str):
    """Scale a fraction when the question asked for a percentage.

    A fraction reported where a percentage was asked for is wrong by a
    hundredfold to anyone reading it. Only columns whose name says they are
    fractions are scaled, so a value that is already a percentage is left
    alone.
    """
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return value, None
    if not PERCENTAGE_ASKED.search((question or "").lower()):
        return value, None
    if not 0.0 <= float(value) <= 1.0:
        return value, None                 # already a percentage, or not a share
    rows = result.data if isinstance(result.data, list) else (result.preview or [])
    named_fraction = any(
        str(column).endswith(("_fraction", "_frac", "_ratio", "_share"))
        and isinstance(row.get(column), (int, float))
        and abs(float(row[column]) - float(value)) < 1e-12
        for row in rows[:1] for column in row)
    if not named_fraction:
        return value, None
    return round(float(value) * 100, 6), "%"


def _lead_with_the_answer(text: str, payload: dict, intent=None) -> str:
    """Put the answer first, when the payload holds one.

    The number is computed and typed before this runs, and the prose still
    opened with "Answer (search_catalog):" and a table for the reader to find
    it in. A question with a direct answer should read as one sentence, with
    the table underneath as support rather than as the answer itself.
    """
    kind, value = payload.get("kind"), payload.get("value")
    unit = payload.get("unit")
    if kind == "scalar" and isinstance(value, (int, float, str)):
        if isinstance(value, float):
            shown = f"{value:,.4g}"
        elif isinstance(value, int):
            shown = f"{value:,}"
        else:
            shown = str(value)
        # A unit belongs to a measurement, not to a name: the answer to
        # "which category" read "Healthcare kWh".
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        lead = f"{shown}{f' {unit}' if unit and numeric else ''}"
        # A summarising number means little without saying what it measures.
        label = ((payload.get("evidence") or {}).get("headline") or {}).get("label")
        if label:
            lead += f" — {label}"
    elif kind == "boolean" and isinstance(value, dict) and "answer" in value:
        # Composition already opens with the verdict, so prepending another
        # printed "No -- n anomalies 0" twice over.
        if text.lstrip().startswith(("Yes", "No")):
            return text
        lead = "Yes" if value["answer"] else "No"
        basis = value.get("basis")
        if isinstance(basis, dict):
            # The basis is the evidence a reader needs to check the verdict, so
            # it has to be readable: a raw mapping printed beside "No" says
            # nothing to anyone who did not write it.
            basis = ", ".join(f"{k.replace('_', ' ')} {v}"
                              for k, v in list(basis.items())[:3])
        if basis:
            lead += f" -- {basis}"
    elif kind == "set" and isinstance(value, list) and value and all(
            not isinstance(v, (dict, list)) for v in value):
        shown = ", ".join(str(v) for v in value[:12])
        lead = shown + (f", and {len(value) - 12:,} more" if len(value) > 12 else "")
    else:
        return text
    if not text:
        return lead
    # The heading states the count with its noun, so a bare number above it
    # repeats. A lead carrying a unit or a label says more, and stays.
    body = text.lstrip()
    first = body.split("\n", 1)[0]
    if lead.strip() and first.startswith(lead.strip() + " "):
        return text
    # "5 buildings" over "5 rows:" counts the same thing twice.
    generic = re.match(r"^([\d,]+) rows?:?$", first.strip())
    if generic and lead.strip().startswith(generic.group(1)):
        rest = body.split("\n", 1)
        return f"{lead}:\n{rest[1]}" if len(rest) > 1 else lead
    # The lead is the answer; what follows explains it.
    return f"{lead}\n\n{text}"


def _answer_payload(spec, plan: Plan, results: dict[str, ToolResult],
                    text: str, status: str, intent=None,
                    store=None) -> AnswerPayload:
    if status == "INSUFFICIENT_DATA":
        return AnswerPayload("refusal", {"status": status, "text": text},
                             method="refusal")

    # An empty result answers a counting question with zero and a yes/no
    # question with No. The text already said so; the typed payload was still
    # reporting a table, so a caller reading the payload saw no answer at all
    #.
    empty_result = any(r.status is Status.EMPTY for r in results.values()) \
        if results else False
    # Zero is the answer to "how many matched". It is not the answer to "what
    # was the maximum demand": a maximum over nothing is unknown, not nought,
    # and a confident 0 reads as a reading.
    counts = bool(re.search(
        r"\bhow many\b|\bcount\b|\btally\b|\bnumber of\b|\btotal number\b",
        (intent.question if intent is not None else "").lower()))
    if empty_result and intent is not None:
        if intent.output_format == "number" and counts and not any(
                r.ok and (r.rows or r.data) for r in results.values()):
            return AnswerPayload("scalar", 0, method="empty_is_zero",
                                 evidence={"matched": 0})
        if intent.output_format == "boolean" and not any(
                r.ok and (r.rows or r.data) for r in results.values()):
            return AnswerPayload("boolean",
                                 {"answer": False, "basis": {"matching_rows": 0}},
                                 method="empty_is_no", evidence={"matched": 0})
    result = results.get(plan.answer_step) if plan.answer_step else None
    if result is None and results:
        result = list(results.values())[-1]

    # A report wraps work already done, so a question wanting a count, a list
    # or a table is answered from the last step that computed something. A
    # figure, a map or a report is itself the deliverable; a text answer is
    # not, and wants the name rather than a document containing it.
    wants_document = intent is not None and intent.output_format in (
        "figure", "map", "report")
    # ...but only a document that exists. A chart step that refused, because
    # the frame it was handed summarises profiles rather than holding one, is
    # not an answer to anything, and treating it as one buried the correlation
    # the question actually asked for behind "cannot answer".
    if wants_document and result is not None and not result.ok:
        wants_document = False
    # ...and a *chart* is never the answer's content. The figure reaches the
    # caller on its own field, so typing the answer as the path to a PNG
    # throws away the number the chart was drawn from: "compare the weekday
    # and weekend load profile for site Fox" answered with a file path where
    # the ratio is 1.1112. A generated report is different -- it is the
    # deliverable and has no number behind it -- so only the chart walks back.
    drew_chart = (result is not None
                  and result.provenance.get("tool") == "create_visualization")
    if drew_chart:
        wants_document = False
    # Prioritising and recommending *are* the answer to "which should be
    # looked at first", so the payload stops at them for the same reason the
    # text does -- otherwise the typed answer is a handle to the sparse fact
    # table the ranking was computed from.
    if (result is not None
            and result.provenance.get("tool") in ANSWERING_REPORTS):
        pass
    elif (result is not None and not wants_document
            and result.provenance.get("agent") == "reporting"):
        usable = [candidate for candidate in reversed(list(results.values()))
                  if (candidate.provenance.get("tool") in ANSWERING_REPORTS
                      or candidate.provenance.get("agent") != "reporting")
                  and candidate.ok and (candidate.rows or candidate.data)]
        # ...and past a lookup that is about something else. Asked which usage
        # category is most variable, the plan computed statistics per category,
        # then read the buildings' metadata, then wrote a report: walking back
        # to the last step that computed anything landed on the metadata, whose
        # rows are buildings. The subject the question is about says which
        # result is the answer.
        # ...but the subject is a tie-breaker among results that computed
        # something, never a licence to prefer the step that found the cohort.
        # "Compare the weekday and weekend load profile for site Fox" has
        # subject *sites*, and the only result carrying a site_id column was
        # find_buildings: the comparison was made, and the answer handed back
        # was a 137-row catalog listing of the site it was made over.
        computed = [candidate for candidate in usable
                    if (registry.REGISTRY.get(candidate.provenance.get("tool"))
                        or NO_SPEC).agent != "data_discovery"]
        pool = computed or usable
        about = [candidate for candidate in pool
                 if intent is not None and intent.subject
                 and any(SUBJECT_OF_COLUMN.get(column) == intent.subject
                         for column in (candidate.columns or []))]
        if about or pool:
            result = (about or pool)[0]
    if result is None:
        return AnswerPayload("refusal", {"status": status, "text": text},
                             method="no_result")
    # A tool that computed one summarising number says so in evidence, and
    # then that number is the answer: the frame it came from is support. Before
    # this, "quantify the impact of missing values on annual EUI" typed its
    # answer as a 1,547-row table with the figure nowhere in it.
    # "Which buildings ..." is answered by a set, and the size of the set is
    # the finding: "285 buildings have...", "0 buildings...", "68 office
    # buildings...". Answered from a frame with no headline the reply reads
    # "Answer (rank_buildings): 217 rows", which describes the output
    # rather than the portfolio. A tool that already declared its own headline
    # keeps it -- it knows better what it computed.
    wanted_set = asks_which_set(plan.question)
    # A tool's own headline is usually the better answer -- it knows what it
    # computed -- but it does not know what was asked. For "which buildings are
    # eligible", calculate_normalized_performance headlines the median EUI of
    # the set, which is a fact about the set and not the answer to a question
    # about its membership. The size wins unless the headline is itself a count
    # of the same thing.
    own = (result.evidence or {}).get("headline") or {}
    headline_counts = str(own.get("unit") or "").rstrip("s").lower() in (
        "building", "site", "stream", "meter")
    if (wanted_set and not wants_document
            and not headline_counts
            and result.rows is not None):
        return AnswerPayload(
            "scalar", int(result.rows), method="set_size",
            unit=wanted_set, evidence=dict(result.evidence or {}))

    headline = (result.evidence or {}).get("headline")
    if (isinstance(headline, dict) and "value" in headline
            and isinstance(headline["value"], (int, float))
            and not isinstance(headline["value"], bool)
            and not wants_document
            # A yes/no question is answered yes or no. The comparison tools
            # declare their ratio as a headline, which is right for "how does
            # A compare with B" and wrong for "did A use more than B": the
            # verdict was already computed and the ratio replaced it with
            # 0.9911.
            and (intent is None or intent.output_format != "boolean")):
        return AnswerPayload("scalar", headline["value"],
                             method="tool_headline",
                             unit=headline.get("unit"),
                             evidence=dict(result.evidence))
    if plan.answer_field == "rows":
        value = result.rows
    elif plan.answer_field and isinstance(result.data, dict):
        value = result.data.get(plan.answer_field)
    elif plan.answer_field and isinstance(result.data, list):
        values = [row[plan.answer_field] for row in result.data
                  if plan.answer_field in row]
        value = values[0] if len(values) == 1 else values
    elif result.data is not None:
        value = result.data
    elif result.handle:
        value = {"handle": result.handle}
    else:
        value = result.preview or text
    evidence = dict(result.evidence)

    # Honour the shape the question asked for. The intent already resolves it
    # and nothing downstream was reading it.
    kind = spec.output["kind"] if spec else "table"
    if spec is None and intent is not None:
        wanted = SHAPE_OF_FORMAT.get(intent.output_format or "", "table")
        if wanted == "boolean":
            verdict = _verdict_from(result, intent.question or "",
                                    (intent.entities or {}).get("building_ids"),
                                    store=store)
            if verdict is not None:
                answer, basis = verdict
                kind, value = "boolean", {"answer": answer, "basis": basis}
        elif wanted == "scalar":
            # "How many" wants a count; "what is the peak demand" wants the
            # measure, and a count key would answer neither honestly.
            counting = bool(re.search(
                r"\bhow many\b|\bcount\b|\btally\b|\bnumber of\b"
                r"|\bhow much\b|\btotal number\b",
                (intent.question or "").lower()))
            scalar = _scalar_from(result, intent.question or text,
                                  intent.subject, counting)
            scalar, unit = _as_percentage(scalar, result,
                                          intent.question or text)
            if scalar is not None:
                kind, value = "scalar", scalar
                if unit:
                    evidence = {**evidence, "unit": unit}
        elif wanted == "entity":
            # One named thing. The row carrying it is already the answer; what
            # was missing was saying which column names it, so a correct result
            # came back as a table and was read as its own row count.
            #
            # Only the first row of an *ordered* result is the winner. Taking
            # row one of whatever arrived turned three right answers into three
            # confidently wrong ones -- the first building alphabetically, and
            # London where the answer was Washington DC. So: one row, or a rank
            # column that says the order is meant. Anything else stays a table,
            # which is unhelpful but honest.
            rows = value if isinstance(value, list) else (
                result.data if isinstance(result.data, list) else result.preview)
            if rows and isinstance(rows[0], dict):
                ranked = "rank" in rows[0]
                row = None
                if len(rows) == 1:
                    row = rows[0]
                elif ranked:
                    row = min(rows, key=lambda r: r.get("rank", 0))
                # Nothing else, deliberately. Taking the extreme row of an
                # unordered result is right only when the table computes the
                # quantity that was asked for, and twice it did not: "minimum
                # demand on January 1" was fetched at daily resolution, so the
                # extreme was the smallest daily total rather than the smallest
                # hourly demand, and the answer was a confident wrong building.
                # A table is unhelpful; a wrong name presented as the answer is
                # worse. These belong to the plan, not to composition.
                if row is not None:
                    # A calendar unit is the answer as a number, not as a
                    # named thing. "At what local hour does X peak" resolves
                    # to an entity question because it asks "what ... hour",
                    # and the entity branch went looking for an identifier
                    # column: the answer came back as the building's own name.
                    calendar = next(
                        (column for word, column in
                         (("hour", "peak_hour"), ("hour", "hour_of_day"),
                          ("month", "month"), ("day", "day_of_week"),
                          ("season", "season"), ("year", "year"))
                         if re.search(rf"\b{word}\b", (intent.question or "").lower())
                         and column in row
                         and isinstance(row[column], (int, float))
                         and not isinstance(row[column], bool)), None)
                    if calendar:
                        kind, value = "scalar", row[calendar]
                        evidence = {**evidence, "answered_column": calendar}
                        return AnswerPayload(
                            kind=kind, value=value, unit=evidence.get("unit"),
                            fields=result.columns or [],
                            method=(spec.method if spec else
                                    evidence.get("method")
                                    or result.provenance.get("tool", "legacy")),
                            evidence=evidence)
                    # The subject has to be known for this match to mean
                    # anything: .get() returns None for a column that is not an
                    # identifier at all, so an unresolved subject matched the
                    # first such column. "Which category has the highest
                    # variability" answered 1 -- the rank -- off a row that
                    # named Healthcare two columns along.
                    column = next(
                        (c for c in (result.columns or list(row))
                         if intent.subject
                         and SUBJECT_OF_COLUMN.get(c) == intent.subject), None)
                    column = column or next(
                        (c for c in (result.columns or list(row))
                         if c in SUBJECT_OF_COLUMN), None)
                    if column and column in row:
                        kind, value = "scalar", row[column]
        elif wanted == "table" and not wants_document_shape(intent):
            # A table question can still have one number as its answer. When a
            # tool pools a cohort into a single row and the question named one
            # of that row's columns -- "the mean and median demand for all
            # healthcare buildings", "the daily load factor for site Peacock"
            # -- the named column is the answer, and handing back the row
            # leaves the reader to find it among fifteen others. The table
            # stays in the response; what changes is that the answer is said.
            found = named_measure(result, intent.question or text)
            if found is not None:
                column, measured = found
                kind, value = "scalar", measured
                evidence = {**evidence, "answered_column": column}
        elif wanted in ("ranking", "set"):
            kind = wanted
            if not isinstance(value, list):
                value = result.data if isinstance(result.data, list) else (
                    {"handle": result.handle} if result.handle else value)
            # A set is its members. Rows carrying context columns beside the
            # identifier answer "which years does this cover" with a table of
            # statistics, leaving the reader to pick the years out of it. The
            # subject already names the column that identifies a member.
            if (wanted == "set" and isinstance(value, list) and value
                    and isinstance(value[0], dict)):
                column = next(
                    (c for c in (result.columns or list(value[0]))
                     if intent.subject
                     and SUBJECT_OF_COLUMN.get(c) == intent.subject), None)
                if column:
                    value = [row[column] for row in value if column in row]

    return AnswerPayload(
        kind=kind,
        value=value, unit=evidence.get("unit"), fields=result.columns or [],
        method=(spec.method if spec else evidence.get("method") or
                result.provenance.get("tool", "legacy")),
        evidence=evidence,
    )


class VerificationAgent:
    """Validate a completed run before its answer is returned.

    Checks the things that make an energy answer wrong in ways that look right:
    whether anything was actually computed, whether coverage and timezone were
    recorded, whether a claim outran its evidence, and whether a refusal is
    being dressed up as a result.
    """

    name = "verification"

    def verify(self, plan: Plan, results: dict[str, ToolResult],
               state: SessionState, intent=None) -> dict:
        checks: list[dict] = []
        executed = [s for s in plan.steps if s.id in results]
        terminal = executed[-1] if executed else None
        final = results.get(terminal.id) if terminal else None

        def check(name: str, passed: bool, detail: str = "") -> None:
            checks.append({"check": name, "passed": bool(passed), "detail": detail})

        # The answer has to be about what was asked. Every other check here is
        # about execution health, which is why a question about buildings could
        # be answered with the datasets catalog, and a "how many" with a list,
        # and still pass all nine.
        if intent is not None and intent.subject and final is not None and final.ok:
            produced = set()
            for result in results.values():
                if not result.ok:
                    continue
                entity = result.evidence.get("entity")
                if entity:
                    produced.add(entity)
                for column in (result.columns or []):
                    produced.add(SUBJECT_OF_COLUMN.get(column, ""))
            produced.discard("")
            check("the answer is about what was asked",
                  intent.subject in produced or not produced,
                  f"asked about {intent.subject}; answered with "
                  f"{sorted(produced) or 'no identifiable entity'}")

        check("plan produced at least one step", bool(plan.steps))
        check("every planned step executed",
              len(executed) == len(plan.steps),
              f"{len(executed)}/{len(plan.steps)}")
        check("no tool raised",
              not any(r.status is Status.ERROR for r in results.values()),
              ", ".join(f"{k}: {v.message}" for k, v in results.items()
                        if v.status is Status.ERROR)[:200])
        if final is not None:
            check("final step produced a result", final.ok or final.status in (
                Status.EMPTY, Status.INSUFFICIENT_DATA),
                  final.status.value)

        # Provenance: everything must be attributable to the frozen snapshot.
        snapshots = {r.provenance.get("snapshot") for r in results.values()
                     if r.provenance.get("snapshot")}
        check("single snapshot throughout", len(snapshots) <= 1, str(sorted(snapshots)))

        # Coverage discipline: any normalised or ranked claim must say what
        # coverage floor produced it.
        # These only apply to results that actually produced numbers; a refusal
        # has nothing to state a coverage floor about.
        normalising = [r for r in results.values()
                       if r.ok and r.provenance.get("tool") in
                       {"calculate_normalized_performance", "rank_buildings",
                        "benchmark_building", "identify_peer_outliers"}]
        if normalising:
            check("normalised results state their coverage floor",
                  all("min_coverage" in r.evidence for r in normalising))

        # Local-time discipline for anything shaped by time of day.
        temporal = [r for r in results.values()
                    if r.ok and r.provenance.get("tool") in
                    {"calculate_load_profile", "analyze_operating_pattern",
                     "analyze_spatiotemporal_patterns"}]
        if temporal:
            check("temporal results declare their timezone",
                  all("timezone" in r.evidence or "local" in str(r.evidence).lower()
                      for r in temporal))

        # A report assembled from a step that returned nothing is not an
        # answer, however confident it reads.
        if terminal is not None and final is not None and final.ok:
            upstream = set(terminal.depends_on) | Plan.references(terminal)
            hollow = [sid for sid in upstream
                      if sid in results and results[sid].status in
                      (Status.EMPTY, Status.INSUFFICIENT_DATA)]
            check("final answer rests on non-empty evidence", not hollow,
                  f"empty upstream: {hollow}" if hollow else "")

        # Strategy discipline: a plan that claims graph_only for a measurement
        # scan is wrong even when the answer happens to be right.
        router = QueryRouter()
        routings = [router.route(s.tool, s.resolution)
                    for s in plan.steps if s.id in results]
        if routings:
            check("plan strategies match the engines used",
                  all(r.consistent for r in routings),
                  ", ".join(f"{r.tool}: declared {r.declared_resolution}, "
                            f"engine {r.engine}" for r in routings
                            if not r.consistent)[:180])

        # Graph answers are cross-checked against the catalog wherever they
        # overlap: the graph is a projection, so a disagreement is a graph bug
        #, never a catalog one.
        graph_steps = [s for s in plan.steps
                       if router.engine_for(s.tool) == "graph" and s.id in results]
        if graph_steps:
            check("graph results carry the active snapshot",
                  all(results[s.id].provenance.get("snapshot") in
                      (None, state.snapshot_version) for s in graph_steps)
                  if hasattr(state, "snapshot_version") else True)

        refusals = [r for r in results.values()
                    if r.status is Status.INSUFFICIENT_DATA]
        check("refusals name a predicate",
              all(r.predicate for r in refusals),
              f"{len(refusals)} refusal(s)")

        failed = [c for c in checks if not c["passed"]]
        status = ("ok" if not failed else
                  "reject" if any(c["check"] in
                                  ("no tool raised", "final step produced a result",
                                   "final answer rests on non-empty evidence")
                                  for c in failed) else "warn")
        return {"status": status, "checks": checks,
                "failed": [c["check"] for c in failed],
                "verifier": self.name}


class Orchestrator:
    """Execute a validated plan DAG, honouring dependencies and budgets."""

    name = "orchestration"

    def __init__(self, ctx: ToolContext, mode: str = "multi", intent=None):
        self.ctx = ctx
        self.mode = mode
        self.intent = intent
        self.router = QueryRouter()

    def run(self, plan: Plan, state: SessionState, trace: Trace,
            budget: Budget, started: float, attempt: int = 1
            ) -> tuple[dict[str, ToolResult], str | None]:
        results: dict[str, ToolResult] = {}
        for step in plan.order():
            stop = budget.exceeded(trace, started)
            if stop:
                trace.note(f"halted before {step.id}: {stop}")
                return results, stop

            # In multi mode a step may only use its own agent's tools.
            if self.mode == "multi" and step.agent:
                allowed = {s.name for s in registry.visible_to("multi", step.agent)}
                if step.tool not in allowed:
                    trace.note(f"{step.id}: {step.tool} not visible to {step.agent}")

            # Repair a cohort argument mechanically, before resolution turns
            # "$1.handle" into a plain string that no longer looks like a
            # reference. A cohort that is a handle reference, a literal naming
            # no building, or missing entirely is filled from the step that
            # resolved one. Only for steps that declare a cohort parameter --
            # find_buildings produces one and takes none -- and under every
            # name a cohort goes by, group_a and group_b included.
            declared_cohorts = ((registry.REGISTRY.get(step.tool).parameters or {})
                                .get("properties", {})
                                if registry.REGISTRY.get(step.tool) else {})
            cohort_sources = [earlier.id for earlier in plan.order()
                              if earlier.id != step.id
                              and earlier.tool == "find_buildings"
                              and results.get(earlier.id) is not None
                              and results[earlier.id].ok]
            for position, name in enumerate(("building_ids", "group_a", "group_b",
                                             "peer_ids", "candidate_ids")):
                if name not in declared_cohorts:
                    continue
                literal = step.args.get(name)
                missing = literal is None or (isinstance(literal, list)
                                              and not literal)
                if not (missing or (isinstance(literal, list)
                                    and all(isinstance(b, str) for b in literal))):
                    continue
                known = set() if missing else set(self.ctx.db.df(
                    "SELECT building_id FROM buildings WHERE building_id IN "
                    "(SELECT unnest(?))", [list(literal)])["building_id"])
                if known:
                    continue
                # Two groups take the two cohorts the plan resolved, in the
                # order it resolved them; one group takes the first.
                wanted = (1 if name == "group_b" and len(cohort_sources) > 1
                          else 0)
                source = (cohort_sources[wanted] if len(cohort_sources) > wanted
                          else None)
                if not source:
                    continue
                # A step that names its own attributes is resolving its own
                # cohort, not borrowing one: filling find_buildings(
                # building_type="Education") from the offices resolved a step
                # earlier intersects the two and matches nobody. "How does EUI
                # differ between offices and education buildings" came back as
                # "nothing matched the question's filters".
                if step.tool == "find_buildings" and any(
                        step.args.get(k) for k in
                        ("building_type", "building_subtype", "country",
                         "countries", "city", "site_id", "industry",
                         "subindustry")):
                    step.args.pop(name, None)
                    trace.note(f"{step.id}: {name} was empty and the step names "
                               f"its own attributes, so it resolves its own "
                               f"cohort rather than borrowing {source}'s")
                    continue
                step.args[name] = f"${source}.column.building_id"
                named = ("no cohort at all" if missing else
                         f"no building ({', '.join(map(str, list(literal)[:3]))})")
                trace.note(f"{step.id}: {name} named {named}; using the cohort "
                           f"{source} resolved instead")

            # A fraction written as a percent. Where the parameter has a
            # plausible alternative -- min_coverage=12 is months, not 12% --
            # the validator names it and the plan is wrong in a way only the
            # planner can settle. Where it has none, a percentile of 90 can
            # only be the 90th, and refusing over the scale costs an attempt.
            for key, value in list(step.args.items()):
                if (key not in FRACTION_PARAMS or key in ALTERNATIVE_PARAM
                        or key == "min_shift_pct"):   # already a percent
                    continue
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and 1 < value <= 100):
                    step.args[key] = value / 100
                    trace.note(f"{step.id}: {key}={value:g} read as a percent, "
                               f"so {value / 100:g}")

            for key, value in list(step.args.items()):
                if not (isinstance(value, str) and value.startswith("$")
                        and value.endswith(".handle")
                        and key.endswith("_ids")):
                    continue
                upstream = results.get(value[1:-len(".handle")])
                column = key[:-1]           # building_ids -> building_id
                rows = (upstream.data if upstream is not None
                        and isinstance(upstream.data, list)
                        else (upstream.preview if upstream is not None else None))
                if rows and isinstance(rows[0], dict) and column in rows[0]:
                    step.args[key] = f"{value[:-len('.handle')]}.column.{column}"
                    trace.note(f"{step.id}: {key} given a handle; read "
                               f"{column} from that step instead")

            misuse = [f"{step.tool}.{k} is a scalar and cannot take {v!r}: a "
                      f"handle names a whole table; use $step.column.<name>"
                      for k, v in step.args.items()
                      if isinstance(v, str) and v.startswith("$")
                      and v.endswith(".handle")
                      and k != "handle" and not k.endswith("_handle")]
            if misuse:
                result = ToolResult(Status.ERROR, message="; ".join(misuse)[:300],
                                    evidence={"arguments_rejected": True})
                results[step.id] = result
                state.record(step.id, result)
                trace.record_step(step, result, time.time(), step.args, attempt)
                return results, f"{step.id}: {misuse[0]}"

            try:
                args = resolve_args(step.args, results, self.ctx.store)
            except EmptyUpstream as exc:
                # Not an error and not worth replanning: the filter matched no
                # rows, and that is the answer. Reported as EMPTY so the run
                # ends with "nothing matched" rather than with a column name
                # the next attempt will hunt for and never find.
                result = ToolResult(Status.EMPTY, message=str(exc).strip("'"))
                results[step.id] = result
                state.record(step.id, result)
                trace.record_step(step, result, time.time(), step.args, attempt)
                return results, f"{step.id}: upstream matched nothing"
            except (KeyError, TypeError) as exc:
                result = ToolResult(Status.ERROR,
                                    message=f"argument resolution failed: {exc}",
                                    evidence={"arguments_rejected": True})
                results[step.id] = result
                state.record(step.id, result)
                trace.record_step(step, result, time.time(), step.args, attempt)
                return results, f"{step.id}: {exc}"

            args = {k: v for k, v in args.items() if v is not None}

            # A planner can only have invented an analyst-only argument: it was
            # never in the catalogue it was shown. Dropping it and saying so
            # beats failing the whole analysis over one hallucinated filter,
            # and beats honouring a row-dropping filter the model has no
            # evidence to set.
            spec = registry.get(step.tool)

            # A metric names the tool that owns it: consumption statistics
            # aggregate kWh, performance metrics are intensities, and the two
            # vocabularies do not overlap. A consumption step asking for "eui"
            # is misrouted rather than wrongly argued. Rerouting carries the
            # grouping and drops what has no meaning for an annual intensity --
            # a per-reading weight, an hour-of-day window -- saying which.
            if step.tool in METRIC_OWNERS:
                target, metric_key = METRIC_OWNERS[step.tool]
                declared = spec.parameters.get("properties", {})
                mine = set((declared.get(metric_key, {}).get("items")
                            or declared.get(metric_key, {})).get("enum") or [])
                asked = args.get(metric_key)
                asked = [asked] if isinstance(asked, str) else list(asked or [])
                other = registry.get(target)
                theirs = set((other.parameters.get("properties", {})
                              .get("metric", {})).get("enum") or [])
                stray = [m for m in asked if m not in mine and m in theirs]
                if stray and len(asked) == len(stray):
                    keep = {"building_ids", "across", "year", "min_coverage",
                            "at_least", "at_most", "eligibility"}
                    moved = {k: v for k, v in args.items() if k in keep}
                    allowed = set((other.parameters.get("properties", {})
                                   .get("across", {})).get("enum") or [])
                    if moved.get("across") not in allowed:
                        moved.pop("across", None)
                    year = _calendar_year(args.get("start"), args.get("end"))
                    if year is not None:
                        moved["year"] = year
                    moved["metric"] = stray[0]
                    left = sorted(set(args) - set(moved) - {"start", "end"})
                    step.tool, args, spec = target, moved, other
                    trace.note(
                        f"{step.id}: {stray[0]} is not a consumption "
                        f"statistic, it is a normalised intensity, so the "
                        f"step runs {target}"
                        + (f"; dropped {', '.join(left)}, which do not carry "
                           f"over to an annual intensity" if left else ""))

            # Arguments can name the tool as surely as a metric can. A
            # calculate_normalized_performance step carrying `order` is not a
            # step with one bad key: `order` is rank_buildings' parameter, and
            # so are the others beside it. The plan asked the measuring tool to
            # do the ranking tool's job, and three attempts in a row died on
            # "unknown parameter 'order'" with nothing computed.
            if step.tool == "calculate_normalized_performance":
                mine = set(spec.parameters.get("properties", {}))
                stray = [k for k in args if k not in mine]
                ranker = registry.get("rank_buildings")
                theirs = set(ranker.parameters.get("properties", {}))
                if stray and all(k in theirs for k in stray):
                    carried = {k: v for k, v in args.items() if k in theirs}
                    step.tool, args, spec = "rank_buildings", carried, ranker
                    step.agent = None
                    plan.fill_agents()
                    trace.note(
                        f"{step.id}: {', '.join(stray)} belongs to "
                        f"rank_buildings, not to the measurement, so the step "
                        f"runs the tool its own arguments describe")

            # The count and the end, applied where the arguments are final. A
            # step that becomes a ranking only here was not a ranking when the
            # plan-time repair ran over it, so it kept the model's own
            # order="asc", limit=1636 and returned all 229 offices from the
            # wrong end.
            if self.intent is not None:
                declared = spec.parameters.get("properties", {})
                wanted = resolve_top_n(self.intent.question or "")
                if wanted and {"limit", "order"} <= set(declared):
                    if (args.get("limit"), args.get("order")) != (
                            wanted["limit"], wanted["order"]):
                        args["limit"], args["order"] = wanted["limit"], wanted["order"]
                        trace.note(
                            f"{step.id}: the question asks for "
                            f"{wanted['limit']} from the "
                            f"{'top' if wanted['order'] == 'desc' else 'bottom'}, "
                            f"so the ranking is cut there")

            # A filter the question stated is not the planner's to drop. Asked
            # "how many assembly buildings are there in Berkeley", the model
            # planned search_catalog(entity="cities", city="Berkeley") and left
            # building_type out, so the answer was the city's total of 92 where
            # the true figure is 9. Only catalog lookups are corrected,
            # and only for a parameter the tool declares and the plan left
            # empty -- an explicit value, including null, is left alone.
            if self.intent is not None and spec.agent == "data_discovery":
                declared = spec.parameters.get("properties", {})
                for name, value in (self.intent.entities or {}).items():
                    if (name in declared and value is not None
                            and args.get(name) is None
                            and name not in step.args):
                        args[name] = value
                        trace.note(f"{step.id}: applied {name}={value!r} from the "
                                   f"question; the plan omitted it")
                # And the other way: a filter the question does not contain is
                # the planner's invention, not the question's. "All commercial
                # buildings" was narrowed to building_type="Office", which
                # answers about 307 of the 1,572 and calls the rest absent.
                asked_text = (self.intent.question or "").lower()
                for name in ("building_type", "city", "site_id", "country"):
                    value = args.get(name)
                    if (isinstance(value, str) and value
                            and (self.intent.entities or {}).get(name) is None
                            and value.lower() not in asked_text):
                        args.pop(name)
                        trace.note(f"{step.id}: dropped {name}={value!r} -- the "
                                   f"question does not name it, and a filter "
                                   f"nobody asked for shrinks the answer")

            # A bound the question states is not the plan's to drop. "Peak
            # demand exceeding 50 W/m2" was planned with the right metric and
            # the right year and no threshold at all, so it returned all 589
            # scored buildings where the question asks for the 144 above the
            # line. The tool declares the parameter; the question supplies the
            # number.
            declared_here = spec.parameters.get("properties", {})
            # ...unless the bound is already relative to a group. "More than
            # 50% above the median for its site" gives this resolver the
            # number 50, and the step is measuring a *ratio* to the group
            # median: nothing is 50x its own site's median, so the answer came
            # back empty. The step that knows the units owns the bound.
            if (self.intent is not None and not args.get("relative_to")
                    and ("at_least" in declared_here
                         or "at_most" in declared_here)):
                lower, upper = resolve_threshold(self.intent.question)
                for name, value in (("at_least", lower), ("at_most", upper)):
                    if (value is not None and name in declared_here
                            and args.get(name) is None and name not in step.args):
                        args[name] = value
                        trace.note(f"{step.id}: applied {name}={value:g} from the "
                                   f"question; the plan computed the measure but "
                                   f"kept every building")

            # A coverage floor the question never raises belongs to the tool.
            # In the performance tools min_coverage defaults to None, and None
            # there does not mean "no floor": it means the tool picks the floor
            # the metric and the direction need -- 0 for a peak, which a short
            # record can only understate, 0.9 for a total, which it deflates.
            # That rule is a property of the measure and lives with it. A plan
            # writing min_coverage=0 into a portfolio EUI question that says
            # nothing about coverage admitted 68 part-year buildings and moved
            # the median the answer reports.
            # ...unless the runtime itself set it. A computing step that
            # feeds a ranking is told to screen nothing, and dropping that
            # would hand the decision back to a default the ranking has
            # already overruled.
            delegated = (step.tool == "calculate_normalized_performance"
                         and args.get("min_coverage") == 0.0
                         and any(str(other.args.get("building_ids", ""))
                                 .lstrip("$").split(".")[0] == step.id
                                 for other in plan.steps
                                 if other.tool == "rank_buildings"))
            if (self.intent is not None and spec.agent == "performance"
                    and not delegated
                    and "default" not in declared_here.get("min_coverage", {})
                    and "min_coverage" in declared_here
                    and args.get("min_coverage") is not None
                    and not re.search(
                        r"\bcoverage\b|\bcomplete\w*\b|\bmissing\b|\bgaps?\b"
                        r"|\bzero\s+readings?\b|\bdata\s+quality\b|\boutage",
                        (self.intent.question or "").lower())):
                dropped_floor = args.pop("min_coverage")
                trace.note(f"{step.id}: dropped min_coverage={dropped_floor!r} -- "
                           f"the question does not ask about coverage, and the "
                           f"floor the metric needs is the tool's to choose")

            # The same reasoning for the statistic. "auto" is not a lack of a
            # choice: it is the choice the measure implies -- a total for a
            # total, the middle building for an intensity. A plan writing
            # "mean" into a question that names no statistic overrides that
            # with a guess, and the comparison came back on means where the
            # reference reports medians.
            if (self.intent is not None and "statistic" in declared_here
                    and args.get("statistic") not in (None, "auto")
                    and resolve_named_statistic(self.intent.question) is None):
                guessed = args.pop("statistic")
                trace.note(f"{step.id}: dropped statistic={guessed!r} -- the "
                           f"question names no statistic, and which one the "
                           f"measure needs is the tool's to choose")

            invented = [k for k in list(args) if k in spec.analyst_only]
            for key in invented:
                value = args.pop(key)
                trace.note(f"{step.id}: ignored {step.tool}.{key}={value!r} -- "
                           f"not available to a planner; the default applies")

            # A presentation step draws the frame the analysis produced, and
            # the planner routinely leaves the handle off because the subject
            # is obvious from position: it is the last thing computed. Refused
            # for the missing argument, the chart takes the whole answer down
            # with it, and the analysis underneath had already run.
            required = set(spec.parameters.get("required", []))
            if "handle" in required and not args.get("handle"):
                upstream = [results[other.id] for other in plan.steps
                            if other.id in results
                            and other.id != step.id
                            and getattr(results[other.id], "handle", None)]
                if upstream:
                    args["handle"] = upstream[-1].handle
                    trace.note(f"{step.id}: {step.tool} was given no handle; "
                               f"drawing the frame the step before it produced")

            # A parameter the tool spells differently is a wording mismatch,
            # not an analytical mistake: the question was read correctly and
            # the call is about to be refused over "group_by" against
            # "across". Rename what has an unambiguous home before judging.
            args, renamed = registry.canonical_args(step.tool, args)
            for note in renamed:
                trace.note(f"{step.id}: {note}")

            problems = registry.validate_args(step.tool, args)
            if problems:
                result = ToolResult(Status.ERROR,
                                    message="; ".join(problems)[:300],
                                    evidence={"arguments_rejected": True})
                results[step.id] = result
                state.record(step.id, result)
                trace.record_step(step, result, time.time(), step.args, attempt)
                return results, f"{step.id}: invalid arguments"
            began = time.time()
            result = registry.call(step.tool, self.ctx, **args)
            # Extraction and provenance both need the values actually used,
            # not the "$step.column" references the plan carried.
            result.provenance["resolved_args"] = {
                k: v for k, v in args.items()
                if not isinstance(v, (list, dict)) or len(str(v)) < 200}
            result.provenance["_args"] = args
            routing = self.router.route(step.tool, step.resolution, result=result)
            results[step.id] = result
            state.record(step.id, result)
            trace.record_step(step, result, began, args, attempt, routing=routing)

            # A refusal that leaves nothing to work with ends the chain.
            if result.status is Status.INSUFFICIENT_DATA and step is not plan.steps[-1]:
                dependents = [s for s in plan.steps if step.id in
                              (set(s.depends_on) | Plan.references(s))]
                if dependents:
                    trace.note(f"{step.id} refused ({result.predicate}); "
                               f"{len(dependents)} dependent step(s) skipped")
                    return results, f"{step.id} refused: {result.predicate}"
        return results, None



#: Tools that present a result rather than compute one. A failure in any of
#: them costs the presentation, never the answer.
PRESENTATION_TOOLS = frozenset({
    "create_visualization", "generate_report", "combine_evidence",
    "export_semantic_model",
})


def _narrative_lines(narrative: list) -> list[str]:
    """A tool's narrative as a summary sentence, then bullets under it.

    A run following a line that ends in a colon becomes sub-bullets of it.
    """
    written, parent_open = [], False
    for index, raw in enumerate(narrative):
        text = str(raw).strip()
        if not text:
            continue
        if index == 0:
            written.append(text)
            continue
        if parent_open and re.match(r"^[a-z][\w -]{0,28}:", text):
            written.append(f"    - {text}")
            continue
        parent_open = text.endswith(":")
        written.append(f"  - {text}")
    return written


def _cell(value, column: str = "") -> str:
    """One table cell: em dash for missing, four significant figures for a float."""
    if value is None or (isinstance(value, float) and value != value):
        return "\u2014"
    if isinstance(value, bool):
        return "yes" if value else "no"
    # A year is a label, not a quantity.
    if "year" in column.lower() or "hour" in column.lower():
        return f"{value:g}" if isinstance(value, float) else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # Four significant figures, but never in exponent form: "8.671e+05"
        # is not a number anyone reads.
        return (f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.4g}")
    return str(value)


def _aligned(columns: list[str], rows: list[dict]) -> list[str]:
    """A table whose columns line up, so a value can be read down one."""
    cells = [[_cell(r.get(c), c) for c in columns] for r in rows]
    widths = [max(len(str(columns[i])), *(len(row[i]) for row in cells))
              if cells else len(str(columns[i])) for i in range(len(columns))]
    def line(values):
        return "  " + "  ".join(v.ljust(w) for v, w in zip(values, widths)).rstrip()
    return [line(columns)] + [line(row) for row in cells]


def _heading(result: ToolResult) -> str:
    """Tool, what was counted, and how many.

    "How many sites are in this dataset" returned 19 rows and never said 19 --
    the reader had to tally them. The count is the answer to every "how
    many" question and the context for every other, so it leads the answer.
    """
    entity = result.evidence.get("entity")
    if result.rows is None and not result.preview:
        return "Answer:"
    # The count is what matched, not what a limit let through.
    matched = result.evidence.get("matched") or result.evidence.get("matched_filter")
    total = (int(matched) if matched is not None
             else (result.rows if result.rows is not None else len(result.preview)))
    noun = {"datasets": "dataset", "countries": "country", "cities": "city",
            "sites": "site", "building_types": "building type",
            "buildings": "building"}.get(entity or "")
    if noun:
        unit = noun if total == 1 else (noun[:-1] + "ies" if noun.endswith("y")
                                        else noun + "s")
    else:
        unit = "row" if total == 1 else "rows"
    shown = result.rows if result.rows is not None else len(result.preview or [])
    # A one-row summary needs no count: "1 dataset" atop a row whose buildings
    # column reads 1,636 invites exactly the misreading the count was added to
    # prevent. State it when there is more than one, or when a limit
    # held rows back.
    if total <= 1 and shown >= total:
        return "Answer:"
    suffix = f", {shown:,} shown" if shown < total else ""
    return f"{total:,} {unit}{suffix}:"


def _text_columns(result: ToolResult, all_columns: list[str]) -> list[str]:
    """The columns a text table shows, at most six of them.

    Provenance goes first when something has to go, because a reader asking
    what a dataset contains was shown its source URL, licence and citation
    while buildings, sites and date range were dropped. Then the head
    and the tail are kept over the middle, because the measure a question
    asked about is usually the last column.

    What never goes is a column the headline was computed from. The answer
    read "0.4437 -- median baseload per area, Pre-1980 / Post-2010" over a
    table showing total, p75, min and max, with the two medians it divided
    nowhere in sight.
    """
    PROVENANCE = ("source_url", "license", "citation", "description",
                  "dataset_version")
    columns = list(all_columns)
    if len(columns) > 6:
        columns = [c for c in columns if c not in PROVENANCE] or columns
    if len(columns) > 6:
        label = str(((result.evidence or {}).get("headline") or {})
                    .get("label", "")).lower()
        named = [c for c in columns
                 if label and re.search(rf"\b{re.escape(c)}\b", label)][:3]
        head = [c for c in columns if c not in named][:3]
        room = max(0, 6 - len(named) - len(head))
        rest = [c for c in columns if c not in named and c not in head]
        keep = set(named) | set(head) | set(rest[-room:] if room else [])
        columns = [c for c in columns if c in keep]
    return columns


def _render_frame(result: ToolResult) -> str:
    """One tool result as a labelled text block: header, rows, omission note."""
    lines = [_heading(result)]
    if isinstance(result.data, dict):
        for key, value in list(result.data.items())[:12]:
            if not isinstance(value, (list, dict)):
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    if not result.preview:
        return "\n".join(lines)
    all_columns = list(result.preview[0])
    columns = _text_columns(result, all_columns)
    lines.extend(_aligned(columns, result.preview[:5]))
    if len(columns) < len(all_columns):
        dropped = [c for c in all_columns if c not in columns]
        lines.append(f"  Also in the full table: {', '.join(dropped)}.")
    if result.rows and result.rows > len(result.preview):
        lines.append(f"  and {result.rows - len(result.preview):,} more rows.")
    return "\n".join(lines)


class EnergyXAgent:
    """The entry point. One question in, one verified answer out.

    ``mode`` toggles the architecture under evaluation: ``multi`` routes
    to specialists holding only their own tools, ``single`` gives one agent all
    48. Same registry, same traces -- only the visible tool set differs.
    """

    def __init__(self, mode: str = "multi", llm: LLM | None = None,
                 snapshot: Path | None = None, budget: Budget | None = None,
                 runs_root: Path | None = None, stateless: bool = False,
                 write_back: bool = False, planning_engine: str | None = None,
                 knowledge: bool = True, prefer_rules: bool = False,
                 require_llm: bool | None = None):
        load_all()
        if mode not in ("multi", "single"):
            raise ValueError("mode must be 'multi' or 'single'")
        self.mode = mode
        self.planning_engine = configured_engine(planning_engine)
        # Conversations want entities carried between turns ("now compare with
        # Singapore"). A measured run must not: a question that inherited an
        # entity from the previous question is no longer independent, which is
        # the same hazard as cross-session memory.
        self.stateless = stateless
        # Write-back is off unless asked for. A run that writes facts a later run
        # reads is no longer independent, so a measured run executes against a
        # frozen, read-only graph.
        self.write_back = write_back
        # The knowledge layer, switchable so it can be ablated. Off means the
        # run neither reads what earlier runs computed nor is told at planning
        # time what is already known: every answer is computed from the
        # snapshot. It is an evaluation control, not a mode -- the layer is on
        # in normal use.
        self.knowledge = knowledge
        # Rules first where a route exists for the question's shape, the model
        # everywhere else. The routes are the ones written deliberately for a
        # shape somebody verified; they carry no sampling, so a question they
        # match is answered the same way every time. Everything else is planned
        # by the model exactly as before -- the flag narrows where a model is
        # asked, never what the system can attempt.
        self.prefer_rules = prefer_rules
        # Every plan comes from the model, or the question errors. Three paths
        # could otherwise produce one without asking it -- the structured
        # compiler's shortcut, the verifier's downgrade, the planner's fallback
        # on a raised model call -- and each is indistinguishable in the result
        # from a question the model was never asked. A run that measures a model
        # must not silently measure the rules, so this flag closes all three.
        # prefer_rules is left alone: that is a choice the caller named.
        explicit = require_llm is not None
        if require_llm is None:
            require_llm = os.environ.get("ENERGYX_REQUIRE_LLM", "") not in ("", "0")
        self.llm = llm or NullLLM()
        # The environment variable is a default, not an override. Plenty of
        # call sites -- the tests, the deterministic bench runs -- build an
        # agent with no model at all, and a process-wide variable must not turn
        # those into constructor errors. Asked for explicitly with no model to
        # ask, it still raises: that is a caller mistake, not a setting.
        if not explicit and isinstance(self.llm, NullLLM):
            require_llm = False
        self.require_llm = bool(require_llm)
        self.db = SnapshotDB(snapshot)
        self.budget = budget or Budget()
        self.runs_root = Path(runs_root or PATHS.runs)
        self.state = SessionState(mode=mode)
        self.intent_agent = IntentAgent(Vocabulary.from_db(self.db))
        self.deterministic = DeterministicPlanner()
        if isinstance(self.llm, NullLLM):
            # No model to constrain, so the modes plan identically. Saying so is
            # better than manufacturing a difference that is not there.
            if self.require_llm:
                raise ValueError(
                    "require_llm is set but no model was given: there is "
                    "nothing to plan with. In the service this usually means "
                    "ENERGYX_LLM_PROVIDER is unset (it defaults to 'null'); "
                    "set it to 'ollama' with ENERGYX_LLM_MODEL, or set "
                    "ENERGYX_REQUIRE_LLM=0 to allow rule-based planning.")
            self.planner = self.deterministic
        elif mode == "multi":
            # Hierarchical: the router sees agent roles, each specialist sees
            # only its own tools.
            self.planner = HierarchicalPlanner(self.llm, fallback=self.deterministic,
                                               strict=self.require_llm)
        else:
            self.planner = LLMPlanner(self.llm, fallback=self.deterministic,
                                      strict=self.require_llm)

    def reset_session(self) -> None:
        """Drop carried entities and turn history; keep the loaded snapshot."""
        self.state = SessionState(mode=self.mode)

    def _apply_named_statistic(self, plan, question: str, trace) -> None:
        """Apply the statistic the question named to every step that takes one.

        Run once among the repairs and again after them, so a step added by a
        later repair does not keep the tool's default statistic.
        """
        # A comparison that names two measures asks for both, applied here so
        # a comparison added by a later repair gets it too: the step that
        # answers this question is created after the plan-level repairs have
        # run, and arrived with metric=load_factor and half the question.
        named_metrics = resolve_named_metrics(question)
        if len(named_metrics) > 1:
            for step in plan.steps:
                if step.tool == "compare_building_groups" and not step.args.get("metrics"):
                    step.args["metrics"] = named_metrics
                    step.args.pop("metric", None)
                    trace.note(
                        f"{step.id}: the question names "
                        f"{', '.join(named_metrics)}, so the comparison "
                        f"reports every one rather than the first")

        named_stat = resolve_named_statistic(question)
        if named_stat:
            before = [dict(step.args) for step in plan.steps]
            for step in plan.steps:
                tool = step.tool
                if tool == "calculate_demand_metrics":
                    # The median has its own column; the mean is the plain
                    # one. Whichever the plan is ordering by, say which.
                    order_by = step.args.get("order_by")
                    if order_by in ("daily_load_factor",
                                    "median_daily_load_factor"):
                        step.args["order_by"] = (
                            "median_daily_load_factor" if named_stat == "median"
                            else "daily_load_factor")
                elif tool == "compare_building_groups":
                    if named_stat in ("median", "mean", "total"):
                        step.args["statistic"] = named_stat
                elif tool == "calculate_consumption_statistics":
                    if step.args.get("across") not in (None, "building",
                                                       "cohort"):
                        if named_stat in ("median", "mean"):
                            step.args["group_aggregate"] = named_stat
                            step.args.setdefault("weight", "building")
                    # A limit over several metrics needs to know which
                    # one ranks: the tool refuses otherwise, and the
                    # question already said. Only a metric actually being
                    # computed can order the result.
                    metrics = step.args.get("metrics") or []
                    if step.args.get("limit") and not step.args.get("order_by"):
                        if named_stat in metrics:
                            step.args["order_by"] = named_stat
                        elif metrics:
                            step.args["order_by"] = metrics[0]
            # Said once, when it changed something. This runs twice -- among
            # the repairs and again after the ones that add steps -- and a
            # note repeated verbatim reads like the repair fired twice.
            if before != [dict(step.args) for step in plan.steps]:
                trace.note(f"the question names the {named_stat}, so that is "
                           f"the statistic computed rather than whichever the "
                           f"tool defaults to")


    # -- main loop ---------------------------------------------------------
    def ask(self, question: str,
            intent_overrides: dict | None = None) -> Answer:
        started = time.time()
        store = ArtifactStore.for_run(root=self.runs_root)
        ctx = ToolContext(db=self.db, store=store,
                          snapshot_version=SNAPSHOT_VERSION,
                          # The budget a tool must finish inside, so one heavy
                          # step cannot run past the whole run's limit.
                          # None, not infinity: every check reads "if deadline
                          # and ...", and an infinite one still flows into
                          # arithmetic that wants a number (OverflowError:
                          # cannot convert float infinity to integer).
                          deadline=(None if math.isinf(self.budget.max_seconds)
                                    else started + self.budget.max_seconds),
                          options={"write_back": self.write_back,
                                   "prefer_cached": self.knowledge})
        trace = Trace(run_id=store.run_id, question=question, mode=self.mode,
                      snapshot=SNAPSHOT_VERSION,
                      budget=self.budget.__dict__.copy())
        turn = self.state.begin_turn(question)

        carried = {} if self.stateless else self.state.entities
        intent = self.intent_agent.resolve(question, carried=carried)
        # What the reader corrected, before anything is planned from it. The
        # resolved intent is what actually drives planning, and until now it
        # was invisible: a question read as a comparison *between sites* when
        # it compared two day types *within* one produced a confident answer
        # to a question nobody asked, with nothing in the response saying so.
        # Correcting it here is narrower and more honest than rewriting the
        # question, because it changes the thing the plan is built from rather
        # than guessing at new words for it.
        applied = apply_intent_overrides(intent, intent_overrides)
        turn.intent = intent
        trace.intent = intent.to_dict()
        if applied:
            trace.intent_overrides = applied
            for field_name, value in applied.items():
                trace.note(f"intent corrected by the caller: "
                           f"{field_name} = {value!r}")
        if self.stateless:
            trace.note("stateless run: no entities carried from earlier questions")
        else:
            self.state.remember_entities(intent.entities)

        # What the question assumes, checked against the data before anything
        # is computed from it. A premise that is false is not a planning
        # problem -- no plan can recover from a building id that does not
        # exist, because the building is already gone from the intent by the
        # time a plan is written, and the question quietly becomes
        # portfolio-wide.
        premises = check_premises(question, intent, self.db,
                                  self.intent_agent.vocab)
        trace.premises = premises.to_dict()
        for finding in premises.findings:
            trace.note(f"premise [{finding.severity}] {finding.message}")
        if not premises.ok:
            first = premises.fatal[0]
            detail = "; ".join(f.message for f in premises.fatal)
            hint = next((f.suggestion for f in premises.fatal if f.suggestion), "")
            text = (f"Cannot answer: {detail}."
                    + (f" {hint[0].upper()}{hint[1:]}." if hint else "")
                    + f" Failing predicate: {first.code}.")
            turn.answer = text
            trace.answer = text
            trace.status = "INSUFFICIENT_DATA"
            payload = AnswerPayload(
                "refusal", {"status": "INSUFFICIENT_DATA", "text": text,
                            "predicate": first.code},
                method="premise_gate").to_dict()
            trace.answer_payload = payload
            path = trace.write(self.runs_root)
            return Answer(question=question, text=text,
                          status="INSUFFICIENT_DATA", run_id=trace.run_id,
                          mode=self.mode, plan={"steps": [], "notes": []},
                          evidence=[], results={},
                          verification=trace.verification,
                          metrics=trace.metrics(), payload=payload,
                          trace_path=str(path))

        # What is already known shapes the plan, so probe before planning.
        known = probe(ctx.graph, SNAPSHOT_VERSION) if self.knowledge else KnownFacts()
        trace.note(f"known facts at planning time: {known.summary()}"
                   if self.knowledge else
                   "knowledge layer off: planning cold, nothing served from cache")
        spec = None
        if self.planning_engine in ("shadow", "structured"):
            spec = AnalysisSpec.from_intent(intent, known)
            problems = spec.validate()
            turn.analysis_spec = spec
            trace.analysis_spec = spec.to_dict()
            trace.analysis_spec_problems = problems
            trace.note(f"{self.planning_engine} AnalysisSpec: "
                       f"{spec.task}/{spec.resolution}")
            if self.planning_engine == "structured" and (problems or spec.issues):
                detail = "; ".join(problems or spec.issues)
                text = f"Cannot answer: {detail}. Failing predicate: valid_analysis_spec."
                payload = _answer_payload(spec, Plan(), {}, text,
                                          "INSUFFICIENT_DATA").to_dict()
                trace.answer, trace.status = text, "INSUFFICIENT_DATA"
                trace.answer_payload = payload
                trace.verification = {
                    "status": "ok", "verifier": "analysis_spec",
                    "checks": [], "failed": problems or spec.issues,
                }
                path = trace.write(self.runs_root)
                return Answer(
                    question=question, text=text, status="INSUFFICIENT_DATA",
                    run_id=trace.run_id, mode=self.mode,
                    plan={"steps": [], "notes": []}, verification=trace.verification,
                    metrics=trace.metrics(), payload=payload, trace_path=str(path),
                )

        # Refuse what the snapshot cannot hold, before any tool runs. Answering
        # "what is the indoor air temperature" with electricity, or a gas
        # question with kWh, is the worst failure available: it is confidently
        # wrong and passes every execution check.
        # Is this a question at all? Program output pasted into the box was
        # answered with 1,572 buildings.
        readable, why_not = looks_like_a_question(question)
        if not readable:
            text = (f"Cannot answer: {why_not}. "
                    f"Failing predicate: is_a_question.")
            trace.note("refused before planning: input is not a question")
            trace.answer, trace.status = text, "INSUFFICIENT_DATA"
            turn.answer = text
            trace.verification = {
                "status": "ok", "verifier": "input",
                "checks": [{"check": "the input is a question", "passed": False,
                            "detail": why_not}], "failed": []}
            payload = AnswerPayload("refusal",
                                    {"status": "INSUFFICIENT_DATA", "text": text,
                                     "predicate": "is_a_question"},
                                    method="input_gate").to_dict()
            trace.answer_payload = payload
            path = trace.write(self.runs_root)
            return Answer(question=question, text=text,
                          status="INSUFFICIENT_DATA", run_id=trace.run_id,
                          mode=self.mode, plan={"steps": [], "notes": []},
                          evidence=[], results={}, verification=trace.verification,
                          metrics=trace.metrics(), payload=payload,
                          trace_path=str(path))

        # Capability gate: the snapshot's resolution, metering level, span and
        # dataset count, checked against what the question needs.
        limit_check = None
        if not intent.out_of_scope:
            try:
                limits = SnapshotLimits.from_db(self.db)
                limit_check = unsupported(question, limits)
            except Exception:                  # a gate must never fail a run
                limit_check = None
        if limit_check:
            predicate, why = limit_check
            text = f"Cannot answer: {why}. Failing predicate: {predicate}."
            self._consult_planner_for_the_record(intent, trace)
            trace.note(f"refused before planning: {predicate}")
            trace.answer, trace.status = text, "INSUFFICIENT_DATA"
            turn.answer = text
            trace.verification = {
                "status": "ok", "verifier": "capability",
                "checks": [{"check": "the snapshot supports what was asked",
                            "passed": False, "detail": predicate}], "failed": []}
            payload = AnswerPayload("refusal",
                                    {"status": "INSUFFICIENT_DATA", "text": text,
                                     "predicate": predicate},
                                    method="capability_gate").to_dict()
            trace.answer_payload = payload
            path = trace.write(self.runs_root)
            return Answer(question=question, text=text,
                          status="INSUFFICIENT_DATA", run_id=trace.run_id,
                          mode=self.mode, plan={"steps": [], "notes": []},
                          evidence=[], results={}, verification=trace.verification,
                          metrics=trace.metrics(), payload=payload,
                          trace_path=str(path))

        if intent.out_of_scope:
            text = (f"Cannot answer: this snapshot meters electricity only, "
                    f"with site weather and building metadata as context, and "
                    f"the question asks about {intent.out_of_scope}. "
                    f"Failing predicate: in_scope({intent.out_of_scope}).")
            self._consult_planner_for_the_record(intent, trace)
            trace.note(f"refused before planning: {intent.out_of_scope} is "
                       f"outside the snapshot's scope")
            trace.answer, trace.status = text, "INSUFFICIENT_DATA"
            turn.answer = text
            trace.verification = {
                "status": "ok", "verifier": "intent",
                "checks": [{"check": "question is within the snapshot's scope",
                            "passed": False, "detail": intent.out_of_scope}],
                "failed": []}
            path = trace.write(self.runs_root)
            return Answer(question=question, text=text,
                          status="INSUFFICIENT_DATA", run_id=trace.run_id,
                          mode=self.mode, plan={"steps": [], "notes": []},
                          evidence=[], results={},
                          verification=trace.verification,
                          metrics=trace.metrics(), trace_path=str(path))

        orchestrator = Orchestrator(ctx, mode=self.mode, intent=intent)
        verifier = VerificationAgent()
        results: dict[str, ToolResult] = {}
        verification: dict = {}
        plan = Plan()

        # Per-question, never per-agent. This was `self.planner = self.deterministic`,
        # which permanently replaced the planner on the first verifier rejection --
        # and since the service caches one agent per mode for the life of the
        # process, one rejected question downgraded every later question to
        # keyword routing, silently and for good.
        force_deterministic = (self.planning_engine == "structured"
                               and not self.require_llm)
        correction = ""

        # One extra attempt is allowed when a bad argument was corrected, so a
        # correction does not consume the replan that the fallback needs. A
        # plan corrected once and still wrong then still gets rules rather
        # than an error.
        bonus = 0
        halt = ""            # bound here: every attempt can end on a `continue`
        for attempt in range(1, self.budget.max_replans + 3):
            if attempt > self.budget.max_replans + 1 + bonus:
                break
            # Planning costs time too, and a replan that cannot be paid for is
            # worse than none. One question took three planning attempts of
            # about forty seconds each and spent its entire 120-second budget
            # before any step ran, then reported "no analysis could be run" --
            # which blames the question for what the clock did. Stop when the
            # budget is gone, whether or not anything was produced, and say so.
            if attempt > 1 and self.budget.exceeded(trace, started):
                trace.note(f"attempt {attempt} not started: "
                           f"{self.budget.exceeded(trace, started)}")
                break
            plan, meta = self._make_plan(intent, trace, attempt, known, spec=spec,
                                         force_deterministic=force_deterministic,
                                         correction=correction)
            if self.planning_engine == "structured" and plan.steps:
                plan.answer_step = plan.answer_step or plan.steps[-1].id
                if spec.aggregation == "count":
                    plan.answer_field = plan.answer_field or "rows"
            # A question about the building is not a question about the
            # finding store. Asked for the apparent operating schedule of one
            # office, single mode planned lookup_derived_facts alone and
            # returned a finding IRI -- a cached fact recomputed under an
            # older definition, presented as the answer. The store holds what
            # some earlier run wrote; the question asks what the meter says.
            # Every fact kind there names the tool that computes it, so the
            # substitution has one right answer.
            asks_for_record = re.search(
                r"\b(?:previously|already|earlier)\b|\bwhat\s+(?:has\s+been|was)\s+"
                r"(?:computed|recorded|found|stored|asserted)|\bon\s+record\b"
                r"|\bknowledge\s+graph\b|\bfindings?\s+store\b", question.lower())
            if plan.steps and not asks_for_record:
                computes = {kind: tool for tool, kind in CACHEABLE_TOOLS.items()}
                for step in plan.steps:
                    if step.tool != "lookup_derived_facts":
                        continue
                    instead = computes.get(step.args.get("finding_type"))
                    if not instead or not step.args.get("building_ids"):
                        continue
                    step.tool = instead
                    step.args = {"building_ids": step.args["building_ids"]}
                    step.resolution = "compute"
                    trace.note(f"{step.id}: the question asks what the readings "
                               f"show, so {instead} computes it rather than "
                               f"lookup_derived_facts reading what was stored")

            # The subject decides the grain of the answer. "The daily load
            # factor for site Peacock" is about the site, and both modes
            # answered with a ranked list of the site's buildings: the tool
            # can pool a cohort, the question said which cohort, and the plan
            # asked for members instead. Only when the question ranks nothing
            # -- no highest, lowest, top or worst -- is a per-member table
            # clearly not what was meant.
            ranking_words = re.search(
                r"\bhighest|\blowest|\btop\b|\bbottom\b|\bworst\b|\bbest\b"
                r"|\brank\w*|\bmost\b|\bleast\b|\bwhich\s+buildings?\b",
                question.lower())
            names_one_cohort = bool(
                (intent.entities or {}).get("site_id")
                or (intent.entities or {}).get("building_type")) and not (
                    intent.entities or {}).get("building_ids")
            # "For all healthcare facilities" pools them; "for each" or "per
            # building" asks for the members. The subject alone could not tell
            # these apart -- both are about buildings -- and the question says
            # which it wants.
            pools_all = re.search(r"\b(?:for|across|over|among)\s+all\b",
                                  question.lower())
            per_member = re.search(r"\beach\b|\bper\s+building\b"
                                   r"|\bby\s+building\b|\bevery\b",
                                   question.lower())
            # eui_analysis means "has a floor area and enough record to divide
            # a year by". A peak per square metre needs the floor area and not
            # the year: it divides a maximum, and a short record can only
            # understate one. Asked of the education cohort the requirement
            # dropped 110 of 604 buildings and 24 of the 144 answers, so where
            # a later step measures a peak intensity the requirement is not
            # the one the question implies.
            peak_step = any(s.args.get("metric") in ("peak_power_density",
                                                     "peak_per_area", "peak_demand")
                            for s in plan.steps)
            if peak_step:
                for step in plan.steps:
                    needs = step.args.get("requires")
                    if isinstance(needs, list) and "eui_analysis" in needs:
                        step.args["requires"] = [r for r in needs
                                                 if r != "eui_analysis"] or None
                        trace.note(f"{step.id}: dropped requires=eui_analysis -- "
                                   f"a peak needs a floor area, not a whole year")

            # Where an analytic step *is* in the plan, its method and its
            # bound still come from the question rather than from the
            # planner's imagination. Asked whether any building in Ireland
            # shows unusual weekend consumption -- a question naming no
            # threshold at all -- the planner wrote threshold=1.0 against a
            # tool whose default is 3.0, and the answer was 9,968 "anomalous"
            # days: one standard deviation flags roughly a third of every
            # building's year, so the scan found normal variation and reported
            # it as a finding.
            for step in plan.steps:
                if step.tool != "detect_anomalies":
                    continue
                # "Unusual *weekend* consumption" scans weekends and compares
                # them with weekends. Ignoring the word answered a question
                # about Saturdays with one about every day of the week.
                day_type = resolve_day_type(question)
                if day_type != "all" and step.args.get("days") != day_type:
                    step.args["days"] = day_type
                    trace.note(f"{step.id}: the question is about {day_type}s, "
                               f"so both the scan and its baseline are "
                               f"restricted to them")
                # A clock window named in the question restricts the scan and
                # its baseline. "Anomalous nighttime hours from 8 PM to 7 AM"
                # compared against the whole day finds every night.
                clock = resolve_clock_window(question)
                if clock and step.args.get("hour_from") is None:
                    step.args["hour_from"], step.args["hour_to"] = clock
                    trace.note(f"{step.id}: restricted to hours "
                               f"{clock[0]:02d}-{clock[1]:02d}, as the question "
                               f"states")
                method, method_named = resolve_anomaly_method(question)
                bound = ANOMALY_BOUND.search(question.lower())
                if method_named and step.args.get("method") != method:
                    step.args["method"] = method
                    trace.note(f"{step.id}: the question names the "
                               f"{method} test")
                if bound:
                    wanted = resolve_anomaly_threshold(
                        question, step.args.get("method") or method)
                    if step.args.get("threshold") != wanted:
                        step.args["threshold"] = wanted
                        trace.note(f"{step.id}: threshold={wanted:g} as the "
                                   f"question states it")
                elif step.args.get("threshold") is not None:
                    # No bound in the question, so the tool's own default is
                    # the one that was reasoned about. A planner-invented
                    # threshold is the same class of error as an invented
                    # period: it narrows or widens an answer on nobody's
                    # authority.
                    dropped = step.args.pop("threshold")
                    trace.note(f"{step.id}: dropped threshold={dropped!r} -- "
                               f"the question names no bound, so the test's "
                               f"own default applies")

            # "After 2017-07-31" puts the whole of the 31st in the history and
            # starts the horizon on 1 August. The cutoff is exclusive, so the
            # date named has to be stepped past; the rule route does this and
            # a model plan did not, forecasting the very day the question
            # excluded.
            if re.search(r"\bafter\b", question.lower()):
                for step in plan.steps:
                    if step.tool != "forecast_timeseries":
                        continue
                    named = (intent.time_range or {}).get("start")
                    cutoff = step.args.get("cutoff") or named
                    if cutoff and str(cutoff)[:10] == str(named)[:10]:
                        moved = (date.fromisoformat(str(cutoff)[:10])
                                 + timedelta(days=1)).isoformat()
                        step.args["cutoff"] = moved
                        trace.note(f"{step.id}: \"after {named}\" puts that day "
                                   f"in the history, so the cutoff is {moved}")

            # A question asking for a forecast, an anomaly scan or a change
            # point wants an analysis, and a plan that only fetches the
            # measurements has not done it. The test is structural: the intent
            # carries the task category, the registry knows which tools serve
            # it, and none of them is in the plan.
            # A named percentile is a different number from the peak the tool
            # reports by default, so pass what the question named.
            wanted_pcts = resolve_percentiles(question)
            if wanted_pcts:
                for step in plan.steps:
                    if (step.tool == "calculate_demand_metrics"
                            and not step.args.get("percentiles")):
                        step.args["percentiles"] = wanted_pcts
                        plan.answer_step = step.id
                        plan.answer_field = f"p{wanted_pcts[0]:g}_demand"
                        trace.note(
                            f"the question named the "
                            f"{', '.join(f'{p:g}th' for p in wanted_pcts)} "
                            f"percentile, which the peak is not; asked "
                            f"{step.tool} for it")

            # "Twelve consecutive months" is a run, and the model reaches
            # for coverage because that is the argument it knows: it asked for
            # min_months_of_data=12 with min_coverage=0.99, which answered 631
            # where 285 buildings actually hold an unbroken year. Coverage and
            # continuity are different properties and the question named the
            # second.
            unbroken = resolve_unbroken_hours(question)
            if unbroken:
                for step in plan.steps:
                    if (step.tool == "find_buildings"
                            and not step.args.get("min_unbroken_hours")):
                        step.args["min_unbroken_hours"] = unbroken
                        # The screens the model reached for instead would now
                        # cut into the set the run already defines. An unbroken
                        # run of N hours *is* "no gap inside it"; a ceiling on
                        # the longest gap anywhere in the record is a stricter
                        # and different condition, and it took 285 buildings
                        # down to 262.
                        step.args.pop("min_coverage", None)
                        step.args.pop("min_months_of_data", None)
                        step.args.pop("gap_at_most_hours", None)
                        trace.note(
                            f"the question asks for an unbroken run, so the "
                            f"cohort is buildings with {unbroken:,} consecutive "
                            f"hourly readings rather than a coverage fraction")

            # A question that counts a population, or asks what excluding
            # part of it does, needs every building the metric can be computed
            # for. The default cohort has already dropped the ones whose
            # record the snapshot judged too short -- 1,256 of 1,547 -- so the
            # count is answered over a set that silently excluded the subject.
            if asks_over_whole_population(question):
                for step in plan.steps:
                    if (step.tool in POPULATION_TOOLS
                            and not step.args.get("eligibility")):
                        step.args["eligibility"] = "structural"
                        trace.note(
                            f"the question is about the population, so "
                            f"{step.tool} covers every building the metric "
                            f"can be computed for rather than only those "
                            f"whose record is long enough to rank")

            # A "controlling for" question needs the controls passed, or the
            # tool fits the plain slope and answers a question about the
            # working day instead. The model plans the right tool and has no
            # way to know the argument exists.
            # "Above the 90th percentile within its own category" compares
            # each building against its own kind. The model answers it with
            # cohort-wide bands, which rank the categories against each other
            # rather than the buildings within them: 13 buildings where 136
            # qualify.
            within = resolve_within_group_percentile(question)
            existing_peers = next((s_ for s_ in plan.steps
                                   if s_.tool == "identify_peer_outliers"), None)
            if within and existing_peers is not None:
                # The model reached for the right tool and not for the
                # arguments that make it answer *this* question: without the
                # grouping it compares against the whole cohort, and without
                # the fitness floor it counts buildings whose EUI is an
                # artefact of a short record.
                filled = []
                for key, value in (("peers", within["peers"]),
                                   ("method", "percentile"),
                                   ("threshold", within["threshold"]),
                                   ("side", within["side"]),
                                   ("eligibility", "structural"),
                                   ("min_coverage", 0.9),
                                   ("min_band_size", 5)):
                    if not existing_peers.args.get(key):
                        existing_peers.args[key] = value
                        filled.append(key)
                plan.answer_step = existing_peers.id
                plan.answer_field = None
                if filled:
                    trace.note(f"{existing_peers.id}: filled {', '.join(filled)} "
                               f"so the percentile is taken within each "
                               f"{within['peers']}")
            elif within:
                cohort = next((s_ for s_ in plan.steps
                               if s_.tool == "find_buildings"), None)
                # A portfolio question binds no cohort: the model's
                # find_buildings step is often filtered to something narrower
                # (21 ids on one run), and inheriting it answers "within their
                # own category" over a fragment of one category.
                wide = asks_over_whole_population(question)
                ranked = PlanStep(
                    "s_peers", "identify_peer_outliers",
                    {"building_ids": (None if wide else
                                      (f"${cohort.id}.column.building_id"
                                       if cohort else None)),
                     "metric": "eui", "method": "percentile",
                     "threshold": within["threshold"], "side": within["side"],
                     "peers": within["peers"], "eligibility": "structural",
                     # Comparing buildings against each other is what the 0.9
                     # coverage floor is for -- the performance tool says so in
                     # as many words -- and a percentile needs at least a
                     # handful of peers to mean anything. The cohort stays
                     # structural so the floor is applied once, here, rather
                     # than twice with the snapshot's own record-length gate.
                     "min_coverage": 0.9, "min_band_size": 5},
                    depends_on=[cohort.id] if cohort else [],
                    resolution="compute",
                    purpose=f"buildings in the top {within['threshold']:g}% of "
                            f"their own {within['peers']}")
                if not cohort or ranked.args.get("building_ids") is None:
                    ranked.args.pop("building_ids", None)
                    ranked.depends_on = []
                plan.add(ranked)
                plan.answer_step = ranked.id
                plan.answer_field = None
                trace.note(
                    f"the question compares each building with its own "
                    f"{within['peers']}, so the percentile is taken within "
                    f"each group rather than across the cohort")

            # A per-building statistic rolled up to a group is not the same
            # as the group's pooled figure, and the question says which it
            # wants. Pooled, "the highest median within-building variability"
            # named Healthcare; building by building it names Industrial.
            grouping = resolve_group_statistic(question)
            if grouping:
                for step in plan.steps:
                    if (step.tool == "calculate_consumption_statistics"
                            and step.args.get("across") not in (None, "building",
                                                                "cohort")):
                        step.args.setdefault("weight", grouping["weight"])
                        step.args.setdefault("group_aggregate",
                                             grouping["group_aggregate"])
                        trace.note(
                            f"the question asks about the typical building in "
                            f"each group, so each building is measured first "
                            f"and the {grouping['group_aggregate']} taken "
                            f"across them")

            # "At least one complete calendar year" is a property of a
            # year, and the model answers it with a coverage fraction over the
            # whole record -- which is a different set, and on this snapshot a
            # much smaller one: 129 offices where 284 hold a clean year.
            complete = resolve_complete_year(question)
            if complete:
                for step in plan.steps:
                    # A coverage is a fraction, so a value outside 0..1 is a
                    # mistake to correct rather than a value to respect.
                    for key in ("complete_year_coverage",
                                "consecutive_months_coverage"):
                        given = step.args.get(key)
                        if isinstance(given, (int, float)) and not 0 < given <= 1:
                            step.args.pop(key, None)
                    # The floor the question states overrides the one the plan
                    # guessed, rather than only filling a missing value.
                    if step.tool == "find_buildings":
                        for key in ("complete_year_coverage",
                                    "consecutive_months_coverage"):
                            if (step.args.get(key) is not None
                                    and step.args[key] != complete["coverage"]):
                                step.args[key] = complete["coverage"]
                    if step.tool == "find_buildings":
                        # Which twelve months: the calendar's, or any twelve
                        # in a row. The question decides, and the plan the
                        # model wrote always reached for the calendar one.
                        if complete.get("basis") == "rolling":
                            step.args["consecutive_months_coverage"] = \
                                complete["coverage"]
                            step.args.pop("complete_years", None)
                        else:
                            step.args["complete_year_coverage"] = complete["coverage"]
                            if complete["years"]:
                                step.args["complete_years"] = complete["years"]
                            else:
                                step.args.pop("complete_years", None)
                        if complete["weather"]:
                            step.args["weather_coverage_at_least"] = complete["weather"]
                        if complete.get("grouping_attribute"):
                            have = list(step.args.get("has_metadata") or [])
                            if "building_type" not in have:
                                step.args["has_metadata"] = have + ["building_type"]
                        # A record-wide floor now double-counts the screen.
                        step.args.pop("min_coverage", None)
                        # An unbroken-run screen is a different, stricter rule
                        # and cannot stand beside this one: "12 consecutive
                        # months with 95% valid readings" allows 5% to be
                        # missing, and min_unbroken_hours=8760 does not.
                        step.args.pop("min_unbroken_hours", None)
                        # And so does the other twelve-month rule.
                        step.args.pop("complete_year_coverage"
                                      if complete.get("basis") == "rolling"
                                      else "consecutive_months_coverage", None)
                        # "A complete calendar year" is itself the
                        # completeness criterion, so a record-length or gap
                        # filter applied beside it screens on a stricter rule
                        # the question never stated. Kept only when the
                        # question names gaps or a record length itself.
                        # "Consecutive" does not count: that window is the
                        # criterion, not a separate gap filter.
                        if not re.search(r"\bgaps?\b|\bunbroken\b"
                                         r"|\bmonths?\s+of\s+data\b",
                                         (question or "").lower()):
                            for invented_screen in ("min_months_of_data",
                                                    "gap_at_most_hours",
                                                    "gap_longer_than_hours"):
                                step.args.pop(invented_screen, None)
                        trace.note(
                            "the question asks for a complete calendar year, "
                            "so the cohort is buildings with a clean year"
                            + (" and overlapping weather in it"
                               if complete["weather"] else "")
                            + ("; a normalised figure is read against "
                               "comparable buildings, so a recorded use is "
                               "part of being eligible for it"
                               if complete.get("grouping_attribute") else ""))

            # A qualifier on a measure is part of the measure. "Nighttime
            # baseload" is the overnight floor -- the mean between midnight
            # and 05:00, in W/m2 -- and baseload_per_area is the 5th
            # percentile of the whole day's demand in kW/m2. Both are
            # baselines and they are not the same number: 28.9 against 0.106
            # on the office cohort, so the percentile drawn on the wrong one
            # selected a different set of buildings.
            if re.search(r"\b(?:night|nightt?ime|overnight|nocturnal)\b",
                         (question or "").lower()):
                for step in plan.steps:
                    if step.args.get("metric") == "baseload_per_area":
                        step.args["metric"] = "overnight_baseload_per_area"
                        trace.note(
                            f"{step.id}: the question names the nighttime "
                            f"baseload, which is the overnight floor in W/m2, "
                            f"not the day's 5th percentile in kW/m2")

            # A comparison that names two measures asks for both. The tool
            # takes one at a time unless told otherwise, so "compare load
            # factor and peak-to-average between offices and education" was
            # answered on whichever the plan picked and silent on the other.
            named_metrics = resolve_named_metrics(question)
            if len(named_metrics) > 1:
                for step in plan.steps:
                    if step.tool != "compare_building_groups":
                        continue
                    if step.args.get("metrics"):
                        continue
                    step.args["metrics"] = named_metrics
                    step.args.pop("metric", None)
                    trace.note(
                        f"{step.id}: the question names "
                        f"{', '.join(named_metrics)}, so the comparison "
                        f"reports every one rather than the first")

            # Seasonal variation is a statistic over a building's twelve
            # monthly levels, and the seasonal *profile* tool returns the
            # levels: 180 rows of per-building, per-season means, which are
            # the ingredients with the statistic never taken. The measure the
            # question names has a tool that computes it.
            if resolve_measure(question, default=None) == "seasonal_cv" and not any(
                    s_.args.get("metric") == "seasonal_cv" for s_ in plan.steps):
                # Whatever tool the plan reached for. The reroute named the
                # seasonal profile tool and the next run planned monthly
                # consumption statistics instead, so the measure the question
                # named still had nothing computing it: the answer came back
                # as one building's monthly standard deviation.
                measuring = [s_ for s_ in plan.steps
                             if s_.tool in ("calculate_load_profile",
                                            "calculate_consumption_statistics",
                                            "calculate_normalized_performance",
                                            "rank_buildings")]
                profile = measuring[-1] if measuring else None
                if profile is not None:
                    lowered = (question or "").lower()
                    grouping = next(
                        (dim for pattern, dim in (
                            (r"\bcategor\w+|\bbuilding\s+type|\buse\s+type|"
                             r"\bprimary[-\s]use", "building_type"),
                            (r"\bsubtype\b", "building_subtype"),
                            (r"\bwhich\s+site\b|\bby\s+site\b", "site"),
                            (r"\bwhich\s+cit\w+|\bby\s+city\b", "city"),
                            (r"\bwhich\s+countr\w+", "country"))
                         if re.search(pattern, lowered)), "building")
                    profile.tool = "calculate_normalized_performance"
                    profile.args = {
                        k: v for k, v in profile.args.items() if k == "building_ids"}
                    profile.args.update(metric="seasonal_cv", across=grouping)
                    profile.agent = None
                    plan.answer_step = profile.id
                    plan.fill_agents()
                    trace.note(
                        f"{profile.id}: seasonal variation is the spread of a "
                        f"building's monthly levels, so the step computes "
                        f"seasonal_cv across {grouping} rather than returning "
                        f"the seasonal profile it is taken from")

            # "Remain in the top 10% after normalizing" is one tail drawn on
            # two measures, and the answer is who survives both. Drawn once it
            # answered the second half only: the 27 offices in the top decile
            # by EUI, where the question asks which 9 of the top decile by
            # consumption are also there.
            pair = resolve_before_after_metrics(question)
            tail_for_pair = resolve_tail(question)
            if pair and tail_for_pair and plan.steps:
                cohort = next((s_ for s_ in plan.steps
                               if s_.tool == "find_buildings"), None)
                if cohort is not None and not any(
                        s_.tool == "combine_evidence" for s_ in plan.steps):
                    plan.steps = [cohort]
                    plan.fill_agents()
                    ids = f"${cohort.id}.column.building_id"
                    drawn = []
                    for which in ("before", "after"):
                        step = PlanStep(
                            f"{cohort.id}_{which}", "identify_peer_outliers",
                            {"building_ids": ids, "metric": pair[which],
                             "method": "percentile",
                             "threshold": tail_for_pair["threshold"],
                             "side": tail_for_pair["side"]},
                            depends_on=[cohort.id], resolution="compute",
                            purpose=f"the tail on {pair[which]}")
                        plan.add(step)
                        drawn.append(step.id)
                    joined = PlanStep(
                        f"{cohort.id}_both", "combine_evidence",
                        {"sources": [f"${d}.handle" for d in drawn],
                         "key": "building_id", "mode": "all",
                         "labels": [pair["before"], pair["after"]]},
                        depends_on=drawn, resolution="compute",
                        purpose="the buildings in both tails")
                    plan.add(joined)
                    plan.answer_step = joined.id
                    plan.answer_field = None
                    plan.fill_agents()
                    trace.note(
                        f"the question asks who stays in the "
                        f"{tail_for_pair['threshold']}% tail after normalising, "
                        f"so the tail is drawn on {pair['before']} and on "
                        f"{pair['after']} and the two are intersected")

            # Two curves, not one. A weekday-versus-weekend question answered
            # with a single hour-of-day profile states the difference it was
            # asked about nowhere: the two curves differ by 19% on this
            # building and the averaged one shows neither.
            if asks_weekday_weekend_profile(question):
                for step in plan.steps:
                    if (step.tool == "calculate_load_profile"
                            and step.args.get("profile") != "weekday_weekend"):
                        step.args["profile"] = "weekday_weekend"
                        step.args.pop("peak_only", None)
                        trace.note(
                            f"{step.id}: the question contrasts weekday with "
                            f"weekend, so the profile is built for each "
                            f"rather than averaged over both")

            # A breakout is a persistent shift in level, not an outlying
            # hour, and the anomaly tools cannot see one: every hour after the
            # shift is unremarkable against its new neighbours. Asked how many
            # breakouts a building had, the plan scanned for anomalies over a
            # 168-hour window and answered none.
            breakout = resolve_breakout(question)
            if breakout:
                for step in plan.steps:
                    if step.tool not in ("detect_anomalies",
                                         "sweep_anomaly_sensitivity"):
                        continue
                    carried = {k: v for k, v in step.args.items()
                               if k in ("building_ids", "start", "end")}
                    carried.update(min_segment_hours=breakout["min_segment_hours"])
                    step.tool, step.args, step.agent = "detect_breakouts", carried, None
                    plan.answer_step = step.id
                    trace.note(
                        f"{step.id}: a breakout is a level shift that persists, "
                        f"not an unusual hour, so the step segments the series "
                        f"on a {breakout['min_segment_hours']}-hour minimum")
                plan.fill_agents()

            # A question about the *change* between two thresholds is
            # answered by sweeping both, not by scanning at one of them. Asked
            # how the outlier count moves from 3 SD to 2 SD, the plan scanned
            # at 3 SD and reported 0 -- correct for half the question and
            # silent on the 12 the change adds.
            sweep = resolve_threshold_sweep(question)
            if sweep:
                for step in plan.steps:
                    if step.tool != "detect_anomalies":
                        continue
                    carried = {k: v for k, v in step.args.items()
                               if k in ("building_ids", "start", "end")}
                    carried.update(method=sweep["method"],
                                   thresholds=sweep["thresholds"])
                    step.tool = "sweep_anomaly_sensitivity"
                    step.args = carried
                    step.agent = None
                    plan.answer_step = step.id
                    trace.note(
                        f"{step.id}: the question compares "
                        f"{' and '.join(f'{t:g} SD' for t in sweep['thresholds'])}, "
                        f"so the step sweeps both rather than scanning at one")
                plan.fill_agents()

            # "On which day" wants a dated day. day_of_week answers Thursday,
            # and grain="daily" aggregates the days away entirely, so the
            # question had no route to the date it asked for.
            if asks_calendar_day(question):
                for step in plan.steps:
                    if (step.tool == "calculate_consumption_statistics"
                            and step.args.get("across") in (None, "building",
                                                            "day_of_week")):
                        step.args["across"] = "date"
                        step.args.setdefault("grain", "daily")
                        trace.note(
                            f"{step.id}: the question asks which calendar day, "
                            f"so the statistic is grouped by date rather than "
                            f"by weekday or rolled up to the building")

            # A forecast fitted on daily values is a different model from an
            # hourly one summed afterwards: seven points from thirty, not 168
            # from 720. The horizon counts steps at that grain too.
            if asks_daily_grain(question):
                for step in plan.steps:
                    if step.tool != "forecast_timeseries":
                        continue
                    step.args["grain"] = "daily"
                    hours = step.args.get("horizon_hours")
                    if isinstance(hours, int) and hours >= 24 and hours % 24 == 0:
                        step.args["horizon_hours"] = hours // 24
                    trace.note(
                        f"{step.id}: the question fits on daily data, so the "
                        f"model is fitted at the daily grain and the horizon "
                        f"counts days rather than hours")

            # Every clustering algorithm the question names, not the first.
            cluster_methods = resolve_cluster_methods(question)
            wanted_distance = resolve_cluster_distance(question)
            wanted_k = resolve_n_clusters(question)
            for step in plan.steps:
                if step.tool == "compare_clustering_methods" and len(cluster_methods) > 1:
                    if set(step.args.get("methods") or []) != set(cluster_methods):
                        step.args["methods"] = cluster_methods
                        trace.note(
                            f"{step.id}: the question compares "
                            f"{' and '.join(cluster_methods)}, so both are scored")
                if step.tool in ("cluster_timeseries", "compare_clustering_methods"):
                    if wanted_distance != "euclidean" and not step.args.get("distance"):
                        step.args["distance"] = wanted_distance
                        trace.note(
                            f"{step.id}: the question names the "
                            f"{wanted_distance} distance, which is not the "
                            f"default the clustering would have used")
                    if wanted_k is not None and not step.args.get("n_clusters"):
                        step.args["n_clusters"] = wanted_k

            # A rank correlation when the question names one. Unqualified,
            # "correlation" is Pearson; "the Spearman correlation" is a
            # different statistic and on dew point the two differ enough to
            # move buildings across a 0.75 threshold.
            if asks_spearman(question):
                for step in plan.steps:
                    if (step.tool == "analyze_weather_relationship"
                            and step.args.get("method") in (None, "correlation")):
                        step.args["method"] = "spearman"
                        trace.note(
                            f"{step.id}: the question asks for the Spearman "
                            f"correlation, so the series are ranked before "
                            f"they are correlated")

            # "On how many days did at least one outlier occur" counts dates.
            # The scan returns one row per incident, so without this the
            # answer is the incident count wearing the word "days".
            if asks_day_count(question):
                for step in plan.steps:
                    if (step.tool == "detect_anomalies"
                            and not step.args.get("report")):
                        step.args["report"] = "days"
                        trace.note(
                            f"{step.id}: the question counts the days an "
                            f"outlier occurred on, so the scan is summarised "
                            f"by date rather than by incident")

            # "Which buildings had the HIGHEST NUMBER of anomalous readings"
            # is a ranking of buildings. Reported by reading it came back as
            # "65 buildings have at least one reading flagged, out of 65
            # scanned" -- true, and not an answer to which had the most.
            if re.search(r"\b(?:which|what)\b[^.?]{0,80}?"
                         r"\b(?:highest|most|greatest|largest|top)\b"
                         r"[^.?]{0,40}?\bnumber\s+of\b|\bmost\s+anomal\w+",
                         (question or "").lower()):
                for step in plan.steps:
                    if (step.tool == "detect_anomalies"
                            and step.args.get("report") in (None, "readings")):
                        step.args["report"] = "buildings"
                        trace.note(f"{step.id}: the question ranks buildings "
                                   f"by how many anomalies each had, so the "
                                   f"scan is summarised by building")

            # An anomaly measured against the entity's own history, and
            # counted by entity rather than by incident. Both are stated in
            # the question and neither was reaching the tool: the default
            # contextual test answered 571 incidents against a rolling
            # neighbourhood where the question asks how many buildings ever
            # cross their own line.
            self_baseline = resolve_self_baseline(question)
            if self_baseline:
                for step in plan.steps:
                    if step.tool != "detect_anomalies":
                        continue
                    before = dict(step.args)
                    for field, value in self_baseline.items():
                        step.args.setdefault(field, value)
                    # The model's own default is not a stated choice, so the
                    # named test wins over it.
                    step.args["method"] = self_baseline["method"]
                    step.args["report"] = self_baseline["report"]
                    if before != step.args:
                        trace.note(
                            f"{step.id}: the question measures each building "
                            f"against its own mean, so the test is a z-score "
                            f"at {self_baseline['threshold']:g} SD, counted by "
                            f"{self_baseline['report']}")

            # "Which five ... highest" states both the count and the end. A
            # ranking left to the plan came back as 229 buildings ordered
            # ascending -- every office, worst end first, which contains the
            # answer without stating it.
            top_n = resolve_top_n(question)
            if top_n:
                for step in plan.steps:
                    if step.tool != "rank_buildings":
                        continue
                    before = dict(step.args)
                    step.args["limit"] = top_n["limit"]
                    step.args["order"] = top_n["order"]
                    if before != step.args:
                        trace.note(
                            f"{step.id}: the question asks for "
                            f"{top_n['limit']} from the "
                            f"{'top' if top_n['order'] == 'desc' else 'bottom'}, "
                            f"so the ranking is cut there")

            # A threshold can belong to the building's own group rather than
            # to the cohort. "More than 50% above the median EUI of offices at
            # the same site" is one rule and nineteen numbers; applied as a
            # cohort-wide figure it filtered nothing and the answer was all
            # 234 buildings that had an EUI.
            relative = resolve_relative_threshold(question)
            if relative:
                for step in plan.steps:
                    if step.tool != "calculate_normalized_performance":
                        continue
                    # Each field on its own. The model can now see
                    # relative_to in the catalogue and set it without the
                    # bound that gives it meaning; skipping the step because
                    # one field was present left at_least unset, and the tool
                    # dutifully measured every building's ratio to its site
                    # and filtered none of them -- 262 rows where 68 clear the
                    # line. A partially-correct plan needs the rest supplied,
                    # not deference.
                    before = dict(step.args)
                    step.args.setdefault("relative_to", relative["relative_to"])
                    step.args.setdefault("relative_statistic",
                                         relative["relative_statistic"])
                    for bound in ("at_least", "at_most"):
                        if relative[bound] is not None:
                            step.args.setdefault(bound, relative[bound])
                    if before == step.args:
                        continue
                    trace.note(
                        f"{step.id}: the threshold is relative to each "
                        f"building's own {relative['relative_to']}, so it is "
                        f"measured as a multiple of that group's "
                        f"{relative['relative_statistic']} rather than a "
                        f"cohort-wide number")

            # "Above the 90th percentile" draws a line; a ranking only puts
            # the cohort in order and leaves the line undrawn. Asked which
            # offices sit above the 90th percentile of EUI and of nighttime
            # baseload, the plan ranked all 217 on each and intersected the
            # two orders, which is every building twice over. The tail the
            # question named is what identify_peer_outliers takes.
            tail = resolve_tail(question)
            if tail:
                for step in plan.steps:
                    if step.tool != "rank_buildings":
                        continue
                    carried = {k: v for k, v in step.args.items()
                               if k in ("building_ids", "metric", "year",
                                        "min_coverage", "peers")}
                    carried.update(method="percentile",
                                   threshold=tail["threshold"],
                                   side=tail["side"])
                    step.tool = "identify_peer_outliers"
                    step.args = carried
                    step.agent = None
                    trace.note(
                        f"{step.id}: the question names the "
                        f"{'top' if tail['side'] == 'high' else 'bottom'} "
                        f"{tail['threshold']}% of the cohort, so the step "
                        f"selects that tail rather than ordering everyone")
                plan.fill_agents()

            # "Both above X and above Y" is an intersection, and the join
            # that fans the specialists back in defaults to keeping everyone
            # any of them named. Left alone the answer counted the union: 32
            # offices clear one of the two 90th percentiles where 25 clear
            # both, and a question asking which to prioritise got the larger,
            # wrong set. An explicit mode in the plan is left alone.
            if asks_for_conjunction(question):
                for step in plan.steps:
                    if (step.tool == "combine_evidence"
                            and not step.args.get("mode")
                            and len(step.args.get("sources") or []) > 1):
                        step.args["mode"] = "all"
                        trace.note(
                            f"{step.id}: the question asks for entities meeting "
                            f"every criterion, so the evidence is intersected "
                            f"rather than unioned")

            # The statistic the question names is not a preference the
            # planner gets to weigh: "median" and "mean" are exact words, and
            # answering one with the other is wrong however close they land.
            # Left to the model, the same question came back 0.61 on one run
            # and 0.53 on the next.
            self._apply_named_statistic(plan, question, trace)

            # "Cool hours versus warm hours" is a contrast across a line the
            # question does not name, and the model answers it with an overall
            # mean -- one number where two were asked for. The weather tool
            # takes the split; nothing was asking it to.
            split = resolve_temperature_split(question)
            if split:
                placed = False
                for step in plan.steps:
                    if step.tool == "analyze_weather_relationship":
                        step.args["method"] = "threshold_contrast"
                        step.args.setdefault("threshold", split)
                        step.args.setdefault("grain", "hourly")
                        plan.answer_step, plan.answer_field = step.id, None
                        placed = True
                if not placed:
                    named = (intent.entities or {}).get("building_ids") or []
                    contrast = PlanStep(
                        "s_split", "analyze_weather_relationship",
                        {"method": "threshold_contrast", "threshold": split,
                         "grain": "hourly",
                         **({"building_ids": list(named)} if named else {})},
                        resolution="compute",
                        purpose="mean hourly use in cool hours against warm")
                    plan.add(contrast)
                    plan.answer_step, plan.answer_field = contrast.id, None
                trace.note(
                    f"the question contrasts cool with warm hours, so the "
                    f"split is taken at the {split} of the site's own "
                    f"temperatures rather than a fixed degree")

            # A zero-consumption run is a data-quality question and the
            # model reaches for a consumption tool: "more than 72 consecutive
            # hours of zero" came back as a table of annual totals. The check
            # exists and carries the distinction the question turns on --
            # zeros in the cleaned series, or days that were zero before
            # ingest masked them.
            zero_run = resolve_zero_run(question)
            # A plan that already checks zero runs still has to check the ones
            # the question asked about. The length and the basis are stated in
            # the question, not chosen by the planner: asked for runs longer
            # than 72 hours the plan looked for runs of a single day, on a
            # basis it never set, and the repair that would have fixed both
            # stood down because a zero_run step already existed.
            if zero_run:
                for step in plan.steps:
                    if (step.tool != "detect_data_issues"
                            or "zero_run" not in (step.args.get("checks") or [])):
                        continue
                    before = (step.args.get("consecutive_zero_days_at_least"),
                              step.args.get("zero_basis"))
                    step.args["consecutive_zero_days_at_least"] = zero_run["days"]
                    if zero_run["hours"] % 24:
                        step.args.pop("consecutive_zero_hours_at_least", None)
                    step.args["consecutive_zero_hours_at_least"] = zero_run["hours"]
                    step.args.setdefault("zero_basis", zero_run["basis"])
                    if before != (zero_run["days"], step.args["zero_basis"]):
                        trace.note(
                            f"{step.id}: the question asks for runs over "
                            f"{zero_run['days']} day(s) on the "
                            f"{step.args['zero_basis']} series, so the check "
                            f"uses those rather than the plan's "
                            f"{before[0]!r} day(s)")
            if zero_run and not any(s_.tool == "detect_data_issues"
                                    for s_ in plan.steps):
                checked = PlanStep(
                    "s_zero", "detect_data_issues",
                    {"checks": ["zero_run"],
                     "consecutive_zero_days_at_least": zero_run["days"],
                     "consecutive_zero_hours_at_least": zero_run["hours"],
                     "zero_basis": zero_run["basis"]},
                    resolution="compute",
                    purpose="runs of zero consumption, on the basis the "
                            "question named")
                plan.add(checked)
                plan.answer_step = checked.id
                plan.answer_field = None
                trace.note(
                    f"the question asks about runs of zero consumption over "
                    f"{zero_run['days']} day(s), counted on the "
                    f"{zero_run['basis']} series")

            # find_buildings caps at 500 rows. That is right for a listing
            # and wrong for a cohort: when another step consumes the ids, the
            # cap silently decides which buildings the statistic is computed
            # over -- 500 of 1,572, chosen by catalogue order. The evidence
            # says truncated_by_limit, and the number downstream says nothing.
            # A step whose output feeds another is an input, not a listing,
            # unless the question itself asked for a top-N.
            if not re.search(r"\b(?:top|first|bottom|last)\s+\d+|"
                             r"\b\d+\s+(?:highest|lowest|largest|smallest)\b",
                             question, re.I):
                feeding = {dep for step in plan.steps for dep in (step.depends_on or [])}
                for step in plan.steps:
                    if (step.tool == "find_buildings" and step.id in feeding
                            and int(step.args.get("limit") or 0) <= 500):
                        step.args["limit"] = 5000
                        trace.note(
                            f"{step.id}: its cohort feeds a later step, so the "
                            f"500-row listing cap is lifted -- a statistic over "
                            f"the first 500 of 1,572 buildings is not the "
                            f"portfolio's")

            # "Which site has the most education buildings" is a count
            # grouped by site, and the model answers it by listing the cohort:
            # 617 rows of education buildings, which contain the answer and do
            # not state it. The catalogue counts over a dimension and ranks
            # them, which is the question.
            entity = resolve_grouped_count(question)
            if entity and plan.steps and not any(
                    step.tool == "search_catalog" for step in plan.steps):
                # Carry the filter the cohort step already resolved, so
                # "education buildings" survives into the count.
                carried = {}
                for step in plan.steps:
                    for key in ("building_type", "country", "city", "site_id"):
                        if step.args.get(key) and key not in carried:
                            carried[key] = step.args[key]
                counted = PlanStep(
                    "s_count", "search_catalog", {"entity": entity, **carried},
                    resolution="lookup",
                    purpose=f"count buildings per {entity[:-1]} and rank them")
                plan.add(counted)
                plan.answer_step = counted.id
                plan.answer_field = None
                trace.note(
                    f"the question asks which {entity[:-1]} has the most, so "
                    f"the answer is a count grouped by {entity[:-1]} rather "
                    f"than the cohort itself")

            # The resolution the question names. A correlation of hourly
            # pairs and one of daily totals are different statistics, and the
            # tool defaults to daily: asked for the hourly correlation it
            # answered 0.39 where the hourly figure is 0.46.
            grain = resolve_weather_grain(question)
            if grain:
                for step in plan.steps:
                    if step.tool in WEATHER_TOOLS and not step.args.get("grain"):
                        step.args["grain"] = grain
                        trace.note(f"the question names {grain} resolution, so "
                                   f"{step.tool} pairs {grain} values")

            wanted_controls = resolve_weather_controls(question)
            if wanted_controls:
                for step in plan.steps:
                    # Not "only when the model forgot": it often sets controls
                    # *and* asks for a correlation, where controls do nothing.
                    # Skipping then left them set and ignored, and the answer
                    # was the raw figure the question said to control away.
                    if step.tool == "analyze_weather_relationship":
                        step.args["controls"] = (step.args.get("controls")
                                                 or wanted_controls)
                        # Controls only mean something in the regression, and
                        # the model asks for a correlation: setdefault left
                        # them set and ignored, so the answer was the raw
                        # figure the question said to control away.
                        step.args["method"] = "regression"
                        trace.note(
                            f"the question asks to control for "
                            f"{', '.join(wanted_controls)}, so the weather fit "
                            f"holds them constant rather than reporting the "
                            f"raw slope")

            # Clustering one building "across buildings" has one profile to
            # work with and cannot be done at all -- the tool answered
            # "1 profiles cannot be clustered". With a single subject the only
            # thing the question can mean is that building's own days, so the
            # scope is set rather than the plan refused.
            for step in plan.steps:
                if step.tool != "cluster_timeseries" or step.args.get("scope"):
                    continue
                named = step.args.get("building_ids")
                if isinstance(named, (list, tuple)) and len(named) == 1:
                    step.args["scope"] = "within_building"
                    trace.note(
                        f"one building named, so {step.tool} clusters its own "
                        f"daily profiles rather than comparing it with others")

            # When the question is about X and a tool that answers X is in the
            # plan, the answer comes from that tool. Steps after it are dropped
            # -- they exist only to have something to say -- and steps before it
            # are kept, because they built its inputs.
            for asks, answering in ANSWERING_TOOLS:
                if not asks(question) or not plan.steps:
                    continue
                owned = [s for s in plan.steps if s.tool in answering]
                if not owned or plan.steps[-1].tool in answering:
                    continue
                last = plan.steps.index(owned[-1])
                dropped = [s.tool for s in plan.steps[last + 1:]]
                plan.steps = plan.steps[:last + 1]
                plan.answer_step = owned[-1].id
                plan.answer_field = None
                trace.note(
                    f"the question is answered by {owned[-1].tool}"
                    + (f"; dropped {', '.join(dropped)} which ran after it"
                       if dropped else ""))
                break

            analytic = ANALYTIC_TOOLS.get(intent.task_category or "")
            if (analytic and plan.steps
                    and not any(s.tool in analytic["tools"] for s in plan.steps)):
                named = (intent.entities or {}).get("building_ids") or []
                # Without a target there is nothing to analyse, and inventing
                # one would be worse than handing back what the plan found.
                # A cohort question keeps whatever step resolved the cohort.
                upstream = [s for s in plan.steps
                            if registry.REGISTRY.get(s.tool)
                            and registry.REGISTRY[s.tool].agent == "data_discovery"]
                args = dict(analytic["args"](question, intent))
                if named:
                    args[analytic["target"]] = (named[0] if analytic["one"]
                                                else list(named))
                    # The question named its subject, so nothing upstream is
                    # needed to find it. Keeping the fetch that the unrepaired
                    # plan opened with costs a 744-row scan to learn a building
                    # id that was in the question.
                    upstream = []
                elif upstream:
                    args[analytic["target"]] = f"${upstream[0].id}.column.building_id"
                else:
                    args = None
                if args is not None:
                    chosen = analytic["tool"]
                    chosen = chosen(question) if callable(chosen) else chosen
                    keep = upstream[:1]
                    plan.steps = keep + [PlanStep(
                        "a1", chosen, args, depends_on=[s.id for s in keep],
                        purpose=analytic["purpose"])]
                    plan.answer_step = "a1"
                    trace.note(
                        f"the question asks for {analytic['what']} and the plan "
                        f"only read the measurements; replaced it with {chosen}")

            # "Which 10 offices have the highest use per square metre" puts the
            # ten at the end, not at the start. A limit on the step that finds
            # the cohort truncates it before the measure exists, so the top ten
            # is chosen from an arbitrary ten -- multi returned the first ten
            # offices alphabetically and called them the highest. When a later
            # step ranks, the cohort step is not where the limit belongs.
            ranks_later = any(s.tool in ("rank_buildings", "identify_peer_outliers")
                              or (s.args.get("order_by") or s.args.get("order"))
                              for s in plan.steps)
            if ranks_later:
                for step in plan.steps:
                    spec_for = registry.REGISTRY.get(step.tool)
                    if (spec_for and spec_for.agent == "data_discovery"
                            and isinstance(step.args.get("limit"), int)
                            and step.args["limit"] < 1000):
                        step.args["limit"] = 5000
                        trace.note(f"{step.id}: the limit belongs to the ranking, "
                                   f"not to finding the cohort; searching the "
                                   f"whole population instead")

            # A two-regime comparison is about the cohort by construction.
            # "How the peak timing of education buildings shifts between the
            # semesters" came back as 604 rows, one per building, which is the
            # comparison left to be done by hand rather than an answer.
            contrasts_two = (plan.steps
                             and plan.steps[-1].tool == "compare_temporal_profiles")
            if (plan.steps and not ranking_words and names_one_cohort
                    and not per_member
                    and (intent.subject in ("sites", "building_types")
                         or pools_all or contrasts_two)):
                last = plan.steps[-1]
                spec_for = registry.REGISTRY.get(last.tool)
                accepts = ((spec_for.parameters or {}).get("properties") or {}
                           if spec_for else {})
                # ...and only where the tool means the same thing by it. A
                # repair that writes a value the tool does not declare turns a
                # right plan into an argument error: fit_temperature_response
                # pools by site, not by "cohort", and the site fit died three
                # times over on the word.
                across_values = accepts.get("across", {}).get("enum") or []
                named_site = (intent.entities or {}).get("site_id")

                if "cohort" in across_values:
                    last.args["across"] = "cohort"
                elif "site" in across_values and named_site:
                    # A tool that pools by site says so in its own word. "The
                    # cooling change-point for site Robin" is one fit over the
                    # site's daily consumption, and the plan asked for one fit
                    # per building and handed back the list.
                    last.args["across"] = "site"
                    last.args["site_ids"] = [named_site]
                    last.args.pop("building_ids", None)
                    for dropped in ("order_by", "order", "limit"):
                        last.args.pop(dropped, None)
                    trace.note(f"{last.id}: the question is about the "
                               f"{intent.subject[:-1]}, so its statistics are "
                               "pooled rather than listed per building")

            # "Which buildings have meter data available" asks for readings,
            # not for a registered meter. Six streams in this snapshot hold
            # nothing, so the two readings differ by six buildings and the
            # question named one of them.
            if re.search(r"\b(?:meter|measurement|consumption|electricity)\s+"
                         r"data\b[^.?]{0,20}\bavailable\b"
                         r"|\bhave\b[^.?]{0,24}\breadings?\b"
                         r"|\bwith\s+(?:any\s+)?(?:meter|electricity)\s+data\b",
                         question.lower()):
                for step in plan.steps:
                    spec_step = registry.REGISTRY.get(step.tool)
                    accepts = ((spec_step.parameters or {}).get("properties") or {}
                               if spec_step else {})
                    if "has_readings" in accepts and step.args.get(
                            "has_readings") is None:
                        step.args["has_readings"] = True
                        trace.note(f"{step.id}: data *available* means readings "
                                   f"exist, not that a meter is registered")
                        break

            # A stated data-quality threshold is a filter, and
            # profile_data_quality is a description. "Which buildings have
            # less than 2% missing electricity data for year 2017" was planned
            # as find_buildings then profile_data_quality, and the answer was
            # that tool's 1,327 rows -- every stream it described, presented
            # as the ones that passed. The deterministic route has drawn this
            # distinction since the quality tag was written; the model does
            # not, so the swap is made here.
            missing_pct = MISSING_SHARE.search(question.lower())
            if missing_pct:
                share = float(missing_pct.group(1)) / 100
                # ...and the filter is a *floor on coverage*, not a ceiling.
                # `detect_data_issues(max_coverage=0.02)` flags streams whose
                # coverage is below 2% -- ten nearly-empty meters, the exact
                # opposite of "less than 2% missing". The cohort step already
                # takes the right filter, measured against the hours the year
                # should hold.
                year = (intent.time_range or {}).get("year")
                for step in plan.steps:
                    if step.tool != "find_buildings":
                        continue
                    step.args["min_coverage"] = round(1 - share, 6)
                    if year:
                        step.args["coverage_year"] = int(year)
                    plan.answer_step = step.id
                    trace.note(
                        f"{step.id}: less than {share:.0%} missing is a floor "
                        f"of {1 - share:.0%} coverage"
                        + (f" within {year}" if year else ""))
                    break
                # The describing step is then not the answer; drop it so its
                # row count cannot stand in for the filtered set.
                plan.steps = [s for s in plan.steps
                              if s.tool != "profile_data_quality"]

            # A band the question names has to be applied, not merely
            # computed. "Which office buildings maintain a top-decile baseload
            # ratio" was planned as find_buildings then calculate_baseload for
            # all 296 offices, with the decile nowhere in it: every office
            # returned, presented as the top tenth. The measuring tool carries
            # the cut, and the question says which.
            band = TOP_BAND.search(question.lower())
            if band:
                wanted_top = bool(re.search(r"\btop\b|\bhighest\b|\bupper\b",
                                            band.group(0)))
                for step in plan.steps:
                    spec_step = registry.REGISTRY.get(step.tool)
                    accepts = ((spec_step.parameters or {}).get("properties") or {}
                               if spec_step else {})
                    if "decile_at_least" not in accepts:
                        continue
                    if step.args.get("decile_at_least") is not None:
                        break
                    if not wanted_top:
                        break            # a bottom band is a different cut
                    step.args["decile_at_least"] = 10
                    plan.answer_step = step.id
                    trace.note(f"{step.id}: the question asks for the top "
                               f"decile, so the cut is applied rather than "
                               f"every member returned")
                    break

            # A cohort requirement has to match the measure it screens for.
            # Asked which buildings use most per occupant, the plan required
            # both eui_analysis and occupancy_normalisation: the first wants a
            # floor area and a full year, neither of which a per-occupant
            # figure divides by, and it narrowed the population the ranking
            # was drawn from.
            # The *intent's* metric, not the measure word: "consumption
            # relative to occupants" reads as total_kwh to the measure
            # resolver and as per_occupant to the intent tag, and it is the
            # tag that is right. Keying this on the resolver made the repair
            # dead code that appeared to work only because one prompt happened
            # to plan correctly without it.
            metric_named = (intent.metric
                            if intent.metric in PERFORMANCE_METRICS
                            else resolve_measure(question, default=None))
            if metric_named == "per_occupant":
                for step in plan.steps:
                    needs = step.args.get("requires")
                    if isinstance(needs, list) and "eui_analysis" in needs:
                        step.args["requires"] = [r for r in needs
                                                 if r != "eui_analysis"] or None
                        trace.note(f"{step.id}: dropped requires=eui_analysis -- "
                                   f"a per-occupant figure divides by occupants, "
                                   f"not by floor area")

            # The clustering algorithm and the number of groups are the
            # question's, the same way a forecasting model is. A plan that
            # runs k-means where Ward was asked for answers a different
            # question with the same table shape.
            for step in plan.steps:
                if step.tool != "cluster_timeseries":
                    continue
                method, named = resolve_cluster_method(question)
                if named and step.args.get("method") != method:
                    step.args["method"] = method
                    trace.note(f"{step.id}: the question names {method} "
                               f"clustering")
                metric = resolve_cluster_distance(question)
                if metric != "euclidean" and step.args.get("distance") != metric:
                    step.args["distance"] = metric
                    trace.note(f"{step.id}: clusters formed under {metric} "
                               f"distance, as the question states")
                wanted = resolve_n_clusters(question, default=0)
                if wanted and step.args.get("n_clusters") != wanted:
                    step.args["n_clusters"] = wanted
                    trace.note(f"{step.id}: the question asks for {wanted} "
                               f"clusters")
                elif wanted == "auto" and step.args.get("n_clusters") != "auto":
                    # The question asks how many clusters there are. Answering
                    # with the planner's default and reporting that count back
                    # is circular, so k is handed to the data.
                    step.args["n_clusters"] = "auto"
                    trace.note(f"{step.id}: the question asks how many "
                               f"clusters there are, so k is chosen by "
                               f"silhouette rather than assumed")

            # A finer type the question named is a filter too. The catalog
            # carries building_subtype on 1,615 of 1,636 buildings, and
            # "dormitories", "college laboratories" and "fire stations" are
            # questions about it: answering them with the canonical type
            # conflates a dormitory with a lecture hall, because both are
            # Education. Applied here rather than at each of the fifteen
            # places a cohort step is built, and it reaches a model's plan as
            # well as a rule's.
            for field_name in ("building_subtype", "industry", "subindustry"):
                value = (intent.entities or {}).get(field_name)
                if not value:
                    continue
                for step in plan.steps:
                    spec_step = registry.REGISTRY.get(step.tool)
                    accepts = ((spec_step.parameters or {}).get("properties") or {}
                               if spec_step else {})
                    if field_name not in accepts or step.args.get(field_name):
                        continue
                    if step.args.get("building_ids"):
                        continue
                    step.args[field_name] = value
                    trace.note(f"{step.id}: the question names "
                               f"{field_name.replace('_', ' ')} {value!r}")

            # A site the question named is a filter the plan may not drop.
            # "What was the highest wind speed recorded at site Bull" was
            # planned as find_buildings(site_id=Bull), fetch_weather, then
            # calculate_weather_statistics with no site at all -- so the
            # answer was nineteen rows, one per site, with Bull among them.
            # The intent resolved the site; a step that accepts one and
            # carries no cohort of any kind is missing it rather than
            # deliberately pooling.
            named_site = (intent.entities or {}).get("site_id")
            if named_site and plan.steps:
                for step in plan.steps:
                    spec_step = registry.REGISTRY.get(step.tool)
                    accepts = ((spec_step.parameters or {}).get("properties") or {}
                               if spec_step else {})
                    if any(step.args.get(key) for key in
                           ("building_ids", "site_ids", "site_id", "city",
                            "country")):
                        continue
                    for key in ("site_ids", "site_id"):
                        if key in accepts:
                            step.args[key] = ([named_site] if key == "site_ids"
                                              else named_site)
                            trace.note(f"{step.id}: the question names site "
                                       f"{named_site}, and this step carried no "
                                       f"cohort at all")
                            break

            # A period the caller stated outright is not a suggestion. The
            # resolved intent goes into the planning prompt, and a prompt rule
            # can still talk the model out of it: told the window was 2016,
            # the planner wrote year=2017 because a rule says an EUI with no
            # year named means the latest full one. The rule is right about
            # questions and wrong about this question, and a correction the
            # planner may overrule is not a correction at all.
            if (intent.time_range or {}).get("stated_by_caller"):
                window = intent.time_range
                for step in plan.steps:
                    spec_step = registry.REGISTRY.get(step.tool)
                    accepts = ((spec_step.parameters or {}).get("properties") or {}
                               if spec_step else {})
                    for key in ("start", "end", "year"):
                        if key in accepts and window.get(key) is not None:
                            if step.args.get(key) != window[key]:
                                step.args[key] = window[key]
                                trace.note(f"{step.id}: {key}={window[key]!r} "
                                           f"as the caller stated it")

            # The resolved intent is the authority on which period was asked
            # for. When it carries none, a window in the plan was invented --
            # a season plus a one-year window answers "over the summer" with
            # one of the two summers the question covers.
            if plan.steps and not (intent.time_range or {}).get("start"):
                for step in plan.steps:
                    # An intensity is per year by definition -- EUI is kWh per
                    # square metre per year -- so a year on one of those steps
                    # is part of the measure, not an invented window. Stripping
                    # it leaves the tool pooling the snapshot's eighteen months
                    # and returning a larger number that is not an EUI, which
                    # reordered four of the top ten offices.
                    if step.args.get("metric") in ANNUAL_METRICS:
                        continue
                    # year is the same invention in a different parameter, and
                    # it slipped through a repair that only knew about start
                    # and end: a comparison over the whole record was computed
                    # for 2017 alone.
                    if (step.args.get("start") or step.args.get("end")
                            or step.args.get("year")):
                        step.args.pop("start", None)
                        step.args.pop("end", None)
                        step.args.pop("year", None)
                        trace.note(f"{step.id}: dropped a period the question "
                                   "did not ask for")

            # A plan made entirely of catalog lookups has not measured
            # anything. "What is Lamb_office_Bertha's annual electricity EUI"
            # was planned as get_building_metadata and nothing else: the floor
            # area was fetched, the division never happened, and a metadata
            # row was handed back as the intensity. Looking up the cohort is a
            # step towards an answer and never the answer itself when the
            # question names a measure.
            named_metric = (intent.metric if intent.metric in PERFORMANCE_METRICS
                            else resolve_measure(question, default=None))
            all_lookups = bool(plan.steps) and all(
                (registry.REGISTRY.get(s.tool) or NO_SPEC).agent
                == "data_discovery" for s in plan.steps)
            # Not when the question is about the record itself, and not when
            # it is about the cohort: a metric word can appear in both without
            # the question asking for that measurement. There the lookup, or
            # the selection, is the measurement.
            if (named_metric and all_lookups
                    and not asks_about_completeness(question)
                    and not asks_which_are_eligible(question)
                    and intent.output_format in ("number", "text", "table",
                                                 "entity", "ranking")):
                last = plan.steps[-1]
                cohort = (last.args.get("building_ids")
                          or (intent.entities or {}).get("building_ids")
                          or f"${last.id}.column.building_id")
                period = intent.time_range or {}
                window = (period.get("start"), period.get("end"))
                # The performance tool takes a year and nothing finer, so a
                # total over a named month belongs to the windowed tool, which
                # takes the period the question gave.
                if named_metric == "total_kwh" and any(window):
                    measure = PlanStep(
                        f"{last.id}_measure", "calculate_consumption_statistics",
                        {"building_ids": cohort, "metrics": ["total"],
                         "start": window[0], "end": window[1]},
                        depends_on=[last.id], resolution="compute",
                        purpose="the total over the period the question named")
                    plan.add(measure)
                    plan.answer_step = measure.id
                    plan.answer_field = "total"
                    trace.note(
                        f"the plan looked the cohort up and measured nothing; "
                        f"added the total over {window[0]}..{window[1]}")
                    return_early = True
                else:
                    return_early = False
                args = {"building_ids": cohort, "metric": named_metric}
                # An intensity is a per-year figure, so it needs a year; the
                # resolved period says which, and otherwise the latest full
                # one is what "annual" means here.
                if named_metric in ANNUAL_METRICS:
                    year = (intent.time_range or {}).get("year") or LATEST_FULL_YEAR
                    args["year"] = int(year)
                if not return_early:
                    plan.add(PlanStep(
                        f"{last.id}_measure", "calculate_normalized_performance",
                        args, depends_on=[last.id], resolution="compute",
                        purpose=f"the {named_metric} the question asked for"))
                    plan.answer_step = f"{last.id}_measure"
                    trace.note(f"the plan looked the cohort up and measured "
                               f"nothing; added {named_metric}")

            # A plan that only finds the cohort has not answered a question
            # that ranks it: the measure has to be computed, and a superlative
            # settles the ranking's direction and size, which the plan does not
            # get to disagree with. A limit the size of the cohort is not a
            # limit. "At least" and "at most" are bounds, not superlatives.
            superlative = re.search(r"\bhighest|\blargest|(?<!at )\bmost\b|\btop\b"
                                    r"|\blowest|\bsmallest|(?<!at )\bleast\b"
                                    r"|\bbottom\b",
                                    question.lower())
            if superlative:
                wants_low = bool(re.search(r"\blowest|\bsmallest|\bleast\b"
                                           r"|\bbottom\b", question.lower()))
                asked_for = (intent.entities or {}).get("limit")
                for step in plan.steps:
                    if step.tool not in ("rank_buildings", "identify_peer_outliers"):
                        continue
                    if step.args.get("order") and (
                            step.args["order"] == "asc") != wants_low:
                        step.args["order"] = "asc" if wants_low else "desc"
                        trace.note(f"{step.id}: the question asks for the "
                                   f"{'lowest' if wants_low else 'highest'}, so "
                                   f"the ranking runs that way")
                    # ...where the tool has one. identify_peer_outliers
                    # reports every member that stands out, and writing a
                    # limit into it is an argument error rather than a
                    # narrower answer.
                    spec_step = registry.REGISTRY.get(step.tool)
                    takes_limit = "limit" in ((spec_step.parameters or {})
                                              .get("properties", {})
                                              if spec_step else {})
                    limit = step.args.get("limit")
                    if takes_limit and not asked_for and (limit is None
                                                          or limit >= 100):
                        step.args["limit"] = 10
                        trace.note(f"{step.id}: limit={limit} is the size of the "
                                   f"cohort, not how many to return; a "
                                   f"superlative with no number named is ten")

            # A question counting types, cities, sites or countries is
            # answered from the catalog, not by counting buildings. It applies
            # only when the question counts that entity: the counting phrase
            # has to reach the entity's own word, and a question pinning the
            # entity to one value -- site Bobcat -- is about that one, not the
            # set.
            noun = {"building_types": r"types?\b", "cities": r"cit(?:y|ies)\b",
                    "countries": r"countr(?:y|ies)\b", "sites": r"sites?\b",
                    "datasets": r"datasets?\b"}.get(intent.subject or "")
            pinned = {"cities": "city", "countries": "country",
                      "sites": "site_id",
                      "building_types": "building_type"}.get(intent.subject or "")
            counting = bool(noun and re.search(
                r"\b(?:how\s+many|number\s+of|count\s+of|how\s+much)\b"
                r"[^.?]{0,40}?" + noun, question.lower()))
            if (counting and intent.output_format in ("number", "entity")
                    and not (pinned and (intent.entities or {}).get(pinned))):
                for step in plan.steps:
                    if step.tool != "find_buildings":
                        continue
                    filters = {name: step.args.get(name)
                               or (intent.entities or {}).get(name)
                               for name in ("country", "city", "site_id",
                                            "building_type")}
                    step.tool = "search_catalog"
                    step.args = {"entity": intent.subject, **filters}
                    step.resolution = "graph_only"
                    plan.answer_step = step.id
                    trace.note(f"{step.id}: the question counts "
                               f"{intent.subject}, which the catalog holds; "
                               f"counting buildings answers a different "
                               f"question")
                    break

            # The subject names the grain of the answer, wherever a tool has
            # one. "Which usage category shows the highest demand variability"
            # is one row per building type; the plan computed one row per
            # building, and the answer that came back was a building.
            subject_grain = {"sites": "site", "cities": "city",
                             "countries": "country",
                             "building_types": "building_type"}.get(
                                 intent.subject or "")
            if subject_grain:
                for step in plan.steps:
                    spec_here = registry.REGISTRY.get(step.tool)
                    allowed = (((spec_here.parameters or {}).get("properties", {})
                                .get("across", {}).get("enum") or [])
                               if spec_here else [])
                    if subject_grain in allowed and step.args.get("across") != subject_grain:
                        step.args["across"] = subject_grain
                        trace.note(f"{step.id}: the question asks about "
                                   f"{intent.subject}, so the statistics are "
                                   f"one row per {subject_grain}")

            # Computing the measure is not the same as ranking it. A plan of
            # find_buildings then calculate_normalized_performance hands back
            # every eligible building -- 123 of them for "which have the
            # highest kWh/person" -- because that tool sorts nothing and keeps
            # everything. What counts as having answered a superlative is a
            # step that selects: a ranking, or a threshold on the measure.
            selects = any(
                s.tool in ("rank_buildings", "identify_peer_outliers")
                or s.args.get("at_least") is not None
                or s.args.get("at_most") is not None
                for s in plan.steps)
            # ...and only where the thing being ranked is buildings. "Which
            # month had the highest use for Rat_office_Adele" is a superlative
            # over the calendar, and appending a building ranking answered it
            # with the building's own name -- the top of a ranking of one.
            ranks_buildings = (intent.subject == "buildings"
                               or bool(re.search(r"\bwhich\s+buildings?\b",
                                                 question.lower())))
            names_calendar = bool(re.search(
                r"\bwhich\s+(?:month|day|hour|season|year|week)\b"
                r"|\bwhat\s+(?:month|day|hour|season|year|week)\b",
                question.lower()))
            if (plan.steps and superlative and not selects and ranks_buildings
                    and not names_calendar
                    and intent.metric in ("eui", "rank", "per_occupant")):
                last = plan.steps[-1]
                spec_last = registry.REGISTRY.get(last.tool)
                if spec_last and spec_last.agent in ("data_discovery", "performance"):
                    wanted = (intent.entities or {}).get("limit") or 10
                    if "limit" in ((spec_last.parameters or {})
                                   .get("properties", {})):
                        last.args["limit"] = 5000
                    plan.add(PlanStep(
                        f"{last.id}_rank", "rank_buildings",
                        {"building_ids": f"${last.id}.column.building_id",
                         "metric": resolve_measure(question),
                         "order": "asc" if re.search(r"\blowest|\bleast\b",
                                                     question.lower()) else "desc",
                         "limit": wanted, "year": LATEST_FULL_YEAR},
                        depends_on=[last.id],
                        purpose="the measure the question ranks on, which "
                                "finding the cohort does not compute"))
                    plan.answer_step = f"{last.id}_rank"
                    trace.note(f"{last.id}: the plan found the cohort but never "
                               f"ranked it; added the ranking the question asked "
                               f"for")

            # "More than three consecutive days" is four. A bound the tool
            # names at_least is inclusive, and the question's "more than" is
            # not: asked for buildings with more than 3 consecutive zero days,
            # the plan wrote 3 and answered with the 71 that have three or
            # more where the question asks after 49.
            more_than = re.search(r"\b(?:more\s+than|over|longer\s+than|greater\s+"
                                  r"than|exceed\w*)\s+(\d+)\b", question.lower())
            if more_than:
                stated = int(more_than.group(1))
                for step in plan.steps:
                    for name, value in list(step.args.items()):
                        if (name.endswith("_at_least")
                                and isinstance(value, int) and not isinstance(value, bool)
                                and value == stated):
                            step.args[name] = stated + 1
                            trace.note(f"{step.id}: {name}={stated} is "
                                       f"'at least {stated}', and the question "
                                       f"asks for more than {stated}, so "
                                       f"{stated + 1}")

            # "A year of continuous data" is a definition with three numbers
            # in it, and a plan that keeps one of them answers a different
            # question: gap_at_most_hours=0 asks for a perfect record and
            # finds 113 buildings where 628 have a continuous year.
            if re.search(r"\bcontinuous\b|\buninterrupted\b|\bwithout\s+gaps?\b",
                         question.lower()) and not re.search(
                             r"\b\d+\s*(?:hours?|hrs?)\b", question.lower()):
                for step in plan.steps:
                    if (step.tool == "find_buildings"
                            and step.args.get("min_months_of_data")):
                        changed = []
                        if step.args.get("gap_at_most_hours") != 24:
                            step.args["gap_at_most_hours"] = 24
                            changed.append("a break of at most a day")
                        if step.args.get("min_coverage") is None:
                            step.args["min_coverage"] = 0.99
                            changed.append("99% of the hours present")
                        if changed:
                            trace.note(f"{step.id}: continuous means "
                                       f"{' and '.join(changed)}")

            # Imputation is three steps and a plan that omits the middle one
            # is refused by the tool it reaches: gaps here are absent rows,
            # not empty values, so a series has to be put on a regular grid
            # before anything can fill it. The rule says so and the plan wrote
            # fetch then impute anyway, twice over.
            for position, step in enumerate(list(plan.steps)):
                if step.tool != "impute_missing_values":
                    continue
                if any(other.tool == "resample_timeseries" for other in plan.steps):
                    continue
                upstream = next((earlier.id for earlier in plan.steps[:position]
                                 if earlier.tool.startswith("fetch_")), None)
                if not upstream:
                    continue
                grid = PlanStep(f"{step.id}_grid", "resample_timeseries",
                                {"handle": f"${upstream}.handle",
                                 "resolution": "hourly", "how": "mean"},
                                depends_on=[upstream],
                                purpose="put the series on a regular grid, so "
                                        "the gaps are rows that can be filled")
                plan.steps.insert(position, grid)
                step.args["handle"] = f"${grid.id}.handle"
                step.depends_on = [grid.id]
                trace.note(f"{step.id}: a gap here is an absent row; added the "
                           f"resampling step that turns it into one that can "
                           f"be filled")
                break

            # "Typically" means the average day, not the single biggest one.
            # calculate_demand_metrics reports the hour of the highest reading
            # in the record; the hour a building typically peaks is where its
            # mean profile peaks, and for Lamb_office_Bertha those are 12 and
            # 10. The tool that draws the profile answers the second.
            if re.search(r"\btypical\w*|\busually\b|\bon\s+average\b"
                         r"|\baverage\s+day\b", question.lower()) and re.search(
                             r"\bpeak\b|\bhighest\b|\bmaximum\b", question.lower()):
                # The repair named one tool, so it only caught the plan the
                # planner happened to write that week. Any tool that reports
                # the record's extreme -- or does not report an hour at all --
                # answers a different question from "when does it typically
                # peak": multi-kg reached fetch_electricity then normalised
                # performance and replied with the building's NAME.
                for step in plan.steps:
                    if step.tool not in ("calculate_demand_metrics",
                                         "fetch_electricity",
                                         "calculate_normalized_performance",
                                         "calculate_consumption_statistics"):
                        continue
                    cohort = step.args.get("building_ids")
                    step.tool = "calculate_load_profile"
                    step.args = {"building_ids": cohort, "profile": "hourly",
                                 "peak_only": True,
                                 **{k: v for k, v in step.args.items()
                                    if k in ("start", "end")}}
                    plan.answer_step = step.id
                    # Steps after it consumed the raw series and would
                    # otherwise overwrite the answer with their own shape.
                    # Presentation steps are kept: a chart of the profile is
                    # still a chart of the profile, and dropping it would
                    # silently refuse a drawing the question asked for.
                    presents = ("create_visualization", "generate_report")
                    plan.steps = [x for x in plan.steps
                                  if x.id == step.id
                                  or x.tool in presents
                                  or not (x.depends_on
                                          and step.id in x.depends_on)]
                    trace.note(f"{step.id}: a typical peak is where the average "
                               f"day peaks, not when the record's highest "
                               f"reading happened")
                    break

            # "What will the consumption be FROM 8 May TO 14 May" names the
            # week to forecast, not the history to forecast from. The plan set
            # the cutoff to the window's end and forecast 15-21 May instead --
            # and with a seasonal-naive model at a 168-hour period that copies
            # the previous week, so the total came back equal to the answer for
            # the right week and looked correct. A number that agrees for the
            # wrong reason is worse than one that disagrees.
            window = intent.time_range or {}
            if window.get("start") and window.get("end") and not re.search(
                    r"\bnext\b|\bfollowing\b|\bahead\b|\bsubsequent\b|\bafter\b",
                    question.lower()):
                try:
                    import pandas as _pd
                    begin = _pd.Timestamp(window["start"])
                    finish = _pd.Timestamp(window["end"])
                    span = int((finish - begin).total_seconds() // 3600)
                except Exception:                           # noqa: BLE001
                    begin = finish = None
                    span = 0
                if span > 0:
                    for step in plan.steps:
                        if step.tool != "forecast_timeseries":
                            continue
                        cutoff = step.args.get("cutoff")
                        moved = False
                        if cutoff and str(cutoff)[:10] >= str(finish)[:10]:
                            step.args["cutoff"] = str(begin)[:10]
                            moved = True
                            trace.note(
                                f"{step.id}: the question names "
                                f"{str(begin)[:10]} to {str(finish)[:10]} as the "
                                f"period to forecast; the cutoff sat at its end, "
                                f"which forecast the period after it")
                        # A horizon shorter than the named period answers for
                        # part of it: 24 hours offered for a seven-day question
                        # is a day's answer wearing the question's dates.
                        if (moved or str(step.args.get("cutoff") or "")[:10]
                                == str(begin)[:10]):
                            if int(step.args.get("horizon_hours") or 0) < span:
                                step.args["horizon_hours"] = span
                                trace.note(
                                    f"{step.id}: horizon widened to {span} h, the "
                                    f"length of the period the question names")

            # "What proportion of consumption fell in the top 5% of hours" is
            # one column the catalogue computes -- share_in_top_hours. The
            # planning rules name it, and the model still built six steps that
            # ended on mean air temperature. A positive route is sturdier than
            # a rule: the measure is named in the question, so it can be read
            # out of it.
            share_pct = resolve_top_hours_share(question)
            existing = next((x for x in plan.steps
                             if "top_hours_share_pct" in (x.args or {})), None)
            if share_pct is not None and existing is not None:
                # The measure is already being computed; it is simply not the
                # step the answer is taken from. The plan even said so -- "step
                # 1 calculates the share, step 3 combines these" -- and step 3
                # combined nothing, so a total in kWh was returned for a
                # question asking for a fraction.
                order = [x.id for x in plan.steps]
                after = set(order[order.index(existing.id) + 1:])
                keep = [x for x in plan.steps
                        if x.id not in after
                        or not (x.tool.startswith("calculate_")
                                or x.tool.startswith("generate_")
                                or x.tool.startswith("fetch_"))]
                plan.steps = keep
                plan.fill_agents()
                plan.answer_step = existing.id
                trace.note(f"{existing.id}: this step already computes the "
                           f"share of consumption in the top {share_pct:g}% of "
                           f"hours; the steps after it answered something else")
            elif share_pct is not None:
                cohort = next((x.args.get("building_ids") or x.args.get("building_id")
                               for x in plan.steps
                               if (x.args or {}).get("building_ids")
                               or (x.args or {}).get("building_id")), None)
                if cohort is not None:
                    if isinstance(cohort, str):
                        cohort = [cohort]
                    keep = [x for x in plan.steps if x.tool == "find_buildings"]
                    args = {"building_ids": cohort,
                            "top_hours_share_pct": float(share_pct)}
                    for key in ("start", "end"):
                        value = (intent.time_range or {}).get(key)
                        if value:
                            args[key] = value
                    step = PlanStep(f"s{len(keep) + 1}", "calculate_demand_metrics",
                                    args, depends_on=[x.id for x in keep],
                                    purpose=f"share of consumption in the top "
                                            f"{share_pct:g}% of hours")
                    plan.steps = keep + [step]
                    plan.fill_agents()
                    plan.answer_step = step.id
                    trace.note(f"{step.id}: the question asks for the share of "
                               f"consumption in the top {share_pct:g}% of hours, "
                               f"which is one measure the catalogue reports")

            # A question that names the hours -- "nighttime hours from 8 PM to
            # 7 AM" -- is asking about those hours only. The model's plan left
            # the window off and scanned the whole clock, so 3,946 hours came
            # back where 1,007 night hours were asked for: a bigger number
            # answering a broader question, with nothing in the reply to say so.
            clock = resolve_clock_window(question)
            if clock:
                for step in plan.steps:
                    if step.tool != "detect_anomalies":
                        continue
                    if step.args.get("hour_from") is None:
                        step.args["hour_from"], step.args["hour_to"] = clock
                        trace.note(f"{step.id}: restricted to hours "
                                   f"{clock[0]:02d}:00-{clock[1]:02d}:59, which "
                                   f"the question names")
                    break

            # hour_from/hour_to are real parameters -- on
            # calculate_consumption_statistics and detect_anomalies, not on
            # fetch_electricity. A planner that reads "between 2 PM and 6 PM"
            # correctly and writes the window onto the fetch step is right
            # about the question and wrong about one tool, and the run died on
            # "unknown parameter 'hour_from'". The window is what matters, so
            # the step is moved to the tool that honours it.
            clock_hosts = {"calculate_consumption_statistics",
                           "detect_anomalies", "calculate_demand_metrics"}
            for step in plan.steps:
                if step.tool in clock_hosts:
                    continue
                if step.args.get("hour_from") is None \
                        and step.args.get("hour_to") is None:
                    continue
                spec = registry.get(step.tool)
                accepted = set(
                    (getattr(spec, "parameters", {}) or {}).get("properties", {}))
                if "hour_from" in accepted:
                    continue
                window = (step.args.pop("hour_from", None),
                          step.args.pop("hour_to", None))
                if step.tool == "fetch_electricity":
                    moved = dict(step.args)
                    moved["hour_from"], moved["hour_to"] = window
                    moved.pop("resolution", None)
                    moved.pop("timezone", None)
                    step.tool = "calculate_consumption_statistics"
                    step.args = moved
                    trace.note(
                        f"{step.id}: the question names the hours "
                        f"{window[0]}:00-{window[1]}:00; fetch_electricity "
                        f"cannot restrict to them, so the step is answered "
                        f"by calculate_consumption_statistics")
                else:
                    trace.note(f"{step.id}: dropped an hour window that "
                               f"{step.tool} does not accept")

            # A stated origin makes this a forecast, not a backtest. The
            # rolling origin answers "how accurately does this model forecast
            # this building over March and April"; the question here says
            # "the next 24 hours starting at 00:00 on 1 March 2016", which is
            # one forecast issued once and scored once. The two give
            # different numbers, and the backtest costs up to 200x more.
            origin = resolve_forecast_origin(question)
            horizon = resolve_horizon_hours(question)
            season = resolve_season_length(question)
            # A season named in the question reaches every forecast step,
            # rerouted or not: "Seasonal Naive with a 24-hour season" is a
            # different forecast from the weekly default, 0.1417 against
            # 0.2250 on one building over one horizon.
            if season is not None:
                for step in plan.steps:
                    if (step.tool == "forecast_timeseries"
                            and step.args.get("season_length") is None):
                        step.args["season_length"] = season
                        trace.note(f"{step.id}: the question names a "
                                   f"{season}-hour season")
            if origin and horizon:
                backtests = [x for x in plan.steps
                             if x.tool == "backtest_forecast_models"]
                for step in backtests:
                    models = (step.args.get("models")
                              or resolve_forecast_models(question)
                              or [step.args.get("model") or "seasonal_naive"])
                    cohort = (step.args.get("building_ids")
                              or ([step.args["building_id"]]
                                  if step.args.get("building_id") else None))
                    keep = [x for x in plan.steps if x is not step]
                    added = []
                    for n, name in enumerate(models, 1):
                        args = {"model": name, "cutoff": origin,
                                "horizon_hours": horizon}
                        if season is not None:
                            args["season_length"] = season
                        if cohort and len(cohort) == 1:
                            args["building_id"] = cohort[0]
                        elif cohort:
                            args["building_ids"] = cohort
                        fid = f"{step.id}_f{n}"
                        added.append(PlanStep(
                            fid, "forecast_timeseries", args,
                            depends_on=[d for d in step.depends_on
                                        if d in {x.id for x in keep}],
                            purpose=f"{name} from the stated origin {origin}"))
                        added.append(PlanStep(
                            f"{step.id}_e{n}", "evaluate_forecast",
                            {"forecast_handle": f"${fid}.handle"},
                            depends_on=[fid],
                            purpose=f"score {name} against what happened"))
                    # Whatever depended on the backtest has to be pointed at
                    # what replaced it. Without this the plan still names a
                    # step that no longer exists -- "3: depends on missing
                    # step 2" -- and validation throws the whole plan away,
                    # so a question that was one reroute from being right
                    # came back "no analysis could be run".
                    for other in keep:
                        if step.id in other.depends_on:
                            other.depends_on = [d for d in other.depends_on
                                                if d != step.id] + [added[-1].id]
                    plan.steps = keep + added
                    plan.fill_agents()
                    plan.answer_step = added[-1].id
                    trace.note(
                        f"{step.id}: the question names one origin "
                        f"({origin}) and a {horizon}-hour horizon, so it is a "
                        f"forecast scored once, not a rolling-origin backtest")
                    break

            # A count of period-over-period moves is not an anomaly scan and
            # not a change-point search, whatever the plan reached for. Both
            # return a number for this question and neither returns this one:
            # first_difference thresholds in sigma of the building's own
            # changes, and detect_change_points finds at most one shift per
            # building by construction.
            step_change = resolve_step_change(question)
            if step_change and not any(x.tool == "count_step_changes"
                                       for x in plan.steps):
                # Not a list of wrong tools: any tool is the wrong tool here.
                # Naming a family of them -- the anomaly tools, say -- lets a
                # plan that reached for calculate_consumption_statistics sail
                # past and answer "how many times did it rise by 20%" with the
                # total consumption. What decides is that the question counts
                # period-over-period moves, not what the model happened to
                # pick.
                wrong = [x for x in plan.steps if x.tool != "find_buildings"]
                if wrong:
                    keep = [x for x in plan.steps if x not in wrong]
                    donor = wrong[-1]
                    args = dict(step_change)
                    for key in ("building_ids", "building_id", "start", "end"):
                        value = donor.args.get(key)
                        if value is None:
                            continue
                        if key == "building_id":
                            args["building_ids"] = [value]
                        else:
                            args[key] = value
                    if not args.get("building_ids"):
                        args["building_ids"] = next(
                            (x.args.get("building_ids") or
                             ([x.args["building_id"]] if x.args.get("building_id")
                              else None)
                             for x in plan.steps
                             if (x.args or {}).get("building_ids")
                             or (x.args or {}).get("building_id")), None)
                    for key in ("start", "end"):
                        if key not in args and (intent.time_range or {}).get(key):
                            args[key] = intent.time_range[key]
                    # Only the steps that survive can be depended on: the
                    # donor's own dependencies usually include the fetch that
                    # was just removed, and a dangling id fails validation --
                    # which showed up as a plan with no steps at all and
                    # "no analysis could be run".
                    surviving = {x.id for x in keep}
                    added = PlanStep(f"{donor.id}_steps", "count_step_changes",
                                     args,
                                     depends_on=[d for d in donor.depends_on
                                                 if d in surviving],
                                     purpose=f"how many {step_change['grain']} "
                                             f"periods moved more than "
                                             f"{step_change['change_at_least_pct']:g}% "
                                             f"against the one before")
                    plan.steps = keep + [added]
                    plan.fill_agents()
                    plan.answer_step = added.id
                    trace.note(f"{added.id}: the question counts "
                               f"{step_change['grain']} moves past a stated "
                               f"percentage, which is a count on that scale "
                               f"rather than an anomaly test in sigma")

            # "Gaps longer than seven consecutive days" selects the broken
            # streams, and the parameter for it is gap_longer_than_hours.
            # gap_at_most_hours is its opposite and returns the sound ones, so
            # a plan that reaches for the wrong one answers "which buildings
            # are broken" with the buildings that are fine.
            gap = resolve_gap_hours(question)
            if gap is not None:
                for step in plan.steps:
                    if step.tool != "find_buildings":
                        continue
                    if step.args.get("gap_longer_than_hours") is None:
                        step.args.pop("gap_at_most_hours", None)
                        step.args["gap_longer_than_hours"] = gap
                        trace.note(f"{step.id}: the question asks which "
                                   f"buildings have a gap longer than {gap} "
                                   f"hours, which selects the broken streams")

            # The hour of the highest average is not the peak. A peak is the
            # single largest reading -- one hour of one day, 195.4 kW here --
            # and the question asks which hour of the day is highest on
            # average across the year, which is 11:00 at 123.6.
            if ASKS_PEAK_HOUR.search(question or ""):
                for step in plan.steps:
                    if step.tool != "calculate_demand_metrics":
                        continue
                    cohort = (step.args.get("building_ids")
                              or ([step.args["building_id"]]
                                  if step.args.get("building_id") else None))
                    low = re.search(r"\blowest\b|\bminimum\b|\bleast\b",
                                    (question or "").lower())
                    args = {"across": "hour_of_day", "metrics": ["mean"],
                            "order_by": "mean",
                            "order": "asc" if low else "desc", "limit": 1}
                    if cohort:
                        args["building_ids"] = cohort
                    for key in ("start", "end"):
                        if step.args.get(key):
                            args[key] = step.args[key]
                        elif (intent.time_range or {}).get(key):
                            args[key] = intent.time_range[key]
                    keep = [x for x in plan.steps if x is not step]
                    added = PlanStep(f"{step.id}_hour",
                                     "calculate_consumption_statistics", args,
                                     depends_on=[d for d in step.depends_on
                                                 if d in {x.id for x in keep}],
                                     purpose="the hour of day with the "
                                             "highest average demand")
                    for other in keep:
                        if step.id in other.depends_on:
                            other.depends_on = [d for d in other.depends_on
                                                if d != step.id] + [added.id]
                    plan.steps = keep + [added]
                    plan.fill_agents()
                    plan.answer_step = added.id
                    trace.note(f"{step.id}: the question asks which hour is "
                               f"highest on average, not what the peak was")
                    break

            # A question that says "cluster" is answered by clustering. Asked
            # to cluster one building's daily profiles the plan fetched the
            # readings and drew a chart: no partition, no counts, and nothing
            # in the reply to say the question had not been answered.
            if ASKS_CLUSTERING.search(question or "") and not any(
                    x.tool in ("cluster_timeseries", "compare_clustering_methods",
                               "compare_cluster_stability")
                    for x in plan.steps):
                cohort = next((x.args.get("building_ids")
                               or ([x.args["building_id"]]
                                   if x.args.get("building_id") else None)
                               for x in plan.steps
                               if (x.args or {}).get("building_ids")
                               or (x.args or {}).get("building_id")), None)
                if cohort:
                    one = len(cohort) == 1
                    args = {"building_ids": cohort,
                            "scope": "within_building" if one else "across_buildings",
                            "n_clusters": resolve_n_clusters(question),
                            "method": resolve_cluster_method(question)[0],
                            "distance": resolve_cluster_distance(question),
                            # Across a cohort, normalising is required or
                            # the clusters rank by floor area. Within one
                            # building every day comes off the same meter,
                            # so it is done only when asked -- and "raw,
                            # un-normalised" is not asking.
                            "normalize": ("mean" if (not one
                                                     or asks_normalised(question))
                                          else "none")}
                    for key in ("start", "end"):
                        if (intent.time_range or {}).get(key):
                            args[key] = intent.time_range[key]
                    keep = [x for x in plan.steps
                            if x.tool in ("find_buildings", "fetch_electricity")]
                    added = PlanStep("s_cluster", "cluster_timeseries", args,
                                     depends_on=[], purpose="group the daily "
                                                 "load profiles")
                    plan.steps = keep + [added]
                    plan.fill_agents()
                    plan.answer_step = added.id
                    trace.note("s_cluster: the question asks for a clustering "
                               "and the plan contained none")

            # "Above the 90th percentile" names a point; identify_peer_outliers
            # takes the WIDTH of the tail beyond it. Passing 90 asks for the
            # top 90% and returned 1,188 buildings for a question whose answer
            # is 139. resolve_tail has read this correctly all along and
            # nothing applied it.
            tail = resolve_tail(question)
            if tail:
                for step in plan.steps:
                    if step.tool != "identify_peer_outliers":
                        continue
                    if step.args.get("method") in (None, "percentile"):
                        if step.args.get("threshold") != tail["threshold"]:
                            step.args["threshold"] = tail["threshold"]
                            trace.note(f"{step.id}: the tail beyond the "
                                       f"percentile is {tail['threshold']}% of "
                                       f"the group, not {90 if tail['threshold'] != 90 else ''}"
                                       f" the percentile itself")
                        step.args.setdefault("side", tail["side"])

            # A minimum peer-group size is a guard against a percentile drawn
            # over three buildings, and it is the right default. It is not
            # right when the question says to use every category: with the
            # guard at 10 the small uses are dropped and the answer is 134
            # buildings where every category gives 139.
            if re.search(r"\bevery\s+(?:use\s+)?categor\w+|\ball\s+categor\w+"
                         r"|\beach\s+(?:use\s+)?categor\w+",
                         (question or "").lower()):
                for step in plan.steps:
                    if (step.tool == "identify_peer_outliers"
                            and (step.args.get("min_band_size") or 1) > 1):
                        step.args["min_band_size"] = 1
                        trace.note(f"{step.id}: every use category is scored, "
                                   f"including the small ones, as the question "
                                   f"states")

            # The age cohorts the question names, written onto the steps that
            # select them: a minimum and a maximum set to the same year is one
            # year, not the two cohorts a comparison asked for.
            vintages = resolve_vintages(question)
            if len(vintages) == 2:
                cohorts = [x for x in plan.steps if x.tool == "find_buildings"]
                if len(cohorts) == 2:
                    for step, (label, parameter, value) in zip(cohorts, vintages):
                        before = {k: step.args.get(k) for k in
                                  ("min_year_built", "max_year_built")}
                        step.args.pop("min_year_built", None)
                        step.args.pop("max_year_built", None)
                        step.args[parameter] = value
                        if before != {"min_year_built": step.args.get("min_year_built"),
                                      "max_year_built": step.args.get("max_year_built")}:
                            trace.note(f"{step.id}: the {label} cohort is "
                                       f"{parameter}={value}, as the question "
                                       f"states it")

            # The same rule for the building-to-buildings tool.
            if re.search(r"\bmost\s+similar\b|\bclosest\b|\bmost\s+alike\b",
                         (question or "").lower()) and not re.search(
                             r"\bcorrelat\w+", (question or "").lower()):
                for step in plan.steps:
                    if (step.tool == "find_similar_timeseries"
                            and step.args.get("metric") in (None, "correlation")):
                        step.args["metric"] = "euclidean"
                        trace.note(f"{step.id}: \"most similar\" is a distance "
                                   f"between profiles, not a correlation")

            # A constraint on which pairs are eligible is part of the
            # question, not a preference. correlate_group_profiles has taken
            # require_different since it was written and nothing ever passed
            # it, so "two cities in different countries" was answered with
            # whichever pair scored highest -- usually two cities in one
            # country, which is the answer the question rules out.
            day_type = resolve_day_type(question)
            for step in plan.steps:
                if step.tool != "correlate_group_profiles":
                    continue
                wanted = resolve_require_different(
                    question, step.args.get("group_by") or "building_type")
                if wanted != "none" and step.args.get("require_different") in (
                        None, "none"):
                    step.args["require_different"] = wanted
                    trace.note(f"{step.id}: only pairs in different "
                               f"{wanted.replace('_', ' ')}s are eligible, as "
                               f"the question states")
                if day_type != "all" and step.args.get("days") in (None, "all"):
                    step.args["days"] = day_type
                    trace.note(f"{step.id}: profiles built from {day_type}s "
                               f"only, as the question states")
                # A "load profile" is the 24-hour shape, not the year of
                # hours it was averaged from. Compared as raw series the
                # nearest cross-country pair is Ithaca and London; compared
                # as profiles it is Dublin and Tempe, which is the question.
                if (re.search(r"\bprofiles?\b", (question or "").lower())
                        and step.args.get("shape") in (None, "series")):
                    step.args["shape"] = "daily_profile"
                    trace.note(f"{step.id}: compared as 24-hour load profiles, "
                               f"which is what the question calls a profile")
                # "Most similar" is a distance, not a correlation -- the tool's
                # own docstring says so and nothing acted on it. Two cities can
                # move in step and draw quite different amounts.
                if (re.search(r"\bmost\s+similar\b|\bclosest\b|\bmost\s+alike\b",
                              (question or "").lower())
                        and not re.search(r"\bcorrelat\w+", (question or "").lower())
                        and step.args.get("metric") in (None, "pearson")):
                    step.args["metric"] = "euclidean"
                    trace.note(f"{step.id}: \"most similar\" is a distance "
                               f"between profiles, not a correlation between "
                               f"them")

            # The same window has to reach a plain statistics question.
            # "Highest average consumption between 2 PM and 6 PM" ranked on the
            # whole day's mean, which is a different ranking: the afternoon
            # peak is exactly what separates these buildings, and averaging it
            # with the night hides it. calculate_consumption_statistics takes
            # the window, so a question that states one states it here too.
            if clock:
                for step in plan.steps:
                    if (step.tool == "calculate_consumption_statistics"
                            and step.args.get("hour_from") is None):
                        step.args["hour_from"], step.args["hour_to"] = clock
                        trace.note(f"{step.id}: averaged over hours "
                                   f"{clock[0]:02d}:00-{clock[1]:02d}:59 only, "
                                   f"which the question names")

            # detect_anomalies is a legal tool for an anomaly question, so the
            # repair table accepts it -- but for a question about a sustained
            # SHIFT it is the wrong legal tool. A step change shows up as no
            # anomalies at all once the new level becomes the norm, which is
            # why q12 answered "no days found" over six months containing three
            # documented shifts. Being permitted is not the same as being right.
            if asks_for_change_point(question):
                for step in plan.steps:
                    if step.tool != "detect_anomalies":
                        continue
                    step.tool = "detect_change_points"
                    step.args = {k: v for k, v in (step.args or {}).items()
                                 if k in ("building_ids", "min_segment_days",
                                          "min_shift_pct")}
                    plan.answer_step = step.id
                    trace.note(f"{step.id}: the question asks about a sustained "
                               f"shift; an anomaly scan finds none once the new "
                               f"level becomes normal")
                    break

            # Weather may be fetched as working data, but it cannot be the
            # answer to a question that never mentions it. The test is on the
            # question, not the tool -- and it allows for questions about
            # temperature that never use the word, such as a split into cool
            # and warm hours.
            if not (re.search(r"\bweather\b|\btemperature\b|\bdegree[-\s]?days?\b"
                              r"|\bhumid\w*|\bwind\b|\bcloud\w*|\bdew\b"
                              r"|\bheating\b|\bcooling\b|\bclimat\w*",
                              question.lower())
                    or resolve_temperature_split(question)):
                weather_steps = [x for x in plan.steps
                                 if "weather" in (x.tool or "")]
                if weather_steps and plan.steps and "weather" in (plan.steps[-1].tool or ""):
                    keep = [x for x in plan.steps if "weather" not in (x.tool or "")]
                    if keep:
                        plan.steps = keep
                        plan.fill_agents()
                        plan.answer_step = keep[-1].id
                        trace.note(
                            f"dropped {len(weather_steps)} weather step(s): the "
                            f"question does not ask about weather, and the "
                            f"answer was being taken from one of them")

            # "How many education buildings at each site" is a catalogue count,
            # and the catalogue answers it exactly -- 111 at Rat, 83 at Lamb.
            # The plan ran search_catalog and then appended a consumption step,
            # which became the answer and reported n_observations per site:
            # reading counts where buildings were asked for, in kWh units, with
            # the right answer already computed one step earlier.
            if re.search(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\s+of\b",
                         question.lower()) and re.search(
                             r"\b(?:at|per|for|across|by)\s+each\b|\bper\s+\w+\b"
                             r"|\beach\s+(?:site|city|country|type|year)\b",
                             question.lower()):
                catalogue = next((x for x in plan.steps
                                  if x.tool == "search_catalog"), None)
                if catalogue is not None and plan.steps[-1].tool != "search_catalog":
                    # Anything that measures demand rather than counting
                    # rows in the catalogue.
                    order = [x.id for x in plan.steps]
                    after = order[order.index(catalogue.id) + 1:]
                    dropped = [x.id for x in plan.steps
                               if x.id in after
                               and (x.tool.startswith("calculate_")
                                    or x.tool.startswith("fetch_")
                                    or x.tool.startswith("analyze_"))]
                    plan.steps = [x for x in plan.steps if x.id not in dropped]
                    plan.answer_step = catalogue.id
                    trace.note(f"{catalogue.id}: the catalogue already counts "
                               f"this; the measurement step after it answered a "
                               f"different question and was dropped")

            # "The average 24-hour intraday pattern" is a mean profile: 24
            # rows, one per hour. fetch_electricity returns the raw series --
            # every hour of every day, tens of thousands of rows -- and a
            # chart drawn from it is a year of wiggles, not a day shape. The
            # planner reached for it after an unrelated tool gained an
            # argument, which is how little it takes to move an answer; the
            # repair lives here so the shape is settled by what was asked
            # rather than by the catalogue's wording on the day.
            if re.search(r"\b(average|mean|typical|intraday)\b",
                         question.lower()) and re.search(
                             r"\b24[-\s]?hour\b|\bintraday\b|\bdaily\b"
                             r"|\bhourly\b", question.lower()) and re.search(
                             r"\bpattern\b|\bprofile\b|\bshape\b|\bcurve\b",
                             question.lower()):
                for step in plan.steps:
                    if step.tool != "fetch_electricity":
                        continue
                    step.tool = "calculate_load_profile"
                    step.args = {"building_ids": step.args.get("building_ids"),
                                 "profile": "hourly",
                                 **{k: v for k, v in step.args.items()
                                    if k in ("start", "end")}}
                    plan.answer_step = step.id
                    trace.note(f"{step.id}: an average 24-hour pattern is a "
                               f"mean profile of 24 rows, not the raw series "
                               f"the chart would otherwise be drawn from")
                    break
                    break

            # A weather threshold names a contrast, not a correlation. "Did
            # consumption spike during the heatwave above 35C" planned a
            # relationship over the whole temperature range, whose answer says
            # nothing about the days above the line.
            heat = re.search(r"(?:above|over|exceed\w*|greater\s+than|>)\s*(?:a\s+)?"
                             r"(\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", question.lower())
            if heat:
                for step in plan.steps:
                    if (step.tool == "analyze_weather_relationship"
                            and step.args.get("method") != "threshold_contrast"):
                        step.args["method"] = "threshold_contrast"
                        step.args["threshold"] = float(heat.group(1))
                        step.args.setdefault("months",
                                             (intent.time_range or {}).get("months"))
                        plan.answer_step = step.id
                        trace.note(f"{step.id}: the question names a "
                                   f"{heat.group(1)}C line, so the answer is "
                                   f"the contrast across it")

            # An answer must not stop at the tool's default page size, whatever
            # its shape: a truncated list reads as the whole set, and a
            # truncated count reads as the answer.
            if (not (intent.entities or {}).get("limit")
                    and not re.search(r"\btop\b|\bfirst\b|\bhighest\b|\blowest\b",
                                      question.lower())):
                for step in plan.steps:
                    if step.tool == "find_buildings":
                        limit = step.args.get("limit")
                        if limit is None or limit < 1000:
                            step.args["limit"] = 5000
                            trace.note(f"{step.id}: limit={limit} truncates a "
                                       f"question that asked for all of them")

            # A grouping the question states is not the plan's to drop. "A
            # matrix of missing metadata by country" was planned without the
            # grouping and answered with one portfolio-wide figure.
            if intent.group_by:
                # The intent names the dimension in the plural a question uses
                # -- "countries" -- and the tools name it in the singular.
                wanted_group = {"sites": "site", "cities": "city",
                                "countries": "country",
                                "building_types": "building_type"}.get(
                                    intent.group_by, intent.group_by)
                for step in plan.steps:
                    spec_here = registry.REGISTRY.get(step.tool)
                    properties = ((spec_here.parameters or {}).get("properties", {})
                                  if spec_here else {})
                    allowed = properties.get("group_by", {}).get("enum") or []
                    # "none" is the tool's default written out, not a grouping
                    # the plan chose.
                    if (wanted_group in allowed
                            and step.args.get("group_by") in (None, "", "none")):
                        step.args["group_by"] = wanted_group
                        trace.note(f"{step.id}: the question asks for one row "
                                   f"per {wanted_group}, which the plan "
                                   f"left out")

            # The screen belongs to the step that selects, not the step that
            # computes. A plan of find -> compute -> rank has the computing
            # step apply a 0.9 coverage floor and hand 123 buildings to a
            # ranking whose own floor is 0 at the top -- and the ranking cannot
            # recover what it never saw. Asked which buildings use most per
            # occupant, that dropped one of the ten.
            feeds_a_ranking = {
                str(step.args.get("building_ids", "")).lstrip("$").split(".")[0]
                for step in plan.steps if step.tool == "rank_buildings"}
            for step in plan.steps:
                if (step.tool == "calculate_normalized_performance"
                        and step.id in feeds_a_ranking
                        and step.args.get("min_coverage") != 0.0):
                    step.args["min_coverage"] = 0.0
                    trace.note(f"{step.id}: it computes and the ranking "
                               f"selects, so the coverage floor is the "
                               f"ranking's to apply")

            # A question that asks which buildings match a rule is answered by
            # the set, not by a ranking of it. "Find buildings with at least 12
            # months of continuous data" was planned as the right filter and
            # then a top-ten by intensity, which answers a question nobody
            # asked and hides 618 of the 628.
            if (not re.search(r"\bhighest\b|\blowest\b|\btop\b|\bbottom\b"
                              r"|\bworst\b|\bbest\b|\bmost\b|\bleast\b|\brank\w*",
                              question.lower())
                    and not (intent.entities or {}).get("limit")
                    and intent.output_format in ("list", "table")):
                answering = plan.answer_step or (plan.steps[-1].id if plan.steps else None)
                ranking = next((st for st in plan.steps
                                if st.id == answering and st.tool == "rank_buildings"),
                               None)
                cohort = next((st for st in reversed(plan.steps)
                               if st.tool == "find_buildings"), None)
                if ranking is not None and cohort is not None:
                    plan.answer_step = cohort.id
                    trace.note(f"{ranking.id}: the question asks which "
                               f"buildings match a rule, so the set is the "
                               f"answer and not a ranking of it")

            # A month is one of twelve. "From 2016 to 2017" came back as
            # months=[2016, 2017], which matches no month at all: the tool
            # found no measurements in the requested window and the answer was
            # a confident zero. The window the same plan passed already says
            # which years are meant.
            for step in plan.steps:
                months = step.args.get("months")
                if isinstance(months, list) and months:
                    real = [int(m) for m in months
                            if isinstance(m, (int, float))
                            and not isinstance(m, bool) and 1 <= int(m) <= 12]
                    if len(real) != len(months):
                        if real:
                            step.args["months"] = real
                        else:
                            step.args.pop("months")
                        trace.note(f"{step.id}: months={months} are not months; "
                                   f"{'kept ' + str(real) if real else 'dropped'}")

            # A percentage written into a decile field. "The bottom 50%" came
            # back as floor_area_decile_at_most=50, which matches all ten
            # deciles: the cohort was the whole portfolio and the comparison
            # built on it was wrong by the size of the dataset. Unlike
            # min_coverage=12, which might mean twelve months, a 50 in a field
            # that runs 1 to 10 has only one reading.
            for step in plan.steps:
                for name, value in list(step.args.items()):
                    if (name.endswith("_decile_at_least")
                            or name.endswith("_decile_at_most")):
                        if (isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and 10 < value <= 100 and value % 10 == 0):
                            step.args[name] = int(value // 10)
                            trace.note(f"{step.id}: {name}={value:g} read as a "
                                       f"percentage, so decile "
                                       f"{int(value // 10)} of ten")

            # A percentile of demand is a level in the demand distribution.
            # The share of consumption falling in the top 5% of hours is a
            # different quantity with the same "5" in it, and the model asked
            # for that: 0.0609 where the 95th percentile is 430.9 kW. Two
            # tools carry percentiles -- the statistics tool at p05 and p95,
            # and calculate_baseload at any level -- and a plan naming neither
            # has not been asked the question.
            percentile = resolve_percentile(question)
            if (percentile in (0.95, 0.05)
                    and re.search(r"\bdemand\b|\bconsumption\b", question.lower())):
                wanted = "p95" if percentile == 0.95 else "p05"
                computed = any(
                    (step.tool == "calculate_consumption_statistics"
                     and wanted in (step.args.get("metrics") or []))
                    or (step.tool == "calculate_baseload"
                        and step.args.get("method") == "percentile")
                    for step in plan.steps)
                if not computed:
                    for step in plan.steps:
                        if (step.tool == "calculate_demand_metrics"
                                and step.args.get("top_hours_share_pct") is not None):
                            step.tool = "calculate_consumption_statistics"
                            step.args = {k: v for k, v in step.args.items()
                                         if k in ("building_ids", "start", "end",
                                                  "months")}
                            step.args["metrics"] = [wanted]
                            plan.answer_step = step.id
                            trace.note(f"{step.id}: the question names the "
                                       f"{int(percentile * 100)}th percentile of "
                                       f"demand, which is a level, not the share "
                                       f"of consumption in the top hours")

            # A profile comparison is about size unless it asks about shape.
            # "Compare Mouse_health_Estela's Monday profile to its Sunday
            # profile" asks which day draws more, and the answer is the ratio
            # of the two means, 1.23. The model reached for measure=
            # "profile_shape" on that and on the weekday/weekend comparison,
            # answering 0.97 and 0.92 -- correlations, which say the rhythm
            # holds and nothing about the size the question asked after.
            if not asks_about_shape(question):
                for step in plan.steps:
                    if (step.tool == "compare_temporal_profiles"
                            and step.args.get("measure") == "profile_shape"):
                        step.args.pop("measure")
                        trace.note(f"{step.id}: the question compares two "
                                   f"profiles without asking whether the shape "
                                   f"moved, so it is answered by the ratio of "
                                   f"their means")

            # An index question is answered by the index. The hierarchy
            # wandered through three data-quality sub-goals and answered with
            # 1,418 rows of meter gaps, having built the ranking it was asked
            # for two steps earlier.
            index_components = resolve_index_components(question)
            if index_components:
                # The cohort an index needs is what its components can be
                # computed for, not one capability per phrase in the question.
                # Asked to prioritise on EUI, peer position, weather
                # sensitivity and anomalies, the plan required eui_analysis,
                # weather_analysis, peer_benchmark *and* forecasting -- the
                # last two screen on meter flatness and record length, which
                # no component uses -- and scored 217 offices where 262 can be
                # scored.
                needed = {"eui": "eui_analysis",
                          "peer_relative_eui": "eui_analysis",
                          "baseload_per_area": "eui_analysis",
                          "peak_power_density": "eui_analysis",
                          "weather_sensitivity": "weather_analysis"}
                wanted = {needed[c] for c in index_components if c in needed}
                for step in plan.steps:
                    if step.tool != "find_buildings":
                        continue
                    asked = set(step.args.get("requires") or [])
                    if wanted and asked - wanted:
                        step.args["requires"] = sorted(wanted)
                        trace.note(
                            f"{step.id}: dropped "
                            f"{', '.join(sorted(asked - wanted))} -- the index "
                            f"combines {', '.join(index_components)}, and no "
                            f"component reads those capabilities")
                # An index over an annual measure is an index for a year.
                period = (intent.time_range or {}) if intent is not None else {}
                asked_year = _calendar_year(period.get("start"), period.get("end"))
                if asked_year is None:
                    found_year = re.search(r"\b(20\d{2})\b", question or "")
                    asked_year = int(found_year.group(1)) if found_year else None
                if asked_year:
                    for step in plan.steps:
                        if (step.tool == "construct_performance_index"
                                and not step.args.get("year")):
                            step.args["year"] = asked_year
                built = [s for s in plan.steps
                         if s.tool == "construct_performance_index"]
                # Which end of an index is the answer is the question's to
                # set, and where it does not say, the convention does: worst
                # first, because an index of efficiency exists to say what to
                # look at. Left to the model, single mode read it one way and
                # multi the other, and the same question had two answers.
                index_order = ("asc" if re.search(
                    r"\bmost\s+efficient\b|\bbest\b|\blowest\b",
                    question.lower()) else "desc")
                for st in built:
                    if st.args.get("order") != index_order:
                        st.args["order"] = index_order
                        trace.note(f"{st.id}: the index runs "
                                   f"{'best' if index_order == 'asc' else 'worst'}"
                                   f" first, which is what the question asks for")
                if built and plan.answer_step != built[-1].id:
                    plan.answer_step = built[-1].id
                    trace.note(f"{built[-1].id}: the plan built the index and "
                               f"then reported past it; the index is the answer")
                elif not built and plan.steps:
                    # And where the plan never built one, build it. A plan
                    # that fetched 26 million rows of contextual data answered
                    # a question that asked for an index with the readings it
                    # would have been computed from.
                    cohort = next((st.id for st in reversed(plan.steps)
                                   if st.tool == "find_buildings"), None)
                    plan.add(PlanStep(
                        "index", "construct_performance_index",
                        {"building_ids": (f"${cohort}.column.building_id"
                                          if cohort else None),
                         "components": index_components,
                         "order": index_order,
                         "limit": (intent.entities or {}).get("limit", 20)},
                        depends_on=[cohort] if cohort else [],
                        purpose="the index the question asked for, which the "
                                "plan computed the inputs for and never built"))
                    plan.answer_step = "index"
                    trace.note("the question asks for an index over "
                               f"{index_components} and the plan never built "
                               f"one; added it")

            # Two cohorts measured separately are not a comparison. "Compare
            # the normalised baseload of buildings built before 1980 versus
            # those built after 2010" was planned as the measure over each
            # cohort and a combining step, which sets two tables side by side
            # and compares nothing: the answer then quoted one cohort's median
            # as though it were the result. compare_building_groups reports
            # both distributions and the ratio between their medians, which is
            # what "versus" asks for.
            if re.search(r"\bcompare[ds]?\b|\bversus\b|\bvs\.?\b"
                         r"|\bcompared\s+(?:to|with)\b", question.lower()):
                # A plan that already compares them answers with the
                # comparison. Multi-agent hierarchy ends on a reporting
                # sub-goal, and combine_evidence set the two cohorts side by
                # side in 534 rows with the comparison it had already computed
                # three steps earlier left out of the answer.
                comparisons = [s for s in plan.steps
                               if s.tool == "compare_building_groups"]
                if comparisons and plan.answer_step != comparisons[-1].id:
                    plan.answer_step = comparisons[-1].id
                    trace.note(f"{comparisons[-1].id}: the plan compared the "
                               f"cohorts and then reported past it; the "
                               f"comparison is the answer")
                # Any tool that measures a cohort, not only the two taking a
                # metric argument: a comparison answered from one side of it
                # is one of the two numbers, not the ratio.
                MEASURING = ("calculate_normalized_performance", "rank_buildings",
                             "calculate_demand_metrics",
                             "calculate_consumption_statistics")
                measured = [s for s in plan.steps if s.tool in MEASURING]
                # One cohort can be measured by several steps -- the model
                # called the demand tool four times for two groups -- so what
                # counts is how many distinct cohorts were measured, not how
                # many calls were made.
                cohorts: list = []
                for step in measured:
                    reference = step.args.get("building_ids")
                    key = json.dumps(reference, sort_keys=True, default=str)
                    if key not in [c[0] for c in cohorts]:
                        cohorts.append((key, step))
                measured = [step for _, step in cohorts]
                comparer = registry.REGISTRY.get("compare_building_groups")
                allowed = ((comparer.parameters or {}).get("properties", {})
                           .get("metric", {}).get("enum", []) if comparer else [])
                # The tools that take no metric argument state theirs in the
                # question instead, which is where the measure was named.
                metric = ((measured[0].args.get("metric") if measured else None)
                          or resolve_measure(question, default=""))
                if (len(measured) == 2 and metric in allowed
                        and measured[1].args.get("metric") in (metric, None)
                        and (measured[0].args.get("building_ids")
                             != measured[1].args.get("building_ids"))
                        and not any(s.tool == "compare_building_groups"
                                    for s in plan.steps)):
                    a, b = measured
                    named = resolve_vintages(question)
                    labels = ([v[0] for v in named] if len(named) == 2
                              else ["group_a", "group_b"])
                    # A step added after the repairs have run would keep the
                    # tool's defaults, so what the question states travels with
                    # the step that answers it.
                    stated = dict(a.args)
                    added_args = {"group_a": a.args.get("building_ids"),
                                  "group_b": b.args.get("building_ids"),
                                  "metric": metric,
                                  "label_a": labels[0], "label_b": labels[1]}
                    statistic = resolve_named_statistic(question)
                    if statistic in ("median", "mean", "total"):
                        added_args["statistic"] = statistic
                    asked_year = (stated.get("year")
                                  or _calendar_year(stated.get("start"),
                                                    stated.get("end")))
                    if asked_year:
                        added_args["year"] = int(asked_year)
                    plan.add(PlanStep(
                        f"{b.id}_compare", "compare_building_groups",
                        added_args,
                        depends_on=[d for d in (a.depends_on + b.depends_on)],
                        purpose="the two cohorts against each other, with the "
                                "ratio between their medians"))
                    plan.answer_step = f"{b.id}_compare"
                    trace.note(f"{a.id}/{b.id}: the plan measured each cohort "
                               f"but never compared them; added the comparison "
                               f"the question asked for")

            # A plan that ends in a fetch hands back a series where a number
            # was asked for, so a question wanting one gets the aggregation
            # appended, over the fetch's own period and cohort. Not when the
            # question asks for the readings themselves, and not when its
            # measure is one MEASURED_ELSEWHERE names -- a correlation is not
            # answered by a total.
            measures_something_else = (intent.metric or "") in MEASURED_ELSEWHERE
            if (intent.output_format in ("number", "entity", "text")
                    and plan.steps
                    and plan.steps[-1].tool.startswith("fetch_")
                    and not measures_something_else
                    and not any(s.tool == "calculate_consumption_statistics"
                                for s in plan.steps)):
                last = plan.steps[-1]
                source = last.id
                # Where the question names a calendar dimension, the
                # aggregation groups by it and takes the top row; otherwise it
                # is the plain total the fetch was standing in for.
                asked = question.lower()
                calendar = ("month" if re.search(r"\bwhich\s+month\b|\bwhat\s+month\b",
                                                 asked)
                            else "hour_of_day" if re.search(
                                r"\bwhich\s+hour\b|\bwhat\s+hour\b", asked)
                            else "day_of_week" if re.search(
                                r"\bwhich\s+day\s+of\s+(?:the\s+)?week\b", asked)
                            else None)
                grouping = ({"across": calendar, "metrics": ["total"],
                             "order_by": "total",
                             "order": "asc" if re.search(r"\blowest|\bleast\b",
                                                         asked) else "desc",
                             "limit": 1} if calendar else {})
                plan.add(PlanStep(
                    f"{source}_total", "calculate_consumption_statistics",
                    {"building_ids": last.args.get("building_ids"),
                     "start": last.args.get("start"),
                     "end": last.args.get("end"), **grouping},
                    depends_on=[source], resolution="compute",
                    purpose="the number the question asked for, over the "
                            "period that was fetched"))
                plan.answer_step = f"{source}_total"
            # A weather question that fetched both series and stopped gets
            # the relationship it asked for, rather than nothing. The fetch's
            # own cohort and period are reused, so the correlation covers
            # exactly what was fetched.
            if (measures_something_else and intent.metric == "weather_relationship"
                    and plan.steps
                    and plan.steps[-1].tool.startswith("fetch_")
                    and not any(s.tool in ("analyze_weather_relationship",
                                           "fit_temperature_response")
                                for s in plan.steps)):
                last = plan.steps[-1]
                cohort = (last.args.get("building_ids")
                          or (intent.entities or {}).get("building_ids"))
                if cohort:
                    plan.add(PlanStep(
                        f"{last.id}_weather", "analyze_weather_relationship",
                        {"building_ids": cohort, "variable": "air_temperature",
                         "method": "correlation", "grain": "daily",
                         "start": last.args.get("start"),
                         "end": last.args.get("end")},
                        depends_on=[last.id], resolution="compute",
                        purpose="the relationship the question asked for, over "
                                "the period that was fetched"))
                    plan.answer_step = f"{last.id}_weather"
                    trace.note(f"{last.id}: the plan fetched the two series and "
                               f"never related them; added the correlation")

            # The question asked for a chart, so draw one: told to end a
            # plotting plan with create_visualization, the model returns a
            # table of numbers instead and the chart tool is never reached.
            if (intent.output_format in ("figure", "map") and plan.steps
                    and not any(s.tool == "create_visualization"
                                for s in plan.steps)):
                source = plan.steps[-1].id
                # Where the step publishes the curves it compared, draw those:
                # a comparison's own frame is one row of ratios, which has no
                # hour axis to plot.
                curves = PUBLISHES_PROFILE.get(plan.steps[-1].tool)
                plan.add(PlanStep(
                    f"{source}_fig", "create_visualization",
                    {"handle": (f"${source}.evidence.{curves}" if curves
                                else f"${source}.handle"),
                     **({"x": "hour", "y": "mean_kw", "group": "period"}
                        if curves else {}),
                     # The chart kind follows what is being plotted, not how
                     # the question was worded: a profile is keyed by hour and
                     # has no time axis, so drawing it as a line asked pandas
                     # to parse "hour 0" as a date. The step that produced the
                     # frame says which it is, and the wording is only the
                     # fallback.
                     "kind": "map" if intent.output_format == "map" else
                             ("load_profile"
                              if curves
                              or plan.steps[-1].tool == "calculate_load_profile"
                              or "profile" in question.lower()
                              or "pattern" in question.lower()
                              or "load shape" in question.lower() else "line"),
                     # Without a series column the renderer joins the last hour
                     # of one building to the first hour of the next, drawing
                     # diagonals across the chart that are not in the data.
                     **({} if curves else {"group": "building_id"}),
                     "title": question[:70]},
                    depends_on=[source], resolution="compute",
                    purpose="render the chart the question asked for"))
                plan.answer_step = f"{source}_fig"

            # Once more, now that every repair that adds a step has run.
            self._apply_named_statistic(plan, question, trace)

            problems = plan.validate()
            trace.record_plan(plan, attempt, problems)
            turn.plan = plan
            if problems:
                trace.note(f"attempt {attempt}: invalid plan {problems[:3]}")
                if attempt > self.budget.max_replans + bonus:
                    break
                continue

            # Where a repair named the step that answers, the steps after it
            # only present it, so one of them failing must not throw away an
            # answer that was already computed.
            if (plan.answer_step and intent is not None
                    and intent.output_format not in ("figure", "map", "report")):
                answering = next((step for step in plan.steps
                                  if step.id == plan.answer_step), None)
                answering_agent = ((registry.REGISTRY.get(answering.tool).agent
                                    if answering and registry.REGISTRY.get(answering.tool)
                                    else "") if answering else "")
                if answering is not None and answering_agent != "reporting":
                    # Every reporting step, wherever it sits in the list: a
                    # repair appends the step that answers, so the report it
                    # was going to write is upstream of it in the list and
                    # downstream of it in meaning.
                    trailing = [step for step in plan.steps
                                if step is not answering
                                and (registry.REGISTRY.get(step.tool).agent
                                     if registry.REGISTRY.get(step.tool) else "")
                                == "reporting"]
                    if trailing:
                        keep = [step for step in plan.steps if step not in trailing]
                        plan.steps[:] = keep
                        trace.note(f"dropped {len(trailing)} reporting step(s) "
                                   f"after {plan.answer_step}: the answer is "
                                   f"already there")

            results, halt = orchestrator.run(plan, self.state, trace,
                                             self.budget, started, attempt)
            verification = verifier.verify(plan, results, self.state, intent)
            trace.verification = verification
            if halt:
                trace.note(f"attempt {attempt} halted: {halt}")
            # A plan that stopped before its answer step was never tested
            # against the question. What it stopped on is one step refusing
            # what an earlier step handed it -- a resampler given a quality
            # profile, which has no timestamp column -- and that says nothing
            # about whether the snapshot can answer. Reporting it as
            # INSUFFICIENT_DATA presents a fault in the plan as a property of
            # the data, and the question went unanswered while the rule-based
            # floor could answer it. A budget halt is different: replanning
            # costs the budget that ran out.
            answer_id = plan.answer_step or (plan.steps[-1].id if plan.steps else None)
            answered = results.get(answer_id) if answer_id else None
            unfinished = (bool(halt) and "budget" not in halt
                          and (answered is None or not answered.ok))
            if ((verification["status"] != "reject" and not unfinished)
                    or attempt > self.budget.max_replans + bonus):
                break
            # A plan rejected on one argument was never tested, so the
            # approach may be sound and one value wrong. Two mistakes are
            # recoverable because the error already carries the answer: a
            # value the tool rejects, and a reference to a column a previous
            # step does not return. Quote the complaint back and let the model
            # correct it; fall back to rules only if that does not take.
            # Rejections are marked where they happen, not recognised here by
            # their wording.
            argument_error = next(
                (r.message for r in results.values()
                 if r.status is Status.ERROR and r.message
                 and (r.evidence or {}).get("arguments_rejected")), "")
            refusal = ""
            if unfinished and verification["status"] != "reject":
                # The refusing step's own words, for the same reason an
                # argument error is quoted back: the message names what was
                # missing from what it was given.
                refusal = next((r.message for r in results.values()
                                if r.status is Status.INSUFFICIENT_DATA
                                and r.message), halt)
            if refusal and not correction:
                correction = refusal
                bonus = 1
                trace.note(f"attempt {attempt} stopped short of an answer "
                           f"({refusal[:90]}); correcting the model")
            elif argument_error and not correction:
                correction = argument_error
                bonus = 1
                trace.note(f"attempt {attempt} failed on an argument "
                           f"({argument_error[:90]}); correcting the model")
            else:
                trace.note(f"attempt {attempt} rejected by verifier: "
                           f"{verification['failed']}; replanning once")
                # Strict keeps the model in the loop: the next attempt is
                # another model plan, not keyword routing.
                force_deterministic = not self.require_llm

        curation = self._curate(ctx, plan, results, trace) if self.write_back else None
        text, status = self._compose(question, plan, results, verification,
                                     intent, halted=halt or "",
                                     store=store)
        payload = _answer_payload(spec, plan, results, text, status,
                                  intent, store=store).to_dict()
        text = _lead_with_the_answer(text, payload, intent)
        # A premise the data contradicted belongs in the answer, not only in
        # the trace. "Moose_education_Lori at the Peacock site" is answered
        # for the building and the site clause is dropped -- which is the
        # right choice and useless to a reader who never learns it was made.
        for finding in premises.findings:
            if finding.severity == "warning":
                text += (f"\n  [premise: {finding.message}"
                         + (f" -- {finding.suggestion}" if finding.suggestion
                            else "") + "]")
        turn.answer, turn.verification = text, verification
        trace.answer, trace.status = text, status
        trace.answer_payload = payload
        if curation:
            trace.note(f"curated {curation['asserted_facts']} fact(s) and "
                       f"{curation['asserted_relations']} relation(s)"
                       + (" (knowledge store held by another process; nothing "
                          "written back)" if curation.get("read_only") else ""))
        path = trace.write(self.runs_root)
        self.state.note("turn_complete", run_id=trace.run_id, status=status)

        return Answer(
            question=question, text=text, status=status, run_id=trace.run_id,
            mode=self.mode, plan=plan.to_dict(),
            evidence=self.state.evidence_ledger(),
            results={k: v.to_payload() for k, v in results.items()},
            verification=verification, metrics=trace.metrics(),
            payload=payload, trace_path=str(path))

    def _curate(self, ctx: ToolContext, plan: Plan,
                results: dict[str, ToolResult], trace: Trace) -> dict:
        """Assert what this run computed, through the Curator.

        Runs after the answer is formed, never before: a finding is only worth
        keeping once the analysis that produced it has completed. The Curator
        itself does not reason -- it validates, pins and supersedes.
        """
        curator = KnowledgeCurator(ctx.graph, SNAPSHOT_VERSION, run_id=trace.run_id)
        facts_total = relations_total = 0
        for step in plan.steps:
            result = results.get(step.id)
            if result is None or not result.ok:
                continue
            args = result.provenance.get("_args", step.args)
            facts, relations = extract(step.tool, args, result, ctx.store)
            if facts:
                facts_total += curator.assert_facts(facts).asserted
            if relations:
                relations_total += curator.assert_relations(relations).asserted
        return {"asserted_facts": facts_total,
                "asserted_relations": relations_total,
                # Say so when the cache could not be written, rather than
                # reporting a curation that did not happen. Another process
                # holding the writer is ordinary, not an error.
                "read_only": bool(getattr(curator.graph, "read_only", False))}

    # -- helpers -----------------------------------------------------------
    def _consult_planner_for_the_record(self, intent, trace) -> None:
        """Ask the model to plan a question a gate is about to refuse.

        The capability and scope gates fire before any planner runs, so their
        rows carry no tokens and no planner name -- indistinguishable, in a
        results table, from a question the model failed on. Under require_llm
        the model is consulted anyway and its tokens recorded; the refusal
        still stands, because a gate that the snapshot's own limits justify is
        not something a plan can talk its way out of. Best effort only: this
        must never turn a clean refusal into an error.
        """
        if not self.require_llm:
            return
        try:
            self._make_plan(intent, trace, attempt=1, known=None)
        except Exception as exc:                            # noqa: BLE001
            trace.note(f"model consulted before the gate refused: {exc}"[:160])

    def _make_plan(self, intent, trace: Trace, attempt: int, known=None, spec=None,
                   force_deterministic: bool = False,
                   correction: str = "") -> tuple[Plan, dict]:
        if (self.planning_engine == "structured" and spec is not None
                and not self.require_llm):
            if spec.task == "quality" and "max_missing_fraction" in spec.parameters:
                args = {
                    "building_ids": spec.entities.get("building_ids"),
                    "site_id": spec.entities.get("site_id"),
                    "start": spec.period.get("start"),
                    "end": spec.period.get("end"),
                    "max_missing_fraction": spec.parameters["max_missing_fraction"],
                }
                plan = Plan(question=intent.question, answer_step="s1")
                plan.add(PlanStep("s1", "profile_data_quality", args,
                                  resolution="compute",
                                  purpose="profile quality in the requested period"))
                trace.planner_used = "structured-deterministic"
                return plan.fill_agents(), {"planner": "structured-deterministic"}
            try:
                compile_analysis(spec)
            except ValueError:
                pass
            else:
                plan = Plan(question=intent.question, answer_step="s1")
                if spec.output.get("kind") == "scalar":
                    plan.answer_field = "value"
                plan.add(PlanStep(
                    "s1", "run_structured_query", {"spec": spec.to_dict()},
                    resolution=spec.resolution,
                    purpose="execute the validated allow-listed analysis",
                ))
                trace.planner_used = "structured-compiler"
                trace.note("AnalysisSpec compiled through the safe query tool")
                return plan.fill_agents(), {"planner": "structured-compiler"}
        planner = self.deterministic if force_deterministic else self.planner
        # A known shape goes to its route. Only a *named* route counts: the
        # generic branch of the rule-based planner is a fallback, not a
        # verified answer to this question's shape, and handing it questions
        # the model plans better would trade accuracy for determinism.
        # require_llm does not veto this branch. prefer_rules is a
        # configuration the caller asked for by name -- it is the `hybrid` arm
        # of the configuration comparison, and silently disabling it would make
        # that arm a copy of `single-kg` and the comparison meaningless. The
        # two flags close different holes: require_llm forbids *substitution*
        # when the model fails, prefer_rules chooses a verified route up front.
        if (self.prefer_rules and not force_deterministic and attempt == 1
                and not correction):
            candidate = self.deterministic.plan(intent, known)
            named = candidate.route and not candidate.route.startswith("tag:") \
                and candidate.route != "unrecognised"
            if named:
                trace.planner_used = "rules"
                trace.note(f"rule-based route {candidate.route!r} matched this "
                           f"question's shape; no model plan was requested")
                return candidate, {"planner": "rules", "route": candidate.route}
        if isinstance(planner, (LLMPlanner, HierarchicalPlanner)):
            grounding = self.intent_agent.vocab.grounding()
            if correction:
                grounding += (
                    "\n\nThe previous plan for this question failed on an "
                    "argument, not on its shape. Keep the approach and fix the "
                    "argument:\n- " + correction)
            try:
                plan, meta = planner.plan(
                    intent, mode=self.mode, grounding=grounding,
                    known_facts=known)
            except PlannerRefused as refused:
                # The call happened and the tokens were spent; record them
                # before the refusal leaves, or the row reads as though the
                # model was never asked.
                spent = refused.meta or {}
                if spent.get("prompt_tokens") or spent.get("completion_tokens"):
                    trace.record_llm(spent.get("provider", "?"),
                                     spent.get("model", "?"),
                                     purpose="planning (refused)",
                                     prompt_tokens=spent.get("prompt_tokens", 0),
                                     completion_tokens=spent.get("completion_tokens", 0),
                                     duration_s=spent.get("duration_s", 0.0))
                trace.planner_used = "llm-refused"
                raise
            for call in getattr(planner, "calls", None) or []:
                trace.record_llm(call["provider"], call["model"],
                                 purpose=call["purpose"],
                                 prompt_tokens=call["prompt_tokens"],
                                 completion_tokens=call["completion_tokens"],
                                 duration_s=call["duration_s"])
            if not getattr(planner, "calls", None) and (
                    meta.get("prompt_tokens") or meta.get("completion_tokens")):
                trace.record_llm(meta.get("provider", "?"), meta.get("model", "?"),
                                 purpose="planning",
                                 prompt_tokens=meta.get("prompt_tokens", 0),
                                 completion_tokens=meta.get("completion_tokens", 0),
                                 duration_s=meta.get("duration_s", 0.0))
            if meta.get("planning_calls"):
                trace.note(f"hierarchical planning: {meta['planning_calls']} LLM "
                           f"call(s) across {meta.get('stages', 1)} stage(s)")
            if meta.get("fell_back"):
                trace.note(f"attempt {attempt}: LLM plan unusable "
                           f"({meta.get('reason')}); used deterministic plan")
            trace.planner_used = (
                "deterministic" if meta.get("fell_back")
                else "hierarchical" if isinstance(planner, HierarchicalPlanner)
                else "llm")
            return plan, meta
        trace.planner_used = "deterministic"
        if self.planning_engine == "structured":
            trace.planner_used = "structured-deterministic"
            reason = "validated AnalysisSpec selected the deterministic planner"
        else:
            reason = ("the verifier rejected the model plan for this question"
                      if force_deterministic
                      else "no model planner is configured for this agent")
        trace.note("planned deterministically: " + reason)
        return (planner.plan(intent, known_facts=known),
                {"planner": "deterministic"})

    @staticmethod
    def _compose(question: str, plan: Plan, results: dict[str, ToolResult],
                 verification: dict, intent=None, halted: str = "",
                 store=None) -> tuple[str, str]:
        """Turn the final result into a factual answer, or an honest refusal."""
        if not results:
            if halted and "budget exhausted" in halted:
                return (f"The run used its whole budget before any step "
                        f"executed ({halted}). The question is not the problem; "
                        f"raise the budget and ask again.", "ERROR")
            return ("No analysis could be run for this question.", "ERROR")
        executed = [s for s in plan.steps if s.id in results]
        final = results[executed[-1].id]

        # Two scores and a question asking which is lower. The plan forecast
        # with each model the question named and scored each, which is exactly
        # what was asked -- and the reply carried one of the two numbers and
        # named no model, because composition takes the last step and there is
        # no last step for a comparison that lives across two. Comparing them
        # is the whole of the remaining work.
        scored = [(s.id, results[s.id]) for s in executed
                  if s.tool == "evaluate_forecast" and results[s.id].ok
                  and isinstance(results[s.id].data, dict)]
        if len(scored) > 1 and re.search(
                r"\bwhich\s+(?:model|method|approach)\b|\blower\b|\bhigher\b"
                r"|\bbetter\b|\bworse\b|\bmore\s+accurate\b|\bbest\b",
                question or "", re.I):
            measure = next((m for m in ("mae", "rmse", "mape", "mase")
                            if all(m in r.data for _, r in scored)), None)
            if measure:
                ranked = sorted(
                    ((_model_name((r.evidence or {}).get("model") or sid),
                      float(r.data[measure])) for sid, r in scored),
                    key=lambda pair: pair[1])
                winner, best = ranked[0]
                spread = ", ".join(f"{name} {value:,.3f}" for name, value in ranked)
                lines = [f"{winner} has the lower {measure.upper()}: {spread} "
                         f"({measure.upper()} in kWh; lower is better)."]
                for sid, result in scored:
                    name = _model_name((result.evidence or {}).get("model") or sid)
                    rest = ", ".join(f"{k} {v:,.3f}" for k, v in result.data.items()
                                     if k != measure and isinstance(v, (int, float)))
                    if rest:
                        lines.append(f"  {name}: {rest}")
                return "\n".join(lines), "OK"

        # Same rule as the payload: a report does not answer a counting or
        # listing question.
        wants_document = intent is not None and intent.output_format in (
            "figure", "map", "report")
        # ...and a document that refused is not one. The chart step the runtime
        # appends is an extra; when it cannot draw the frame it was handed, the
        # analysis underneath it still ran, and reporting "cannot answer"
        # throws away a result that is sitting right there.
        if wants_document and not final.ok:
            wants_document = False
        # Some tools describe the data rather than answer about it. A plan that
        # ends in one answers "is there data for this" when the question asked
        # what the data says -- an EUI question returning coverage fractions,
        # a ranking question returning a drawing. The agent alone does not
        # separate them: get_data_availability sits in data_discovery, which
        # legitimately answers plenty of other questions.
        diagnostic = (final.provenance.get("tool") in NON_ANSWERING_TOOLS
                      and not (intent is not None
                               and intent.task_category in ("1", "2")))
        answered_here = final.provenance.get("tool") in ANSWERING_REPORTS
        # Same rule as the payload: the chart is drawn *from* the answer, and
        # the answer is what a reader needs to see beside it.
        drew_chart = final.provenance.get("tool") == "create_visualization"
        if drew_chart:
            wants_document = False
        if (not wants_document and not answered_here
                and (final.provenance.get("agent") == "reporting" or diagnostic)):
            for step in reversed(executed[:-1]):
                candidate = results[step.id]
                # ...and the walk-back stops at one of these too, rather than
                # stepping over the ranking on its way to the raw evidence.
                if (candidate.provenance.get("tool") in ANSWERING_REPORTS
                        and candidate.ok and (candidate.rows or candidate.data)):
                    final = candidate
                    break
                if (candidate.provenance.get("agent") != "reporting"
                        and candidate.provenance.get("tool") not in NON_ANSWERING_TOOLS
                        and candidate.ok and (candidate.rows or candidate.data)):
                    final = candidate
                    break

        if final.status is Status.INSUFFICIENT_DATA:
            return (f"Cannot answer: {final.message}. "
                    f"Failing predicate: {final.predicate}.", "INSUFFICIENT_DATA")
        refusal = next((r for r in results.values()
                        if r.status is Status.INSUFFICIENT_DATA), None)
        if refusal is not None and not final.ok:
            return (f"Cannot answer: {refusal.message}. "
                    f"Failing predicate: {refusal.predicate}.", "INSUFFICIENT_DATA")
        if final.status is Status.EMPTY:
            # "How many lodging buildings are in Ireland?" is answered by zero.
            # Returning "no results matched" makes an answerable question look
            # unanswerable.
            #
            # But an empty final step means zero only when nothing upstream
            # broke. After an error the emptiness is a consequence of that
            # failure, not a fact about the data, and reporting it as "0"
            # states a finding the run never made.
            broke = [f"{k}: {v.message}" for k, v in results.items()
                     if v.status is Status.ERROR and v.message]
            if broke:
                return ("The analysis failed upstream, so nothing reached the "
                        f"final step: {'; '.join(broke)[:300]}.", "ERROR")
            if intent is not None and intent.output_format == "number":
                return ("0 — nothing matched the question's filters "
                        f"({final.message}).", "OK")
            if intent is not None and intent.output_format == "boolean":
                return (f"No — nothing matched ({final.message}).", "OK")
            return (f"No results matched: {final.message}.", "EMPTY")
        if final.status is Status.ERROR:
            # A drawing or a report presents an answer rather than being one,
            # so a failure there leaves the computed result standing. The
            # walk-back above keys on provenance, which a step rejected before
            # it ran never gets, so those arrive here instead.
            presenting = (executed and executed[-1].tool in PRESENTATION_TOOLS
                          and results.get(executed[-1].id) is final)
            if presenting:
                for step in reversed(executed[:-1]):
                    candidate = results.get(step.id)
                    if (candidate is not None and candidate.ok
                            and (candidate.rows or candidate.data)
                            and step.tool not in PRESENTATION_TOOLS):
                        trace.note(f"{executed[-1].id}: {executed[-1].tool} "
                                   f"failed; answering from {step.id} "
                                   f"({step.tool}), which computed the result")
                        final = candidate
                        break
            if final.status is Status.ERROR:
                return (f"The analysis failed: {final.message}.", "ERROR")

        # An overview question is answered by several independent catalog
        # lookups -- datasets, then countries, then cities, then types. Showing
        # only the last one computed the other three and threw them away
        #. Independent here means: nothing later consumed its output.
        consumed = {d for step in plan.steps for d in step.depends_on}
        siblings = [results[s.id] for s in executed[:-1]
                    if s.id not in consumed
                    and results[s.id].status is Status.OK
                    and results[s.id].preview
                    and registry.get(s.tool).agent == "data_discovery"]
        if siblings and registry.get(executed[-1].tool).agent == "data_discovery":
            parts = []
            for result in siblings + [final]:
                parts.append(_render_frame(result))
            return ("\n\n".join(parts), "OK")

        # A yes/no question is answered in words first, with the number that
        # decided it. Everything below stays as supporting detail.
        lines = []
        if intent is not None and intent.output_format == "boolean":
            verdict = _verdict_from(
                final, question, (intent.entities or {}).get("building_ids"),
                store=store)
            if verdict is not None:
                answer, basis = verdict
                detail = ", ".join(f"{k.replace('_', ' ')}: {v:,}"
                                   if isinstance(v, (int, float)) else f"{k}: {v}"
                                   for k, v in basis.items())
                lines.append(f"{'Yes' if answer else 'No'} — {detail}.")
        # A tool that can say what its result *means* says it here, and the
        # table becomes support rather than the answer. Forecasts, anomaly
        # scans and clusterings all produced columns of unlabelled numbers --
        # a reader could not tell demand from energy, or a fitted model from a
        # lookup, without reading the tool's source. The narrative is
        # composed by the tool because only the tool knows what it did.
        narrative = (final.evidence or {}).get("narrative")
        if isinstance(narrative, list) and narrative:
            lines.extend(_narrative_lines(narrative))
            lines.append("")
        lines.append(_heading(final))
        if isinstance(final.data, dict):
            for key, value in list(final.data.items())[:12]:
                # A tool that wrote a narrative has already said this, in
                # lines, above the heading. Printing its raw string again as
                # one key-value pair says the same thing twice and worse.
                if key == "narrative" and narrative:
                    continue
                if not isinstance(value, (list, dict)):
                    lines.append(f"  {key.replace('_', ' ')}: {_cell(value, key)}")
        elif final.preview:
            all_columns = list(final.preview[0])
            columns = _text_columns(final, all_columns)
            # The preview carries the whole set when it is short enough to
            # recite, and five rows when it is not.
            lines.extend(_aligned(columns, final.preview[:RECITABLE_ROWS]))
            if len(columns) < len(all_columns):
                dropped = [c for c in all_columns if c not in columns]
                lines.append(f"  Also in the full table: {', '.join(dropped)}.")
            if final.rows and final.rows > len(final.preview):
                lines.append(f"  and {final.rows - len(final.preview):,} more rows.")
        # A filtered count states its population, or it reads as a census.
        ev = final.evidence
        if ev.get("matched_filter") and ev.get("buildings_in_catalog"):
            # Whichever floor shaped the set, named. Reporting only
            # min_coverage said "no coverage floor" under an answer that a
            # 365-complete-day floor had cut from 1,636 to 269 -- a reader
            # would take that count for the whole portfolio.
            floor = ev.get("min_coverage")
            floors = []
            if floor is not None:
                floors.append(f"coverage floor {floor:g}"
                              + (f" in {ev['coverage_year']}"
                                 if ev.get("coverage_year") else ""))
            if ev.get("min_complete_days_in_year"):
                floors.append(f"{ev['min_complete_days_in_year']} complete days"
                              " within one calendar year")
            if ev.get("min_varying_days_in_year"):
                floors.append(f"{ev['min_varying_days_in_year']} of them"
                              " not flat")
            if ev.get("min_months_of_data"):
                floors.append(f"{ev['min_months_of_data']} months of record")
            if ev.get("gap_at_most_hours") is not None:
                floors.append("no gap over "
                              f"{ev['gap_at_most_hours']:g} hours")
            lines.append(
                f"  [{ev['matched_filter']:,} building(s) matched"
                + (f", of {ev['buildings_metered']:,} metered "
                   f"and {ev['buildings_in_catalog']:,} in the catalog"
                   if ev.get("buildings_metered") else "")
                + ("; " + ", ".join(floors) if floors
                   else "; no completeness floor") + "]")
        # Say which requirements shaped the cohort. The filter was applied --
        # 296 office buildings really do have weather -- but nothing in the
        # answer said weather had been required, so a reader could not tell
        # whether the question had been honoured.
        required = ev.get("requires") or []
        if required:
            readable = ", ".join(r.replace("_", " ") for r in required)
            lines.append(f"  [filtered to buildings with: {readable}]")
        # What an imputation did is the answer to "fill these gaps": how many
        # hours were invented, and how well the method does where the truth is
        # known. Leaving it in evidence put a filled series in front of a
        # reader with nothing saying how much of it was filled.
        if ev.get("imputed_rows_flagged") is not None:
            filled = ev.get("filled", 0)
            line = (f"  [{filled:,} value(s) imputed and flagged in the "
                    f"'imputed' column; {ev.get('missing_after', 0):,} left "
                    "empty")
            if ev.get("holdout_mape_pct") is not None:
                line += (f"; hold-out error {ev['holdout_mape_pct']:g}% against "
                         f"{ev.get('holdout_mape_hour_of_day_baseline_pct')}% "
                         "for an hour-of-day average")
            lines.append(line + "]")
        # A composed measure states what went into it. An index is only as
        # good as its components and their weights, and both are choices the
        # question does not make: an answer that shows the ranking without
        # them cannot be checked or argued with.
        # The settings the answer was computed under, on one line.
        settings = [f"{key.replace('_', ' ')} {_cell(final.evidence[key], key)}"
                    for key in ("metric", "unit", "year", "min_coverage",
                                "method", "timezone", "outliers_excluded",
                                "components", "weights", "weighting")
                    if final.evidence.get(key) is not None]
        if settings:
            lines.append("  Computed with " + ", ".join(settings) + ".")
        if drew_chart:
            drawn = next((r for r in reversed(list(results.values()))
                          if isinstance(r.data, dict) and r.data.get("figure_path")),
                         None)
            if drawn is not None:
                lines.append(f"  A {drawn.data.get('kind', 'chart')} of these "
                             f"{drawn.evidence.get('rows_plotted', 0):,} rows "
                             f"is shown with this answer.")
        if verification.get("status") == "warn":
            # The checks are named as assertions -- "the answer is about what
            # was asked" -- so listing them under a heading of "warnings" reads
            # as though they held. They are the ones that did not.
            # ...and what about it. The names alone read as an accusation a
            # reader cannot act on: "plan strategies match the engines used"
            # says nothing about which step, which strategy or which engine.
            detail = {c["check"]: c.get("detail") for c in
                      verification.get("checks", []) if not c.get("passed")}
            lines.append("  [could not confirm: " + "; ".join(
                f"{name}{f' ({detail[name]})' if detail.get(name) else ''}"
                for name in verification["failed"]) + "]")
        return "\n".join(lines), final.status.value
