"""Execution boundary for allow-listed structured analytical queries."""
from __future__ import annotations

from dataclasses import fields as _dataclass_fields

from ..agents.analysis_spec import AnalysisSpec
from ..agents.query_compiler import compile_analysis
from .base import ToolResult, frame_result
from .registry import ToolContext, tool


@tool("consumption")
def run_structured_query(ctx: ToolContext, spec: dict) -> ToolResult:
    """Execute a validated AnalysisSpec; raw SQL is never accepted."""
    # A model that writes a spec also writes commentary into it -- a
    # "description", a "reason", a "notes" -- and AnalysisSpec(**spec) turned
    # every one of those into a TypeError that lost the question. The extra
    # keys carry no meaning for the compiler, so they are dropped rather than
    # allowed to raise; what the spec actually says is unaffected.
    known = {f.name for f in _dataclass_fields(AnalysisSpec)}
    analysis = AnalysisSpec(**{k: v for k, v in (spec or {}).items()
                               if k in known})
    query = compile_analysis(analysis)
    frame = ctx.db.df(query.sql, query.params or None)
    return frame_result(
        frame, ctx.store, name="structured_analysis",
        evidence={"method": analysis.method, "unit": "kWh",
                  "parameters_bound": len(query.params)},
    )
