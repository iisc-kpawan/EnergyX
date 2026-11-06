"""Tool contract: typed outcomes and the handle pattern.

Two rules hold for every one of the 48 runtime tools.

**Tools return outcomes; they do not raise.** An empty result from an eligibility
filter is a routine outcome, not a fault, and for some questions a correct
refusal *is* the right answer. Callers therefore branch on
``status`` rather than catching exceptions.

**Tools exchange handles, not frames.** A single fetch can be millions of rows;
serialising that into an LLM context is impossible and passing it between
processes is wasteful. A tool that produces tabular output writes it to the run
artifact store and returns a reference plus a small preview.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Status(str, Enum):
    OK = "OK"                                # result produced
    EMPTY = "EMPTY"                          # valid query, nothing matched
    PARTIAL = "PARTIAL"                      # some entities resolved, some did not
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # an eligibility predicate failed
    ERROR = "ERROR"                          # genuine fault


@dataclass
class ToolResult:
    """What every tool returns.

    ``data`` carries small inline results (a number, a short table). Anything
    large goes to the artifact store and is referenced by ``handle``.
    """

    status: Status
    data: Any = None
    handle: str | None = None
    rows: int | None = None
    columns: list[str] | None = None
    preview: list[dict] | None = None
    message: str | None = None
    predicate: str | None = None        # named failing predicate for INSUFFICIENT_DATA
    evidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (Status.OK, Status.PARTIAL)

    def to_payload(self) -> dict:
        """The compact dict an agent sees. Never contains bulk data."""
        payload: dict[str, Any] = {"status": self.status.value}
        for key in ("handle", "rows", "columns", "message", "predicate"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.data is not None:
            payload["data"] = self.data
        if self.preview:
            payload["preview"] = self.preview
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.provenance:
            payload["provenance"] = self.provenance
        return payload


PREVIEW_ROWS = 5
INLINE_ROW_LIMIT = 50  # frames at or below this ride inline; larger ones get a handle


def ok(data: Any = None, **kwargs) -> ToolResult:
    return ToolResult(Status.OK, data=data, **kwargs)


def empty(message: str, **kwargs) -> ToolResult:
    return ToolResult(Status.EMPTY, message=message, **kwargs)


def partial(data: Any = None, *, message: str, **kwargs) -> ToolResult:
    return ToolResult(Status.PARTIAL, data=data, message=message, **kwargs)


def insufficient(predicate: str, message: str, **kwargs) -> ToolResult:
    """Refuse honestly, naming the predicate that failed."""
    return ToolResult(Status.INSUFFICIENT_DATA, predicate=predicate,
                      message=message, **kwargs)


def error(message: str, **kwargs) -> ToolResult:
    return ToolResult(Status.ERROR, message=message, **kwargs)


def frame_result(df: pd.DataFrame, store, *, name: str, provenance: dict | None = None,
                 evidence: dict | None = None, force_handle: bool = False) -> ToolResult:
    """Package a DataFrame: inline when small, handle-backed when not."""
    if df is None or df.empty:
        return empty(f"{name}: no rows matched", provenance=provenance or {})
    handle = store.put(df, name=name)
    common = dict(handle=handle, rows=len(df),
                  columns=[str(c) for c in df.columns], preview=_preview(df),
                  evidence=evidence or {}, provenance=provenance or {})
    # Small frames additionally ride inline so an agent can read them without a
    # second call; the handle is always present so chaining never depends on size.
    if len(df) <= INLINE_ROW_LIMIT and not force_handle:
        return ToolResult(Status.OK, data=_records(df), **common)
    return ToolResult(Status.OK, **common)


def _records(df: pd.DataFrame) -> list[dict]:
    return _jsonify(df).to_dict(orient="records")


#: A set small enough to recite is the answer, not a sample of it. "Which
#: offices remain in the top decile after normalising" has nine members, and
#: a five-row preview answers five ninths of it.
RECITABLE_ROWS = 25


def _preview(df: pd.DataFrame) -> list[dict]:
    keep = len(df) if len(df) <= RECITABLE_ROWS else PREVIEW_ROWS
    return _records(df.head(keep))


def jsonify(df: pd.DataFrame) -> pd.DataFrame:
    """A frame every consumer can render: no timestamps, no NaN, no NumPy.

    Timestamps are rendered here and nowhere else, because two readers had
    disagreed about them. The snapshot stores instants and exposes local wall
    clock on the site's constant offset, so a stamp arrives tz-aware at UTC
    while *meaning* local time: printing it raw put "2017-07-25 00:00:00+00:00"
    under a column named timestamp_local, which reads as a timezone bug in an
    answer that had none. The offset carries no information here -- it is the
    same for every row of a site -- so the wall clock is printed and the
    timezone is stated once, in the evidence, where a reader can act on it.

    The seconds go too when every stamp lands on a minute, which is every
    hourly series in the snapshot: 24 rows of ":00:00" is noise in a column a
    reader is scanning for the hour.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        values = out[col]
        if getattr(values.dtype, "tz", None) is not None:
            values = values.dt.tz_localize(None)
        whole_minutes = bool(values.dt.second.eq(0).all()
                             and values.dt.microsecond.eq(0).all())
        midnight = whole_minutes and bool(values.dt.hour.eq(0).all()
                                          and values.dt.minute.eq(0).all())
        fmt = "%Y-%m-%d" if midnight else (
            "%Y-%m-%d %H:%M" if whole_minutes else "%Y-%m-%d %H:%M:%S")
        out[col] = values.dt.strftime(fmt)
    return out.astype(object).where(pd.notna(out), None)


# The private spelling stays: several modules import it, and the name change is
# not the point of this function.
_jsonify = jsonify


class UnknownBuildings(Exception):
    """Every id in the cohort names something that is not a building.

    Raised from the shared loader and turned into a refusal at the tool
    boundary: a plan that passes a site id where building ids belong gets told
    so, rather than an empty result that reads as "no data in that period".
    """

    def __init__(self, ids: list[str]) -> None:
        shown = ", ".join(map(str, ids[:5]))
        super().__init__(
            f"none of these name a building in this dataset: {shown}"
            + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
            + ". A site id or a city is not a building id; resolve the cohort "
              "with find_buildings first.")
        self.ids = ids
