"""Shared helpers for tool implementations."""
from __future__ import annotations

import pandas as pd

from .base import UnknownBuildings
from .registry import ToolContext

AGG_VIEW = {"daily": "agg_daily", "monthly": "agg_monthly", "yearly": "agg_yearly"}

#: The structural half of a capability gate: what a metric needs to be
#: *computable*, as opposed to what it needs to be *trustworthy*. The snapshot
#: bundles both -- can_eui is "has data, has a floor area, and has enough
#: record" -- and the second half is a judgement the analysis already makes,
#: with min_coverage, against the window actually asked about.
#:
#: Applied twice the two halves disagree. Asked to compare 2017 office and
#: education EUI, a plan that selects the cohort with can_eui and then compares
#: it has dropped buildings for a coverage verdict computed over the whole
#: record, before the comparison gets to judge coverage over 2017; the cohort
#: came out 262 -> 229 and the median moved 121.8 -> 111.7. Selection asks
#: whether the number exists; the analysis decides whether to trust it.
STRUCTURAL = {
    "can_eui": "floor_area_m2 IS NOT NULL AND floor_area_m2 > 0",
    "can_occupancy_normalisation": "occupants IS NOT NULL AND occupants > 0",
}





def in_clause(column: str, values: list[str] | None, params: list) -> str:
    """Restrict a column to a set of values. None means no restriction.

    An *empty list* is not the same as None and must not be treated as one. It
    reached here from a step whose upstream matched nothing, and returning no
    clause turned "analyse the buildings we just selected" into "analyse all
    1,578 of them" -- a wrong answer with no sign that anything had gone wrong.
    A caller that means every building omits the argument.
    """
    if values is None:
        return ""
    if not values:
        return " AND 1 = 0"
    params.append(list(values))
    return f" AND {column} IN (SELECT unnest(?))"


def load_series(
    ctx: ToolContext,
    handle: str | None = None,
    building_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """Resolve a working frame from a handle or straight from the snapshot.

    Analysis tools accept either, so a chain can pass handles along without a
    refetch, while a single-shot call still works without one.
    """
    if handle:
        df = ctx.store.get_or_none(handle)
        if df is None:
            raise KeyError(f"unknown handle: {handle}")
        if start is not None and "timestamp" in df:
            df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None and "timestamp" in df:
            df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]
        return df

    # A cohort of ids that name no building is a mistake in the plan, not an
    # empty period: "compare weekday with weekend for site Fox" passed
    # building_ids=["Fox"], the site's own id, and the run reported that no
    # measurements fell in the requested window. Say which ids do not exist.
    if building_ids:
        known = set(ctx.db.df(
            "SELECT building_id FROM buildings WHERE building_id IN "
            "(SELECT unnest(?))", [list(building_ids)])["building_id"])
        unknown = [b for b in building_ids if b not in known]
        if unknown and not known:
            raise UnknownBuildings(unknown)

    params: list = []
    clause = in_clause("building_id", building_ids, params)
    where = f"1=1 {clause}"
    if resolution == "hourly":
        if start:
            where += " AND timestamp_local >= ?"; params.append(start)
        if end:
            where += " AND timestamp_local < ?"; params.append(end)
        return ctx.db.df(f"""
            SELECT building_id, site_id, timestamp_local AS timestamp,
                   value, quality_flag
            FROM electricity WHERE {where}
            ORDER BY building_id, timestamp
        """, params or None)

    view = AGG_VIEW[resolution]
    if start:
        where += " AND period_start >= ?"; params.append(start)
    if end:
        where += " AND period_start < ?"; params.append(end)
    return ctx.db.df(f"""
        SELECT building_id, site_id, period_start AS timestamp,
               total_kwh, mean_demand, peak_demand, min_demand, median_demand,
               n_observations, expected_observations, coverage_fraction
        FROM {view} WHERE {where} ORDER BY building_id, timestamp
    """, params or None)


def with_local_parts(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Add hour / day-of-week / month / weekend columns from a local timestamp."""
    out = df.copy()
    ts = pd.to_datetime(out[column])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek           # Monday = 0
    out["month"] = ts.dt.month
    out["date"] = ts.dt.date
    out["is_weekend"] = out["dow"] >= 5
    return out


SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
           5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
           10: "autumn", 11: "autumn"}


def season_of(month: pd.Series) -> pd.Series:
    return month.map(SEASONS)


def budgeted(ctx, groups, minimum: int = 1):
    """Iterate groups, stopping when the run's deadline passes.

    The time budget was enforced only *between* plan steps, so one step could
    run unbounded: an anomaly-detection question over the portfolio ran for
    nine minutes holding the service's global lock. Every tool that walks a
    cohort now shares this iteration, so the bound applies wherever the work
    actually is.

    ``minimum`` guarantees at least some work is attempted even when the budget
    is already spent -- returning nothing at all is less useful than returning
    one building and saying so.

    Yields ``(key, frame)`` and records coverage on the returned counter.
    """
    covered = {"done": 0, "total": 0, "truncated": False}

    def walk():
        items = list(groups)
        covered["total"] = len(items)
        for index, (key, frame) in enumerate(items):
            if index >= minimum and ctx.out_of_time:
                covered["truncated"] = True
                return
            covered["done"] = index + 1
            yield key, frame

    return walk(), covered


def with_coverage(result, covered: dict):
    """Fold a budgeted walk's coverage into a result, marking it PARTIAL.

    A cohort that was cut short must say so where the number is read: a
    truncated answer that looks complete is the failure this whole guard exists
    to prevent.
    """
    from .base import Status
    if not covered.get("total"):
        return result
    result.evidence.setdefault("cohort_requested", covered["total"])
    result.evidence.setdefault("cohort_covered", covered["done"])
    if covered.get("truncated"):
        result.evidence["truncated_by_time"] = True
        result.status = Status.PARTIAL
        result.message = (f"time budget reached after {covered['done']} of "
                          f"{covered['total']}")
    return result


SERIES_BATCH = 150


def batched_series(ctx, building_ids, *, start=None, end=None,
                   resolution: str = "hourly", batch: int = SERIES_BATCH):
    """Load a cohort's series in batches, stopping at the run's deadline.

    ``budgeted`` bounds the *loop* over a cohort, but the load that feeds it ran
    first and whole: detect_anomalies pulled every hourly reading for 1,256
    buildings -- roughly 22 million rows -- before the deadline was consulted
    once. Batching makes the load interruptible too, so the bound covers the
    expensive half of the work.

    Yields ``(frame, covered)`` per batch; ``covered`` counts buildings loaded
    so far and records whether the walk was cut short.
    """
    ids = list(building_ids or [])
    covered = {"done": 0, "total": len(ids), "truncated": False}
    for offset in range(0, len(ids) or 1, batch):
        if offset and ctx.out_of_time:
            covered["truncated"] = True
            return
        chunk = ids[offset:offset + batch] if ids else None
        frame = load_series(ctx, building_ids=chunk, start=start, end=end,
                            resolution=resolution)
        covered["done"] = min(offset + batch, len(ids)) if ids else 0
        if not frame.empty:
            yield frame, covered
        if not ids:
            return
