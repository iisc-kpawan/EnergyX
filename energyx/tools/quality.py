"""Data Quality & Preprocessing tools (7).

Quality work is answerable from the catalog wherever possible -- coverage,
missingness and gap statistics are precomputed per stream -- so profiling a
thousand buildings does not scan their measurements.

``ZERO_MASKED`` matters here: BDG2's cleaning replaced every zero with a null
, so "extended zero runs" and "missing data" are the same thing in the
cleaned series. The flag is what makes them distinguishable again.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ._common import budgeted, in_clause, load_series
from .base import ToolResult, empty, frame_result, insufficient, ok
from .registry import ToolContext, tool


@tool("data_quality")
def profile_data_quality(ctx: ToolContext, building_ids: list[str] | None = None,
                         site_id: str | None = None, start: str | None = None,
                         end: str | None = None,
                         max_missing_fraction: float | None = None,
                         source: Literal["electricity", "weather"] = "electricity"
                         ) -> ToolResult:
    """Coverage, missing and duplicate timestamps, masked zeros, gaps, status.

    ``source="weather"`` profiles the site weather series instead of the meters,
    so "are there duplicate timestamps in the weather dataset for site Lamb" is
    answerable.


    This answers "does this meter contain duplicated timestamps" and "are there
    duplicates in this stream": ``duplicate_count`` and ``dst_duplicate_hours``
    are reported per stream. get_data_availability describes what a stream
    covers; this one reports what is wrong with it.


    Returns the whole quality profile in one call rather than requiring five
    separate checks.
    """
    if source == "weather":
        params: list = []
        where = "1=1"
        if site_id:
            where += " AND w.site_id = ?"; params.append(site_id)
        df = ctx.db.df(f"""
            SELECT w.site_id, w.source_name, w.resolution_minutes,
                   w.n_observations, w.coverage_fraction,
                   -- the weather grid is built unique per (site, timestamp), so
                   -- any excess of rows over distinct timestamps is a duplicate
                   (SELECT COUNT(*) - COUNT(DISTINCT timestamp_utc)
                    FROM weather e WHERE e.site_id = w.site_id) AS duplicate_count,
                   w.air_temperature_completeness,
                   w.start_timestamp, w.end_timestamp
            FROM weather_sources w WHERE {where}
            ORDER BY w.coverage_fraction
        """, params or None)
        if df.empty:
            return empty("no weather source matched")
        return frame_result(df, ctx.store, name="quality_profile_weather",
                            evidence={"source": "weather",
                                      "sources": len(df),
                                      "duplicate_count":
                                          int(df["duplicate_count"].sum())})

    if start or end or max_missing_fraction is not None:
        params: list = []
        where = "1=1" + in_clause("building_id", building_ids, params)
        if site_id:
            where += " AND site_id = ?"; params.append(site_id)
        if start:
            where += " AND period_start >= ?"; params.append(start)
        if end:
            where += " AND period_start < ?"; params.append(end)
        having = ""
        if max_missing_fraction is not None:
            having = (" HAVING 1 - SUM(n_observations) / "
                      "NULLIF(SUM(expected_observations), 0) < ?")
            params.append(max_missing_fraction)
        df = ctx.db.df(f"""
            SELECT building_id, ANY_VALUE(site_id) AS site_id,
                   SUM(n_observations) / NULLIF(SUM(expected_observations), 0)
                       AS coverage_fraction,
                   1 - SUM(n_observations) / NULLIF(SUM(expected_observations), 0)
                       AS missing_fraction,
                   SUM(n_zero_masked) / NULLIF(SUM(expected_observations), 0)
                       AS zero_masked_fraction,
                   COUNT(DISTINCT period_start::DATE) AS observed_days,
                   -- Carried through the period branch too, so a yes/no
                   -- question about duplicates can be answered whichever
                   -- grain the caller asked for.
                   SUM(n_rows_present - n_observations) AS duplicate_count,
                   MIN(period_start) AS first_period,
                   MAX(period_start) AS last_period
            FROM agg_daily WHERE {where} GROUP BY building_id {having}
            ORDER BY missing_fraction
        """, params or None)
        if df.empty:
            return empty("no streams matched the requested period and threshold")
        return frame_result(
            df, ctx.store, name="quality_profile_period",
            evidence={"period": [start, end],
                      "max_missing_fraction": max_missing_fraction,
                      "streams": len(df)},
        )

    params: list = []
    where = "1=1" + in_clause("q.stream_id", None, params)
    clause = ""
    if building_ids:
        params.append(list(building_ids))
        clause = " AND st.building_id IN (SELECT unnest(?))"
    if site_id:
        clause += " AND st.site_id = ?"; params.append(site_id)
    df = ctx.db.df(f"""
        SELECT st.building_id, st.site_id, q.stream_id, q.coverage_fraction,
               q.missing_fraction, q.zero_masked_fraction, q.longest_gap_hours,
               q.duplicate_count, q.first_valid_timestamp, q.last_valid_timestamp,
               q.quality_status, q.dst_duplicate_hours, q.dst_missing_hours
        FROM data_quality q
        JOIN electricity_streams st ON st.stream_id = q.stream_id
        WHERE {where} {clause}
        ORDER BY q.coverage_fraction
    """, params or None)
    if df.empty:
        return empty("no streams matched")
    summary = {
        "streams": len(df),
        "mean_coverage": round(float(df["coverage_fraction"].mean()), 4),
        "median_coverage": round(float(df["coverage_fraction"].median()), 4),
        "status_counts": df["quality_status"].value_counts().to_dict(),
        "note": "zero_masked_fraction counts readings that were exactly 0 in the "
                "raw source and nulled by BDG2 cleaning",
    }
    return frame_result(df, ctx.store, name="quality_profile", evidence=summary)


@tool("data_quality")
def detect_data_issues(
    ctx: ToolContext,
    # The three checks below read the quality table, which holds one row per
    # stream, so the whole portfolio is an ordinary question here and the
    # planning rules tell a plan to ask it by omitting the cohort. Requiring
    # the cohort made that rule a trap. flatline is the exception -- it reads
    # the readings themselves -- and it says so rather than scanning silently.
    building_ids: list[str] | None = None,
    zero_basis: Literal["masked", "observed"] = "masked",
    checks: list[Literal["coverage", "gaps", "zero_masked", "zero_run",
                         "flatline"]] | None = None,
    gap_hours: int = 24,
    gap_bound: Literal["at_least", "exceeding"] = "at_least",
    max_coverage: float = 0.95,
    consecutive_zero_days_at_least: int | None = None,
    # A run asked for in hours is measured in hours. Whole calendar days
    # cannot express "72 consecutive hours": a run spanning five part-days is
    # longer than three days and contains none that are wholly dark.
    consecutive_zero_hours_at_least: int | None = None,
) -> ToolResult:
    """Flag streams by coverage, gaps, zero_masked or flatline.

    Naming no checks runs coverage, gaps and zero_masked. flatline is asked for
    by name because it reads every reading of every stream.

    ``gap_bound`` decides the edge. "Gaps exceeding one week" is strictly more
    than 168 hours and "at least a week" includes it, and three streams sit
    exactly on that line -- so the default of at_least answered 613 where the
    question asked for 610. A threshold question has an edge, and guessing
    which side it falls on is a three-building error nobody can see.

    zero_masked and zero_run are different questions about the same zeros, and
    picking the wrong one is not a near miss: zero_masked flags a stream whose
    zeros exceed 5% of the record wherever they fall and returns 282 streams,
    while zero_run flags a stretch of whole days where every hour was zero and
    returns 49 buildings. "More than 3 consecutive days" is zero_run with
    ``consecutive_zero_days_at_least=4`` -- the parameter name carries the
    edge. Ask for zero_run alone when the question is about a stretch: the
    checks are combined with or, so adding zero_masked beside it returns both
    sets and buries the answer in the larger one.

    The check names are the four above, named rather than described: a value
    the caller cannot see is a value it will invent.

    ``max_coverage`` is a fraction of one, not a percentage: 0.9 means 90%.
    """
    valid = {"coverage", "gaps", "zero_masked", "zero_run", "flatline"}
    # An empty list is a choice -- no stream checks -- and differs from naming
    # none at all. It matters now that a run check can stand on its own: asking
    # only for consecutive zero days with checks=[] ran all four instead, and
    # was refused for a flatline check the caller had not asked for.
    # Naming no checks runs the three that read the quality table. flatline is
    # left out of that default because it reads every reading of every stream:
    # it is worth asking for and not worth doing by accident.
    if checks is None:
        # Naming the threshold and no checks is asking for the run and nothing
        # else. Adding it to the three defaults instead returned 1,154
        # buildings for a question whose answer is 49, with the answer inside.
        chosen = ({"zero_run"} if consecutive_zero_days_at_least is not None
                  else valid - {"flatline", "zero_run"})
    else:
        chosen = set(checks)
        if consecutive_zero_days_at_least is not None:
            chosen.add("zero_run")
    unknown = chosen - valid
    if unknown:
        return insufficient("unknown_check",
                            f"unknown checks {sorted(unknown)}; valid: {sorted(valid)}")

    params: list = []
    clause = in_clause("st.building_id", building_ids, params)
    base = ctx.db.df(f"""
        SELECT st.building_id, q.coverage_fraction, q.zero_masked_fraction,
               q.longest_gap_hours
        FROM data_quality q JOIN electricity_streams st ON st.stream_id = q.stream_id
        WHERE 1=1 {clause}
    """, params or None)
    if base.empty:
        return empty("no streams matched")

    issues: list[dict] = []
    for _, row in base.iterrows():
        found = []
        if "coverage" in chosen and row["coverage_fraction"] < max_coverage:
            found.append(f"coverage {row['coverage_fraction']:.3f} < {max_coverage}")
        over = (row["longest_gap_hours"] > gap_hours if gap_bound == "exceeding"
                else row["longest_gap_hours"] >= gap_hours)
        if "gaps" in chosen and over:
            found.append(f"longest gap {row['longest_gap_hours']:.0f}h")
        if "zero_masked" in chosen and row["zero_masked_fraction"] > 0.05:
            found.append(f"zero-masked {row['zero_masked_fraction']:.1%}")
        if found:
            issues.append({"building_id": row["building_id"], "issues": "; ".join(found)})

    if not chosen:
        return insufficient(
            "no_checks",
            f"nothing to check: name at least one of {sorted(valid)}")

    if "zero_run" in chosen:
        run_days = (4 if consecutive_zero_days_at_least is None
                    else int(consecutive_zero_days_at_least))
        # Two different questions about zeros, and the snapshot separates them.
        # Ingest masks an exact zero to missing, so the cleaned series holds
        # none at all: "more than 72 consecutive hours of zero consumption,
        # excluding missing observations" is asking about the cleaned series
        # and its answer is nobody. Counting the masked days instead answers
        # "which buildings were dark before cleaning" and returns 71 -- a real
        # finding, and the opposite of what the question excluded.
        if zero_basis == "observed":
            zero_params: list = []
            zero_where = in_clause("building_id", building_ids, zero_params)
            # Cleaned readings only, like every other answer. value = 0
            # already implies one -- a masked row carries NULL -- so the flag
            # is written out to say so rather than leave it to be worked out.
            observed = ctx.db.df(f"""
                SELECT COUNT(*) AS zeros FROM electricity
                WHERE quality_flag = 'OK' AND value = 0{zero_where}
            """, zero_params or None)
            if int(observed["zeros"].iloc[0] or 0) == 0:
                return ok({"buildings": [], "n_buildings": 0}, evidence={
                    "check": "zero_run", "zero_basis": "observed",
                    "consecutive_zero_days_at_least": run_days,
                    "headline": {
                        "label": f"buildings with {run_days * 24}+ consecutive "
                                 f"hours of observed zero consumption",
                        "value": 0, "unit": "buildings"},
                    "note": "the cleaned series holds no exact-zero readings: "
                            "ingest masks them to missing, so excluding "
                            "missing observations leaves no zeros to run. Ask "
                            "with zero_basis='masked' for the stretches that "
                            "were zero before cleaning."})
        if consecutive_zero_hours_at_least:
            hours = int(consecutive_zero_hours_at_least)
            hour_params: list = []
            hour_where = in_clause("building_id", building_ids, hour_params)
            runs = ctx.db.df(f"""
                WITH z AS (
                    SELECT building_id, timestamp_utc FROM electricity
                    WHERE quality_flag = 'ZERO_MASKED'{hour_where}
                ), m AS (
                    SELECT building_id, timestamp_utc,
                           CASE WHEN timestamp_utc - LAG(timestamp_utc) OVER
                                (PARTITION BY building_id ORDER BY timestamp_utc)
                                = INTERVAL 1 HOUR THEN 0 ELSE 1 END AS brk
                    FROM z
                ), g AS (
                    SELECT building_id,
                           SUM(brk) OVER (PARTITION BY building_id
                                          ORDER BY timestamp_utc) AS run_id
                    FROM m
                ), runs AS (
                    SELECT building_id, run_id, COUNT(*) AS n
                    FROM g GROUP BY building_id, run_id
                )
                SELECT building_id, MAX(n) AS longest_zero_run_hours
                FROM runs GROUP BY building_id HAVING MAX(n) >= ?
                ORDER BY longest_zero_run_hours DESC, building_id
            """, (hour_params or []) + [hours])
            found = runs["building_id"].tolist()
            return frame_result(runs, ctx.store, name="zero_runs", evidence={
                "check": "zero_run", "zero_basis": "masked",
                "consecutive_zero_hours_at_least": hours,
                "matched": len(found),
                "headline": {
                    "label": f"buildings with a run of {hours} or more "
                             f"consecutive zero hours",
                    "value": len(found), "unit": "buildings"},
                "note": "a zero hour is one BDG2's cleaning masked; a run is "
                        "consecutive clock hours, so a missing hour ends it"})

        # Runs are found in the daily aggregate rather than the readings: a day
        # whose every observed hour was zero-masked is one dark day, and
        # consecutive such days are found by the difference of two row numbers,
        # which is constant within a run.
        run_params: list = []
        run_clause = in_clause("building_id", building_ids, run_params)
        run_params.append(run_days)
        runs = ctx.db.df(f"""
            WITH marked AS (
                SELECT building_id, period_start,
                       (n_zero_masked >= n_observations AND n_observations > 0)
                           AS all_zero
                FROM agg_daily WHERE grain = 'daily' {run_clause}
            ), grouped AS (
                SELECT building_id, all_zero,
                       ROW_NUMBER() OVER (PARTITION BY building_id
                                          ORDER BY period_start)
                     - ROW_NUMBER() OVER (PARTITION BY building_id, all_zero
                                          ORDER BY period_start) AS run_id
                FROM marked
            )
            SELECT building_id, MAX(n) AS longest_zero_run_days FROM (
                SELECT building_id, run_id, COUNT(*) AS n
                FROM grouped WHERE all_zero GROUP BY 1, 2)
            GROUP BY building_id HAVING MAX(n) >= ?
            ORDER BY longest_zero_run_days DESC, building_id
        """, run_params or None)
        # "41 consecutive days of zero readings" is not what this found. The
        # cleaned series holds no zero readings at all -- ingest masked them
        # to missing -- so these are days whose every observed hour was a
        # masked zero. Said the short way the answer claims a measurement
        # that is not in the data, and a reader cannot tell it apart from a
        # building that genuinely metered zero.
        for _, row in runs.iterrows():
            issues.append({"building_id": row["building_id"],
                           "issues": f"{int(row['longest_zero_run_days'])} "
                                     "consecutive days that were zero before "
                                     "cleaning (masked to missing)"})

    if "flatline" in chosen:
        if not building_ids:
            return insufficient(
                "building_ids",
                "the flatline check reads every reading of every stream, so it "
                "needs a named cohort; the coverage, gaps and zero_masked "
                "checks run over the whole portfolio without one")
        series = load_series(ctx, building_ids=building_ids)
        for building, grp in series.groupby("building_id"):
            values = grp["value"].dropna().to_numpy()
            if len(values) > 48:
                run = _longest_equal_run(values)
                if run >= 24:
                    issues.append({"building_id": building,
                                   "issues": f"flat for {run} consecutive readings"})
    if not issues:
        return ok({"issues": []}, evidence={"checked": len(base),
                                            "checks": sorted(chosen),
                                            "consecutive_zero_days_at_least":
                                                consecutive_zero_days_at_least})
    return frame_result(pd.DataFrame(issues), ctx.store, name="data_issues",
                        evidence={"checked": len(base), "flagged": len(issues),
                                  "checks": sorted(chosen),
                                  "consecutive_zero_days_at_least":
                                      consecutive_zero_days_at_least})


def _longest_equal_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    change = np.flatnonzero(np.diff(values) != 0)
    bounds = np.concatenate(([-1], change, [values.size - 1]))
    return int(np.diff(bounds).max())


@tool("data_quality")
def clean_timeseries(
    ctx: ToolContext,
    handle: str,
    drop_flags: list[Literal["OK", "ZERO_MASKED"]] | None = None,
    clip_quantile: float | None = None,
    drop_nonpositive: bool = False,
) -> ToolResult:
    """Apply explicit removal rules to a fetched series. Never silent."""
    df = ctx.store.get_or_none(handle)
    if df is None:
        return insufficient("unknown_handle", f"no artifact for handle {handle!r}")
    before = len(df)
    removed: dict[str, int] = {}
    if drop_flags and "quality_flag" in df:
        mask = df["quality_flag"].isin(drop_flags)
        removed["flagged"] = int(mask.sum())
        df = df[~mask]
    if drop_nonpositive and "value" in df:
        mask = df["value"] <= 0
        removed["nonpositive"] = int(mask.sum())
        df = df[~mask]
    if clip_quantile and "value" in df:
        hi = df["value"].quantile(clip_quantile)
        mask = df["value"] > hi
        removed[f"above_q{clip_quantile}"] = int(mask.sum())
        df = df.loc[~mask]
    return frame_result(df, ctx.store, name="cleaned", force_handle=True,
                        evidence={"rows_before": before, "rows_after": len(df),
                                  "removed": removed})


@tool("data_quality")
def resample_timeseries(
    ctx: ToolContext,
    handle: str,
    resolution: Literal["hourly", "daily", "weekly", "monthly"],
    how: Literal["sum", "mean", "max", "min"] = "sum",
) -> ToolResult:
    """Change temporal resolution with an explicit aggregation rule.

    A period with no readings comes back empty, not zero. Resampling sums an
    empty bin to 0.0, so three missing hours became three hours of measured
    zero consumption -- indistinguishable downstream from a building that was
    genuinely dark, and counted as real data by everything that averages,
    totals or benchmarks it. ``n_observations`` says how many readings each
    period was built from, and 0 there now means an empty value rather than a
    confident zero.
    """
    df = ctx.store.get_or_none(handle)
    if df is None:
        return insufficient("unknown_handle", f"no artifact for handle {handle!r}")
    if "timestamp" not in df or "value" not in df:
        return insufficient("shape", "frame needs timestamp and value columns")
    rule = {"hourly": "h", "daily": "D", "weekly": "W", "monthly": "MS"}[resolution]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = (df.set_index("timestamp")
                 .groupby("building_id")["value"]
                 .resample(rule).agg(how)
                 .reset_index())
    counts = (df.set_index("timestamp").groupby("building_id")["value"]
                .resample(rule).count().reset_index(name="n_observations"))
    out = grouped.merge(counts, on=["building_id", "timestamp"])
    empty_periods = out["n_observations"] == 0
    out.loc[empty_periods, "value"] = float("nan")
    return frame_result(out, ctx.store, name=f"resampled_{resolution}",
                        force_handle=True,
                        evidence={"resolution": resolution, "how": how,
                                  "rows": len(out),
                                  "periods_with_no_readings":
                                      int(empty_periods.sum())})


@tool("data_quality")
def impute_missing_values(
    ctx: ToolContext,
    handle: str,
    method: Literal["linear", "ffill", "seasonal_hour", "zero",
                    "weather_regression"] = "linear",
    max_gap_hours: int = 6,
    weather_variable: str = "air_temperature",
) -> ToolResult:
    """Fill gaps with a stated method, bounded by a maximum gap length.

    Longer gaps are deliberately left unfilled -- interpolating across a week is
    fabrication, not imputation.

    ``weather_regression`` is the weather-informed baseline: demand is fitted on
    the site's outdoor temperature together with hour-of-day and day-type, from
    the hours that are present, and the missing hours are predicted from it.
    Unlike the other methods it carries no ``max_gap_hours`` bound, because a
    model conditioned on the weather and the clock does not degrade with the
    length of the gap the way a straight line drawn across it does.

    Every row comes back with ``imputed`` saying whether its value was measured
    or produced here, because once the two are in one column nothing downstream
    can tell them apart, and an imputed hour that travels as a measurement is
    the one outcome worse than a gap.

    There is no ground truth for a missing hour, so the accuracy reported is a
    hold-out: a tenth of the hours that *are* present are hidden, imputed by the
    same method, and scored against what was really there. It is in the evidence
    as ``holdout_mape_pct``, beside the same score for the plainest alternative,
    so a reader can see whether the weather model earned its name.

    This runs on a frame that has a row for every period, where a gap is an
    empty value. A fetched series has no empty values at all -- its gaps are
    rows that are simply not there -- so imputing straight after a fetch finds
    nothing to fill and says so, which reads as "no gaps". Put the series on a
    regular grid first: fetch_electricity, then resample_timeseries, then this.
    """
    frame = ctx.store.get_or_none(handle)
    if frame is None:
        return insufficient("unknown_handle", f"no artifact for handle {handle!r}")
    df = frame.copy()
    if "value" not in df:
        return insufficient("shape", "frame needs a value column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    before = int(df["value"].isna().sum())
    # Gaps that are absent rows cannot be filled, because there is no row to
    # fill. Whether the frame also holds a few empty values is beside the
    # point, and missing it is the easy mistake here: this series
    # carries 2 empty values and 27 absent hours, so keying off the empty count
    # let it through, filled the 2, and reported nothing missing -- a clean
    # bill of health for a series still full of holes.
    ordered = df.sort_values(["building_id", "timestamp"])
    spacing = ordered.groupby("building_id")["timestamp"].diff().dropna()
    if not spacing.empty and spacing.nunique() > 1:
        step = spacing.mode().iloc[0]
        skipped = int(((spacing - step) / step).round().clip(lower=0).sum())
        if skipped:
            return insufficient(
                "irregular_grid",
                f"the timestamps skip {skipped} period(s) of {step}: those gaps "
                "are absent rows, not empty values, and nothing here can fill a "
                "row that is not there. Put the series on a regular grid with "
                "resample_timeseries first, then impute.")
    # Which rows were measured has to be settled before anything is filled in.
    df["imputed"] = df["value"].isna()
    filled = []
    holdout: list[tuple[float, float]] = []
    _walk, _cov = budgeted(ctx, df.groupby("building_id", sort=False))
    for building, grp in _walk:
        grp = grp.sort_values("timestamp").copy()
        if method == "weather_regression":
            grp, scored = _weather_regression_fill(ctx, building, grp,
                                                   weather_variable)
            holdout.extend(scored)
            filled.append(grp)
            continue
        if method == "linear":
            # `limit=n` does not mean "gaps of up to n hours" -- it means
            # "fill at most n consecutive NaNs", so a longer gap is filled
            # for its first n hours and left broken after that. Asked to
            # impute gaps of up to 6 hours for Peacock_office_Annie, whose
            # 9 missing hours are two singletons and one run of 7, it filled
            # 8: the two singletons plus six hours of a gap the question
            # excluded. A gap longer than the limit gets nothing.
            missing = grp["value"].isna()
            block = (missing != missing.shift()).cumsum()
            run_length = missing.astype(int).groupby(block).transform("sum")
            interpolated = grp["value"].interpolate(limit_area="inside")
            grp["value"] = interpolated.mask(
                missing & (run_length > max_gap_hours))
        elif method == "ffill":
            grp["value"] = grp["value"].ffill(limit=max_gap_hours)
        elif method == "zero":
            grp["value"] = grp["value"].fillna(0.0)
        else:  # seasonal_hour -- fill from that building's mean at the same hour
            hourly = grp.assign(h=grp["timestamp"].dt.hour).groupby("h")["value"].mean()
            gaps = grp["value"].isna()
            grp.loc[gaps, "value"] = grp.loc[gaps, "timestamp"].dt.hour.map(hourly)
        filled.append(grp)
    out = pd.concat(filled, ignore_index=True)
    after = int(out["value"].isna().sum())
    # A row that is still empty was never imputed, whatever it was marked as
    # before the attempt.
    out["imputed"] = out["imputed"] & out["value"].notna()
    evidence = {"method": method, "missing_before": before,
                "missing_after": after, "filled": before - after,
                "imputed_rows_flagged": int(out["imputed"].sum()),
                "values_are_imputed_where_flagged": True}
    # "How many values were imputed" is answered by this number, and it was
    # only ever in the evidence: the reply handed back the filled series and
    # left the count to be found. What is left unfilled is part of the answer
    # too -- a gap longer than the cap is deliberately not interpolated.
    evidence["headline"] = {
        "value": before - after, "unit": "values",
        "label": (f"{before - after:,} of {before:,} missing value(s) filled by "
                  f"{method.replace('_', ' ')} interpolation"
                  + (f", leaving {after:,} in gaps longer than "
                     f"{max_gap_hours} hour(s)" if after else ""))}
    if method != "weather_regression":
        evidence["max_gap_hours"] = max_gap_hours
    else:
        evidence["weather_variable"] = weather_variable
    if holdout:
        actual = np.array([a for a, _, _ in holdout], dtype=float)
        predicted = np.array([p for _, p, _ in holdout], dtype=float)
        naive = np.array([n for _, _, n in holdout], dtype=float)
        usable = (actual != 0) & np.isfinite(naive) & np.isfinite(predicted)
        if usable.any():
            def mape(values):
                return round(float(np.mean(np.abs(
                    (actual[usable] - values[usable]) / actual[usable]))
                    * 100.0), 2)
            evidence["holdout_mape_pct"] = mape(predicted)
            evidence["holdout_mape_hour_of_day_baseline_pct"] = mape(naive)
            evidence["beats_hour_of_day_baseline"] = bool(
                evidence["holdout_mape_pct"]
                < evidence["holdout_mape_hour_of_day_baseline_pct"])
        evidence["holdout_hours_scored"] = int(len(actual))
    return frame_result(out, ctx.store, name="imputed", force_handle=True,
                        evidence=evidence)


def _weather_regression_fill(ctx: ToolContext, building_id: str,
                             grp: pd.DataFrame, variable: str):
    """Fit demand on weather and the clock from present hours; predict the rest.

    Returns the frame and the hold-out pairs used to score it. A gap has no
    truth to check against, so the score comes from hours that do: a tenth of
    the present ones are hidden from the fit and compared with what was there.
    """
    from sklearn.linear_model import Ridge

    site = ctx.db.df("SELECT site_id FROM buildings WHERE building_id = ?",
                     [building_id])
    if site.empty:
        return grp, []
    weather = ctx.db.df(
        f"SELECT timestamp_local AS timestamp, {variable} AS wx "
        "FROM weather WHERE site_id = ?", [str(site["site_id"].iloc[0])])
    if weather.empty:
        return grp, []
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    merged = grp.merge(weather, on="timestamp", how="left")
    merged["wx"] = merged["wx"].interpolate(limit_direction="both")

    def design(frame: pd.DataFrame) -> np.ndarray:
        t = pd.to_datetime(frame["timestamp"])
        hour, dow = t.dt.hour, t.dt.dayofweek
        wx = frame["wx"].fillna(frame["wx"].mean())
        return np.column_stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(float),
            wx, wx ** 2,
            # Cooling and heating act on demand in opposite directions and only
            # past their own thresholds; one straight temperature term averages
            # the two into a slope that fits neither.
            np.clip(wx - 18.0, 0, None), np.clip(18.0 - wx, 0, None),
        ])

    present = merged["value"].notna() & merged["wx"].notna()
    if int(present.sum()) < 48 or merged["wx"].isna().all():
        return grp, []

    scored: list[tuple[float, float]] = []
    index = np.flatnonzero(present.to_numpy())
    if len(index) >= 200:
        hidden = index[::10]
        keep = np.setdiff1d(index, hidden)
        check = Ridge(alpha=1.0).fit(design(merged.iloc[keep]),
                                     merged.iloc[keep]["value"].to_numpy())
        predicted = check.predict(design(merged.iloc[hidden]))
        # Scored against the plainest thing anyone would do instead -- this
        # building's own average at the same hour of day. A weather-informed
        # model that cannot beat that has not earned the name, and for a flat
        # or erratic meter it often cannot. Reporting one error alone leaves a
        # reader no way to tell which case they are in.
        clock = pd.to_datetime(merged["timestamp"]).dt.hour
        by_hour = (merged.iloc[keep]["value"]
                   .groupby(clock.iloc[keep].to_numpy()).mean())
        naive = clock.iloc[hidden].map(by_hour).to_numpy(dtype=float)
        scored = list(zip(merged.iloc[hidden]["value"].to_numpy(),
                          predicted, naive))

    model = Ridge(alpha=1.0).fit(design(merged[present]),
                                 merged.loc[present, "value"].to_numpy())
    gaps = merged["value"].isna() & merged["wx"].notna()
    if gaps.any():
        merged.loc[gaps, "value"] = np.clip(
            model.predict(design(merged[gaps])), 0, None)
    return merged.drop(columns=["wx"]), scored


@tool("data_quality")
def align_timeseries(ctx: ToolContext, handles: list[str],
                     on: str = "timestamp", how: Literal["inner", "outer"] = "inner") -> ToolResult:
    """Align several fetched frames onto a common timestamp index."""
    if len(handles) < 2:
        return insufficient("too_few", "align_timeseries needs at least two handles")
    frames = []
    for i, h in enumerate(handles):
        frame = ctx.store.get_or_none(h)
        if frame is None:
            return insufficient("unknown_handle", f"no artifact for handle {h!r}")
        df = frame.copy()
        df[on] = pd.to_datetime(df[on])
        suffix = f"_{i}" if i else ""
        df = df.rename(columns={c: f"{c}{suffix}" for c in df.columns if c != on})
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=on, how=how)
    return frame_result(merged, ctx.store, name="aligned", force_handle=True,
                        evidence={"handles": handles, "how": how, "rows": len(merged)})


@tool("data_quality")
def convert_timezone(ctx: ToolContext, handle: str,
                     to: Literal["local", "utc"] = "local",
                     site_id: str | None = None) -> ToolResult:
    """Shift a frame's timestamps between UTC and site-local time.

    Uses the site's constant offset from the catalog; no DST table is
    consulted because the source series carries none.
    """
    frame = ctx.store.get_or_none(handle)
    if frame is None:
        return insufficient("unknown_handle", f"no artifact for handle {handle!r}")
    df = frame.copy()
    if "timestamp" not in df:
        return insufficient("shape", "frame needs a timestamp column")
    if site_id is None and "site_id" in df:
        sites = df["site_id"].dropna().unique().tolist()
        if len(sites) != 1:
            return insufficient("ambiguous_site",
                                f"frame spans {len(sites)} sites; pass site_id")
        site_id = sites[0]
    if site_id is None:
        return insufficient("no_site", "cannot resolve an offset without a site")
    offset = ctx.db.value("SELECT utc_offset_hours FROM sites WHERE site_id = ?", [site_id])
    if offset is None:
        return insufficient("unknown_site", f"no such site: {site_id}")
    delta = pd.Timedelta(hours=float(offset))
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + (delta if to == "local" else -delta)
    return frame_result(df, ctx.store, name=f"tz_{to}", force_handle=True,
                        evidence={"site_id": site_id, "offset_hours": float(offset),
                                  "direction": to})


METADATA_FIELDS = {
    "floor_area": "floor_area_m2",
    "year_built": "year_built",
    "occupants": "occupants",
    "number_of_floors": "number_of_floors",
    "building_type": "building_type_canonical",
    "heating_type": "heating_type_canonical",
    "coordinates": "latitude",
}


@tool("data_quality")
def profile_metadata_completeness(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare",
                           "Industrial", "Lodging", "Office", "Other", "Parking",
                           "PublicServices", "ReligiousWorship", "Retail",
                           "Services", "TechnologyScience", "Utility",
                           "Warehouse"] | None = None,
    site_id: str | None = None,
    country: str | None = None,
    fields: list[Literal["floor_area", "year_built", "occupants",
                         "number_of_floors", "building_type", "heating_type",
                         "coordinates"]] | None = None,
    group_by: Literal["none", "site", "country", "building_type",
                      "building_subtype", "industry", "subindustry"] = "none",
) -> ToolResult:
    """How much of the building metadata is actually recorded, per field.

    Answers "assess metadata completeness": one row per field with how many
    buildings carry it and the share that do, plus an overall figure. This is
    about the catalog, not the meters -- profile_data_quality measures coverage
    of readings and answers a different question.

    Occupancy counts as present only when it is above zero. Forty-one buildings
    record exactly nought occupants, which is a value and not a measurement,
    and counting it as recorded overstates completeness by the same forty-one.
    """
    chosen = list(fields or ("year_built", "floor_area", "occupants"))
    unknown = [f for f in chosen if f not in METADATA_FIELDS]
    if unknown:
        return insufficient("unknown_field",
                            f"no such metadata field: {unknown}; "
                            f"valid: {sorted(METADATA_FIELDS)}")

    params: list = []
    where = "1=1"
    where += in_clause("b.building_id", building_ids, params)
    if building_type:
        where += " AND b.building_type_canonical = ?"; params.append(building_type)
    if site_id:
        where += " AND b.site_id = ?"; params.append(site_id)
    if country:
        where += " AND s.country_canonical = ?"; params.append(country)

    # A group whose value is missing is still a group. Grouping on the raw
    # column dropped the buildings whose country is not recorded, so a matrix
    # of *missing metadata* silently omitted the rows with the most of it.
    grouping = {"none": "'all'",
                "site": "COALESCE(b.site_id, '(no site recorded)')",
                "country": "COALESCE(s.country_canonical, '(no country recorded)')",
                "building_type": "COALESCE(b.building_type_canonical, "
                                 "'(no building type recorded)')",
                # A row for the buildings that carry no value is the point of
                # a completeness report, so each of these keeps its unrecorded
                # group rather than dropping it: industry is recorded on 579
                # of 1,636, and a table that silently omitted the other 1,057
                # would report the opposite of what was asked.
                "building_subtype": "COALESCE(b.building_subtype, "
                                    "'(no subtype recorded)')",
                "industry": "COALESCE(b.industry, '(no industry recorded)')",
                "subindustry": "COALESCE(b.subindustry, "
                               "'(no subindustry recorded)')"}[group_by]

    present = ", ".join(
        f"SUM(CASE WHEN b.{METADATA_FIELDS[f]} IS NOT NULL"
        + (" AND b.occupants > 0" if f == "occupants" else "")
        + f" THEN 1 ELSE 0 END) AS {f}"
        for f in chosen)
    df = ctx.db.df(f"""
        SELECT {grouping} AS grouping, COUNT(*) AS buildings, {present}
        FROM buildings b JOIN sites s ON s.site_id = b.site_id
        WHERE {where} GROUP BY 1 ORDER BY 1""", params or None)
    if df.empty:
        return empty("no buildings matched the filters")

    rows = []
    for _, row in df.iterrows():
        total = int(row["buildings"])
        recorded = 0
        for field in chosen:
            have = int(row[field])
            recorded += have
            rows.append({"grouping": row["grouping"], "field": field,
                         "buildings": total, "recorded": have,
                         "missing": total - have,
                         "completeness": round(have / total, 4) if total else None})
        rows.append({"grouping": row["grouping"], "field": "ALL FIELDS",
                     "buildings": total, "recorded": recorded,
                     "missing": total * len(chosen) - recorded,
                     "completeness": round(recorded / (total * len(chosen)), 4)
                     if total else None})

    frame = pd.DataFrame(rows)
    # Name the column after the dimension it holds. "grouping" tells a reader
    # nothing, and an answer whose members are countries should say country.
    if group_by != "none":
        frame = frame.rename(columns={"grouping": group_by})
    overall = frame.loc[frame["field"] == "ALL FIELDS", "completeness"]
    # "Assess the metadata completeness" is answered by one figure, and the
    # field quotes completeness as a percentage. The table reports the share
    # per field as a fraction; the answer says what share of the attributes
    # asked about are present, across the buildings asked about.
    headline = None
    if group_by == "none" and len(overall) == 1 and overall.iloc[0] is not None:
        headline = {"label": f"metadata completeness across {', '.join(chosen)}",
                    "value": round(float(overall.iloc[0]) * 100, 2), "unit": "%"}
    return frame_result(frame, ctx.store, name="metadata_completeness",
                        evidence={**({"headline": headline} if headline else {}),
                                  "fields": chosen, "group_by": group_by,
                                  "buildings": int(df["buildings"].sum()),
                                  "metric": "completeness",
                                  "overall_completeness":
                                      round(float(overall.mean()), 4)
                                      if len(overall) else None,
                                  "occupancy_zero_counts_as_missing": True})
