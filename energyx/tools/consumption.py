"""Consumption, Demand & Load Analytics tools (7).

Covers task categories 3 and 4. Everything is computed in local building time,
because peak timing and operating hours are meaningless otherwise.

Consolidated from 18 narrower operations by parameterising: one
``calculate_load_profile(profile=...)`` replaces four separate profile tools.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..knowledge.cache import try_cached
from ._common import budgeted, in_clause, load_series, season_of, with_local_parts
from .base import ToolResult, empty, frame_result, insufficient, ok
from .registry import ToolContext, tool

#: Every answer is read from the cleaned series. The filter changes no figure
#: computed with SUM, AVG or a quantile, which skip NULLs anyway; what it
#: changes is every COUNT(*), where an unfiltered masked hour counts as an hour
#: of data. The daily and monthly aggregates are built with COUNT(value) and
#: carry the masked hours separately, so they need none of this.
CLEAN_READINGS = " AND e.quality_flag = 'OK'"

#: Month numbers as people write them. EXTRACT(month ...) is what the group-by
#: needs and is not what the answer to "during which month" looks like.
MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April",
               5: "May", 6: "June", 7: "July", 8: "August",
               9: "September", 10: "October", 11: "November", 12: "December"}


@tool("consumption")
def calculate_consumption_statistics(
    ctx: ToolContext,
    # Grouping by a dimension is a question about the whole portfolio, so the
    # cohort is optional here as it is elsewhere: omit it for all buildings.
    building_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    hour_from: int | None = None,
    hour_to: int | None = None,
    days: Literal["all", "weekday", "weekend"] = "all",
    weight: Literal["readings", "building"] = "readings",
    # "The total consumption of the largest 10% of buildings" is the sum over
    # the group, not the typical building in it. Offered only mean and median,
    # a plan asking for a group total wrote group_aggregate="sum" and the run
    # died on the enum with nothing computed.
    group_aggregate: Literal["mean", "median", "total"] = "mean",
    metrics: list[Literal["total", "mean", "median", "min", "max", "p95", "p05",
                          "stddev", "cv", "n_observations"]] | None = None,
    across: Literal["building", "cohort", "building_type", "building_subtype",
                    "industry", "subindustry", "city",
                    "country", "site",
                    "month", "year", "date", "hour_of_day",
                    "day_of_week"] = "building",
    grain: Literal["hourly", "daily"] = "hourly",
    order_by: str | None = None,
    order: Literal["asc", "desc"] = "desc",
    limit: int | None = None,
) -> ToolResult:
    """Total, mean, median, min, max, percentiles and variability.

    ``across`` says what the statistics are *of*. "building" gives one row per
    building, which answers "what does each of these buildings do". "cohort"
    pools every reading into one row, which answers "what is the mean and
    median demand for all Healthcare buildings" -- a question about the group,
    not about its members, and one this returned 27 rows to.

    The two differ by more than presentation. Pooling weights each building by
    how many readings it has; averaging per-building means weights every
    building equally whatever its size. For Healthcare here that is 308.163
    against 300.893, so the choice is made deliberately rather than fallen
    into.

    ``across`` also groups by time: month, year, hour_of_day or day_of_week,
    which is how "which month had the highest use" and "which hour is the
    peakiest" are asked. Nothing else here groups by a calendar period, so such
    questions had to be answered by fetching the series and reading it.

    ``across`` also groups by a dimension: building_type, city, country or
    site. "Which primary space usage category shows the highest demand
    variability" is one row per building type, and nothing here could produce
    that -- per building or nothing, with a two-cohort comparison as the only
    grouping tool in the box.

    ``grain`` chooses what a reading is. "hourly" summarises the meter's own
    readings; "daily" summarises each day's total first, which is what
    "day-to-day variability" means. The two are not close: the median standard
    deviation across industrial buildings is 44 at hourly grain and 616 at
    daily, because one describes how much demand moves within a day and the
    other how much a day's consumption differs from the next day's.

    ``order_by`` names one of the metrics and ``limit`` cuts to the top of it,
    so "which category has the highest X" comes back as the one row that
    answers it rather than sixteen for a reader to scan. The ordered result
    carries a rank, which is what lets a single row be read as the winner.
    """
    available = {"total", "mean", "median", "min", "max", "p95", "p05",
                 "stddev", "cv", "n_observations"}
    # The snapshot's own columns are total_kwh, peak_demand, mean_demand, so a
    # planner reading the data model asks for those names. Refusing them
    # outright answered "what was the total consumption" with a refusal while
    # the number sat one synonym away.
    SYNONYMS = {"total_kwh": "total", "sum": "total", "consumption": "total",
                "kwh": "total", "peak": "max", "peak_demand": "max",
                "maximum": "max", "minimum": "min", "avg": "mean",
                "average": "mean", "mean_demand": "mean", "count": "n_observations"}
    asked = list(metrics or sorted(available))
    renamed = {m: SYNONYMS[m] for m in asked
               if m not in available and m in SYNONYMS}
    chosen = [SYNONYMS.get(m, m) for m in asked]
    chosen = list(dict.fromkeys(m for m in chosen if m in available))
    if not chosen:
        return insufficient("unknown_metrics",
                            f"none of {asked} is a metric; valid: "
                            f"{sorted(available)}")
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    if start:
        where += " AND timestamp_local >= ?"; params.append(start)
    if end:
        where += " AND timestamp_local < ?"; params.append(end)
    # "Average nighttime consumption between 20:00 and 07:00" had no way to be
    # asked: the statistics ran over the whole day, so a question about the
    # night was answered with a figure that is mostly the afternoon, and named
    # the wrong building type. A window that wraps midnight is the normal case
    # for night hours, so hour_from > hour_to means "from 20:00 round to
    # 07:00" rather than an empty range.
    if hour_from is not None or hour_to is not None:
        low = 0 if hour_from is None else int(hour_from)
        high = 23 if hour_to is None else int(hour_to)
        if low <= high:
            where += " AND EXTRACT(hour FROM timestamp_local) BETWEEN ? AND ?"
            params.extend([low, high])
        else:
            where += (" AND (EXTRACT(hour FROM timestamp_local) >= ?"
                      " OR EXTRACT(hour FROM timestamp_local) <= ?)")
            params.extend([low, high])
    if days != "all":
        # ISODOW: 6 and 7 are Saturday and Sunday.
        where += (" AND ISODOW(timestamp_local) >= 6" if days == "weekend"
                  else " AND ISODOW(timestamp_local) <= 5")
    # The main query aliases electricity as e and may join buildings, so its
    # filter has to name the table or building_id is ambiguous. The span query
    # below is unaliased and keeps the plain one.
    qualified = where.replace("building_id", "e.building_id").replace(
        "timestamp_local", "e.timestamp_local") + CLEAN_READINGS
    DIMENSIONS = {"building": ("e.building_id", "building_id"),
                  "cohort": ("'all buildings'", "building_id"),
                  "building_type": ("b.building_type_canonical", "building_type"),
                  "city": ("s.city", "city"),
                  "country": ("s.country_canonical", "country"),
                  "site": ("b.site_id", "site_id"),
                  # Recorded for 7 of the 13 industrial buildings, so a
                  # segment here is one or two buildings and the evidence says
                  # how many.
                  "subindustry": ("b.subindustry", "subindustry"),
                  # The subtype is the finer reading of the type and is
                  # recorded on 1,615 of 1,636 buildings, so grouping by it is
                  # usually possible where grouping by industry is not.
                  # "Dormitory" and "College Classroom" are both Education and
                  # have nothing else in common.
                  "building_subtype": ("b.building_subtype", "building_subtype"),
                  "industry": ("b.industry", "industry"),
                  # Calendar grouping is over local time, which is what a
                  # question about months or working hours means.
                  "month": ("EXTRACT(month FROM e.{stamp})", "month"),
                  "year": ("EXTRACT(year FROM e.{stamp})", "year"),
                  "hour_of_day": ("EXTRACT(hour FROM e.{stamp})", "hour_of_day"),
                  "day_of_week": ("EXTRACT(dow FROM e.{stamp})", "day_of_week"),
                  # The calendar date, which is a different grouping from the
                  # day of the week: "on which day did this building use the
                  # most" wants 22 June, and day_of_week answers "Thursday".
                  # Without it the question had no route at all -- grain
                  # "daily" aggregates the days away rather than keeping them.
                  "date": ("CAST(e.{stamp} AS DATE)", "date")}
    expression, label = DIMENSIONS[across]
    expression = expression.format(
        stamp="period_start" if grain == "daily" else "timestamp_local")
    CALENDAR = ("month", "year", "hour_of_day", "day_of_week")
    joins = ("" if across in ("building", "cohort") + CALENDAR else
             " JOIN buildings b ON b.building_id = e.building_id" +
             (" JOIN sites s ON s.site_id = b.site_id"
              if across in ("city", "country") else ""))
    # At daily grain a "reading" is a day's total, so the source is the daily
    # aggregate and the filters name its own columns.
    source, reading = (("agg_daily", "total_kwh") if grain == "daily"
                       else ("electricity", "value"))
    if grain == "daily":
        # agg_daily has no quality_flag and needs none: it is built with
        # COUNT(value) and SUM(value), so its counts and totals already
        # exclude the masked hours and count them under n_zero_masked.
        qualified = (where.replace("building_id", "e.building_id")
                     .replace("timestamp_local", "e.period_start")
                     + " AND e.grain = 'daily'")
    # A dimension with no value is not a group: 400 buildings sit at sites with
    # no country recorded, and a null row in a ranking reads as a real one.
    dimension_filter = ("" if across in ("building", "cohort") + CALENDAR
                        else f" AND {expression} IS NOT NULL")
    df = ctx.db.df(f"""
        SELECT {expression} AS {label},
               SUM(e.{reading})                  AS total,
               AVG(e.{reading})                  AS mean,
               MEDIAN(e.{reading})               AS median,
               MIN(e.{reading})                  AS min,
               MAX(e.{reading})                  AS max,
               QUANTILE_CONT(e.{reading}, 0.95)  AS p95,
               QUANTILE_CONT(e.{reading}, 0.05)  AS p05,
               STDDEV_SAMP(e.{reading})          AS stddev,
               STDDEV_SAMP(e.{reading}) / NULLIF(AVG(e.{reading}), 0) AS cv,
               COUNT(e.{reading})                AS n_observations
        FROM {source} e{joins} WHERE {qualified}{dimension_filter}
        {"" if across == "cohort" else "GROUP BY 1 ORDER BY 1"}
    """, params or None)
    if df.empty:
        return empty("no measurements in the requested window")
    # "Which building type has the highest average nighttime consumption" is a
    # question about the typical building of that type, and pooling every
    # reading answers a different one: the Bull site has 17 assembly buildings
    # and 2 public-service ones, so pooling lets the seventeen outvote the two
    # and named the wrong type by 0.2%. weight="building" averages within each
    # building first and then across buildings, which is what "average across
    # buildings of the same use" means.
    if weight == "building" and across not in ("building", "cohort"):
        stages = ctx.db.df(f"""
            SELECT {expression} AS {label}, e.building_id,
                   SUM(e.{reading}) AS total, AVG(e.{reading}) AS mean,
                   MEDIAN(e.{reading}) AS median, MIN(e.{reading}) AS min,
                   MAX(e.{reading}) AS max,
                   QUANTILE_CONT(e.{reading}, 0.95) AS p95,
                   QUANTILE_CONT(e.{reading}, 0.05) AS p05,
                   STDDEV_SAMP(e.{reading}) AS stddev,
                   STDDEV_SAMP(e.{reading}) / NULLIF(AVG(e.{reading}), 0) AS cv,
                   COUNT(e.{reading}) AS n_observations
            FROM {source} e{joins} WHERE {qualified}{dimension_filter}
            GROUP BY 1, 2
        """, params or None)
        if not stages.empty:
            columns = [c for c in stages.columns if c not in (label, "building_id")]
            # "The highest *median* within-building variability" asks for the
            # middle building of each category, not its average: with a
            # category whose CVs run from 0.24 to 1.31 the two differ by a
            # fifth, and the mean is pulled by whichever building is most
            # erratic -- exactly the one the question is trying to place.
            grouped = stages.groupby(label)[columns]
            # A group total sums what each building contributed. Summing a
            # mean or a percentile would be meaningless, so only the additive
            # columns are summed and the rest keep the middle building's
            # figure -- the sum of two buildings' p95 is not the group's p95.
            if group_aggregate == "total":
                additive = [c for c in columns
                            if c in ("total", "n_observations")]
                rolled = grouped.median().reset_index()
                if additive:
                    sums = grouped[additive].sum().reset_index()
                    rolled = rolled.drop(columns=additive).merge(sums, on=label)
            else:
                rolled = (grouped.median() if group_aggregate == "median"
                          else grouped.mean()).reset_index()
            rolled["n_buildings"] = (stages.groupby(label)["building_id"]
                                     .nunique().to_numpy())
            # cv is carried from the per-building stage, not recomputed from
            # the rolled-up columns: median(stddev)/median(mean) is not the
            # median of the ratios, and on this cohort it reads 1.08 where the
            # middle building's CV is 0.70.
            if "cv" not in stages.columns:
                rolled["cv"] = rolled["stddev"] / rolled["mean"].replace(0, np.nan)
            df = rolled
    # A segmented answer with no cohort figure beside it cannot be read: a
    # reader comparing one subindustry to another has no idea what either
    # means for the population. Asked for day-to-day variability segmented by
    # subindustry, the answer is a table of groups and the figure it is
    # usually quoted as -- the middle building of the whole cohort -- was
    # nowhere in it.
    cohort_median = {}
    if across not in ("building", "cohort"):
        per_building = ctx.db.df(f"""
            SELECT e.building_id,
                   SUM(e.{reading}) AS total, AVG(e.{reading}) AS mean,
                   MEDIAN(e.{reading}) AS median, MIN(e.{reading}) AS min,
                   MAX(e.{reading}) AS max,
                   QUANTILE_CONT(e.{reading}, 0.95) AS p95,
                   QUANTILE_CONT(e.{reading}, 0.05) AS p05,
                   STDDEV_SAMP(e.{reading}) AS stddev,
                   STDDEV_SAMP(e.{reading}) / NULLIF(AVG(e.{reading}), 0) AS cv
            FROM {source} e WHERE {qualified}
            GROUP BY 1""", params or None)
        cohort_median = {f"median_{name}_per_building":
                         round(float(per_building[name].median()), 4)
                         for name in chosen
                         if name in per_building.columns
                         and per_building[name].notna().any()}
    # Report the window actually covered, not only the one requested: `end` is
    # exclusive, so a requested "2017-12-31" covers through the 30th and the
    # difference is invisible in a bare total.
    covered = ctx.db.df(f"""
        SELECT MIN(timestamp_local) AS first, MAX(timestamp_local) AS last,
               COUNT(*) AS n_readings
        FROM electricity
        WHERE {where} AND quality_flag = 'OK'""", params or None)
    span = covered.iloc[0].to_dict() if not covered.empty else {}
    ordered_by = None
    # "Which category shows the highest X" arrives with no order_by, so the
    # sole metric orders the result rather than leaving the limit arbitrary.
    sort_column = SYNONYMS.get(order_by, order_by) if order_by else None
    if sort_column not in df.columns:
        sort_column = None
    if sort_column is None and limit and len(chosen) > 1:
        return insufficient(
            "ambiguous_ranking",
            f"a limit of {limit} was asked for over {len(chosen)} metrics "
            f"({', '.join(chosen)}) with no order_by, so there is no one "
            f"ranking to cut: name the metric to rank on")
    if sort_column is None and len(chosen) == 1:
        sort_column = chosen[0]
    if sort_column:
        column = sort_column
        if column in df.columns:
            df = df.sort_values(column, ascending=(order == "asc"))
            ordered_by = f"{column} {order}"
            if limit:
                df = df.head(int(limit))
            df.insert(0, "rank", range(1, len(df) + 1))
    extra_columns = [c for c in ("share_in_top_hours", "hours_in_top_share")
                     if c in df.columns]
    # A month is a name, not a magnitude. "During which month did X consume
    # the most" came back as "1 kWh": the month number sitting in the value
    # column, which a reader takes for one kilowatt-hour. The number stays,
    # because everything sorts on it, and the name travels beside it.
    named = []
    if label == "month" and "month" in df.columns:
        df = df.assign(month_name=df["month"].map(
            lambda m: MONTH_NAMES.get(int(m)) if pd.notna(m) else None))
        named = ["month_name"]
    columns = ((["rank"] if ordered_by else []) + [label] + named + [*chosen]
               + extra_columns)
    return frame_result(df[columns], ctx.store,
                        name="consumption_statistics",
                        evidence={"period_requested": [start, end],
                                  "end_is_exclusive": True,
                                  "first_reading": str(span.get("first")),
                                  "last_reading": str(span.get("last")),
                                  "n_readings": int(span.get("n_readings") or 0),
                                  "unit": "kWh", "across": across,
                                  "ordered_by": ordered_by,
                                  "groups": len(df),
                                  "buildings": len(df),
                                  **cohort_median,
                                  **({"renamed_metrics": renamed} if renamed else {})})


@tool("consumption")
def calculate_demand_metrics(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    months: list[int] | None = None,
    top_hours_share_pct: float | None = None,
    percentiles: list[float] | None = None,
    across: Literal["building", "cohort"] = "building",
    order_by: Literal["min_demand", "peak_demand", "mean_demand",
                      "load_factor", "daily_load_factor",
                      "median_daily_load_factor",
                      "peak_to_average_ratio"] | None = None,
    order: Literal["asc", "desc"] = "desc",
    limit: int | None = None,
) -> ToolResult:
    """Minimum and peak demand, peak hour, load factor and peak-to-average ratio.

    A tool reporting the peak and not the trough cannot answer "which building
    recorded the minimum demand", and the table it returns looks like it should
    -- one row per building, a demand column, an obvious extreme to read off.
    The lowest reading of the period is min_demand, beside the highest, with
    the hour it happened.

    Two load factors, because "the load factor over the summer" and "the daily
    load factor over the summer" are different questions and the gap between
    them is not small. ``load_factor`` is the window's mean over the window's
    peak, which for site Peacock across June to August is 0.14 -- one hot
    afternoon sets the denominator for the whole season. ``daily_load_factor``
    is each day's mean over that day's own peak, averaged over the days, which
    is 0.81. A question asking for a daily figure means the second.

    ``months`` selects a season wherever it falls, which start and end cannot:
    "over the summer" across a two-year record is June to August of both years,
    two disjoint ranges that no single window covers. Every site here is in the
    northern hemisphere, so summer is [6, 7, 8].

    ``top_hours_share_pct`` answers "what share of consumption falls in the
    peakiest hours": at 5 it reports the fraction of the period's kWh drawn
    during its highest-demand 5% of hours, as share_in_top_hours. A flat
    building lands near 0.05 because every hour is alike; 0.12 means the peak
    hours carry two and a half times their share, which is what a demand charge
    is levied on.

    ``across="cohort"`` answers "the daily load factor for site Peacock" as one
    number rather than one per building, pooling every building-day the way a
    cohort statistic pools every reading. Averaging the per-building averages
    instead weights a building with twenty days the same as one with two
    hundred, which is a different figure: 0.8099 against 0.8063.

    "Which building recorded the minimum demand on 1 January" is one building,
    and this returned all 1,529 of them for a reader to scan. ``order_by`` with
    ``limit=1`` asks the question as asked. Read the extreme here rather than
    off an unordered table: the same question, answered by eye from a frame
    built at daily resolution, once named a building whose *daily total* was
    smallest, which is a different quantity and a different building.
    """
    # "The 95th percentile of hourly demand" had no way to be asked: the tool
    # reported the max, the min and the mean, and a plan wanting a quantile
    # got the peak instead -- a different number that reads like an answer.
    # Accepted as either a fraction (0.95) or a percent (95), because both
    # spellings turn up in questions and guessing wrong is silent.
    wanted: list[float] = []
    for value in percentiles or []:
        try:
            share = float(value)
        except (TypeError, ValueError):
            return insufficient("percentiles", f"{value!r} is not a number")
        share = share / 100.0 if share > 1 else share
        if not 0.0 < share < 1.0:
            return insufficient(
                "percentiles",
                f"{value!r} is outside the open range 0 to 1 (or 0 to 100)")
        wanted.append(share)
    wanted = sorted(set(wanted))
    percentile_columns = "".join(
        f",\n               QUANTILE_CONT(e.value, {share}) AS p{share * 100:g}_demand"
        for share in wanted)

    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    if start:
        where += " AND timestamp_local >= ?"; params.append(start)
    if end:
        where += " AND timestamp_local < ?"; params.append(end)
    if months:
        where += (" AND EXTRACT(month FROM timestamp_local) IN ("
                  + ", ".join("?" for _ in months) + ")")
        params.extend(int(m) for m in months)
    df = ctx.db.df(f"""
        WITH per_day AS (
            SELECT building_id, CAST(timestamp_local AS DATE) AS day,
                   AVG(value) / NULLIF(MAX(value), 0) AS day_load_factor
            FROM electricity WHERE {where} AND quality_flag = 'OK'
            GROUP BY 1, 2
        ), daily AS (
            SELECT building_id, AVG(day_load_factor) AS daily_load_factor,
                   MEDIAN(day_load_factor) AS median_daily_load_factor,
                   COUNT(*) AS n_days
            FROM per_day GROUP BY 1
        ), pooled AS (
            SELECT AVG(day_load_factor) AS daily_load_factor,
                   MEDIAN(day_load_factor) AS median_daily_load_factor,
                   COUNT(*) AS n_days
            FROM per_day
        )
        SELECT {"'all buildings' AS building_id" if across == "cohort"
                else "e.building_id"},
               MAX(e.value) AS peak_demand,
               MIN(e.value) AS min_demand,
               AVG(e.value) AS mean_demand,
               ARG_MAX(e.timestamp_local, e.value) AS peak_timestamp_local,
               ARG_MAX(EXTRACT(hour FROM e.timestamp_local), e.value) AS peak_hour,
               ARG_MIN(EXTRACT(hour FROM e.timestamp_local), e.value) AS min_hour,
               AVG(e.value) / NULLIF(MAX(e.value), 0) AS load_factor,
               {"(SELECT daily_load_factor FROM pooled)"
                if across == "cohort"
                else "ANY_VALUE(daily.daily_load_factor)"} AS daily_load_factor,
               {"(SELECT median_daily_load_factor FROM pooled)"
                if across == "cohort"
                else "ANY_VALUE(daily.median_daily_load_factor)"}
                   AS median_daily_load_factor,
               {"(SELECT n_days FROM pooled)" if across == "cohort"
                else "ANY_VALUE(daily.n_days)"} AS n_days,
               MAX(e.value) / NULLIF(AVG(e.value), 0) AS peak_to_average_ratio,
               COUNT(e.value) AS n_observations{percentile_columns}
        FROM electricity e
        LEFT JOIN daily ON daily.building_id = e.building_id
        WHERE {where.replace("building_id", "e.building_id").replace("timestamp_local", "e.timestamp_local")}{CLEAN_READINGS}
        {"" if across == "cohort" else "GROUP BY 1 ORDER BY 1"}
    """, (params or None) if not params else params + params)
    if df.empty:
        return empty("no measurements in the requested window")
    if top_hours_share_pct is not None:
        fraction = max(0.0, min(1.0, float(top_hours_share_pct) / 100.0))
        share = ctx.db.df(f"""
            WITH ranked AS (
                SELECT {"'all buildings'" if across == "cohort"
                        else "e.building_id"} AS building_id, e.value,
                       ROW_NUMBER() OVER (
                           PARTITION BY {"1" if across == "cohort"
                                         else "e.building_id"}
                           ORDER BY e.value DESC) AS rn,
                       COUNT(*) OVER (
                           PARTITION BY {"1" if across == "cohort"
                                         else "e.building_id"}) AS n
                FROM electricity e WHERE {where.replace("building_id", "e.building_id").replace("timestamp_local", "e.timestamp_local")}{CLEAN_READINGS}
            )
            SELECT building_id,
                   SUM(CASE WHEN rn <= CEIL(n * ?) THEN value ELSE 0 END)
                       / NULLIF(SUM(value), 0) AS share_in_top_hours,
                   MAX(CAST(CEIL(n * ?) AS INTEGER)) AS hours_in_top_share
            FROM ranked GROUP BY 1
        """, params + [fraction, fraction])
        df = df.merge(share, on="building_id", how="left")
        # Asking for this is asking for this: a row of demand metrics carries a
        # dozen numbers and the answer came back as the load factor, 0.63,
        # where the share is 0.1236. When one building was asked about, say
        # which of the numbers was the point of the call.
        if len(df) == 1 and pd.notna(df["share_in_top_hours"].iloc[0]):
            share_headline = {
                "label": f"share of consumption in the highest-demand "
                         f"{top_hours_share_pct:g}% of hours",
                "value": round(float(df["share_in_top_hours"].iloc[0]), 6),
                "unit": None}
        else:
            share_headline = None
    else:
        share_headline = None

    # Same reasoning for a named percentile: the row carries fourteen numbers
    # and the question asked for one of them. Without this the answer reads as
    # a table and the reader has to know which column was the point.
    if share_headline is None and wanted and len(df) == 1:
        column = f"p{wanted[0] * 100:g}_demand"
        if column in df.columns and pd.notna(df[column].iloc[0]):
            share_headline = {
                "label": f"{wanted[0] * 100:g}th percentile of hourly demand",
                "value": round(float(df[column].iloc[0]), 4),
                "unit": "kW"}

    ordered_by = None
    if order_by and order_by in df.columns:
        df = df.sort_values(order_by, ascending=(order == "asc"))
        ordered_by = f"{order_by} {order}"
        if limit:
            df = df.head(int(limit))
        # The rank says the order is meant, so one row can be read as the
        # winner rather than as whichever row happened to arrive first.
        df.insert(0, "rank", range(1, len(df) + 1))
        # "Which building had the lowest median daily load factor" is answered
        # by a name *and* a number. Naming the building alone is right and
        # incomplete -- a reader cannot tell 0.533 from 0.9 -- and it left the
        # figure sitting in a column of fourteen.
        if share_headline is None and len(df) and order_by in df.columns:
            top = df.iloc[0]
            subject = top.get("building_id") or top.get("building_type")
            value = top[order_by]
            if subject is not None and pd.notna(value):
                extreme = "lowest" if order == "asc" else "highest"
                share_headline = {
                    "label": f"{subject} — {extreme} {order_by.replace('_', ' ')}",
                    "value": round(float(value), 4), "unit": None}
    return frame_result(df, ctx.store, name="demand_metrics",
                        evidence={**({"headline": share_headline}
                                     if share_headline else {}),
                                  "period": [start, end],
                                  "ordered_by": ordered_by,
                                  "buildings_ranked": len(df) if ordered_by else None,
                                  "note": "load factor = mean / peak over the window"})


@tool("consumption")
def calculate_load_profile(
    ctx: ToolContext,
    building_ids: list[str],
    # "How does the weekday profile differ from the weekend one" is a
    # comparison of two hour-of-day curves, and the tool could only build one
    # curve over all days: the answer came back as a single 24-row profile
    # with the difference the question asked about nowhere in it.
    profile: Literal["hourly", "weekly", "monthly", "seasonal",
                     "weekday_weekend"] = "hourly",
    start: str | None = None,
    end: str | None = None,
    normalize: bool = False,
    peak_only: bool = False,
) -> ToolResult:
    """Typical load shape at hourly, weekly, monthly or seasonal granularity.

    ``hourly`` and ``weekly`` read the precomputed profile tables when the whole
    period is requested; a windowed request is computed from the raw series.

    ``peak_only`` returns the single period of highest mean demand, which is
    what "at what hour does it typically peak" asks. The profile comes back
    ordered by hour otherwise, so reading the first row answers midnight: for
    Lamb_office_Bertha that is hour 0, and the reason is a defect rather than a
    habit -- on 362 of its 731 days the meter repeats one value through hours 1
    to 23, so midnight wins each of those days by default. Averaging over all
    days dilutes the flat ones and gives hour 10; taking the mode of the daily
    peak hour lets them win outright.
    """
    windowed = bool(start or end)
    if profile == "weekday_weekend":
        raw = load_series(ctx, building_ids=building_ids, start=start, end=end)
        if raw.empty:
            return empty("no measurements in the requested window")
        parts = with_local_parts(raw)
        parts["day_type"] = np.where(parts["dow"] >= 5, "weekend", "weekday")
        df = (parts.groupby(["building_id", "day_type", "hour"])["value"]
                   .agg(mean_demand="mean", median_demand="median",
                        n_observations="count")
                   .reset_index().rename(columns={"hour": "hour_of_day"}))
    elif profile in ("hourly", "weekly") and not windowed:
        view = "profile_daily" if profile == "hourly" else "profile_weekly"
        cols = "hour_of_day" if profile == "hourly" else "day_of_week, hour_of_day"
        params: list = []
        where = "1=1" + in_clause("building_id", building_ids, params)
        df = ctx.db.df(f"""
            SELECT building_id, {cols},
                   SUM(mean_demand) AS mean_demand,
                   SUM(median_demand) AS median_demand,
                   MAX(n_observations) AS n_observations
            FROM {view} WHERE {where}
            GROUP BY building_id, {cols} ORDER BY building_id, {cols}
        """, params or None)
    else:
        raw = load_series(ctx, building_ids=building_ids, start=start, end=end)
        if raw.empty:
            return empty("no measurements in the requested window")
        parts = with_local_parts(raw)
        keys = {"hourly": ["hour"], "weekly": ["dow", "hour"],
                "monthly": ["month"], "seasonal": ["season"]}[profile]
        if profile == "seasonal":
            parts["season"] = season_of(parts["month"])
        df = (parts.groupby(["building_id", *keys])["value"]
                   .agg(mean_demand="mean", median_demand="median",
                        n_observations="count")
                   .reset_index())
    if df.empty:
        return empty("no profile could be computed")
    split_headline = None
    if profile == "weekday_weekend":
        # The two curve averages and the gap between them: that is what the
        # question asks, and a 48-row table states it only by implication.
        levels = df.groupby("day_type")["mean_demand"].mean()
        weekday, weekend = levels.get("weekday"), levels.get("weekend")
        if weekday is not None and weekend is not None and weekend:
            busiest = df[df["day_type"] == "weekday"].nlargest(1, "mean_demand")
            split_headline = {
                "value": round(float(weekday), 4), "unit": "kWh/h",
                "label": (f"weekday profile averages {weekday:,.1f} kWh/h vs "
                          f"{weekend:,.1f} at the weekend "
                          f"({100 * (weekday - weekend) / weekend:+.1f}%)"
                          + (f"; weekday peak at "
                             f"{int(busiest['hour_of_day'].iloc[0]):02d}:00 "
                             f"({busiest['mean_demand'].iloc[0]:,.1f} kWh)"
                             if not busiest.empty else ""))}
    if normalize:
        df["mean_demand"] = df.groupby("building_id")["mean_demand"].transform(
            lambda s: s / s.mean() if s.mean() else s)
    peak_headline = None
    if peak_only and not df.empty:
        df = (df.sort_values("mean_demand", ascending=False)
                .groupby("building_id", as_index=False).head(1)
                .sort_values("building_id").reset_index(drop=True))
        # The row holds the building, the hour and two demands, and "at what
        # hour" was answered with the building's name. Say which column is the
        # answer when exactly one building was asked about.
        period_column = next((c for c in ("hour_of_day", "day_of_week",
                                          "month", "season")
                              if c in df.columns), None)
        if len(df) == 1 and period_column:
            peak_headline = {
                "label": f"{period_column.replace('_', ' ')} of the highest "
                         "mean demand in the typical profile",
                "value": int(df[period_column].iloc[0]), "unit": None}
    # Where the curve is for one building, say where it peaks. The answer to
    # "show the average 24-hour pattern" is the curve, so this is evidence
    # rather than a headline -- but the hour it peaks is the one number the
    # curve is read for, and leaving it unsaid means the figure is the only
    # place it exists.
    peak_period = None
    if not peak_only and not df.empty and df["building_id"].nunique() == 1:
        column = next((c for c in ("hour_of_day", "day_of_week", "month", "season")
                       if c in df.columns), None)
        if column:
            peak_period = {column: int(df.loc[df["mean_demand"].idxmax(), column])}
    return frame_result(df, ctx.store, name=f"load_profile_{profile}",
                        evidence={**({"headline": peak_headline or split_headline}
                                     if (peak_headline or split_headline) else {}),
                                  **({"peak_of_profile": peak_period}
                                     if peak_period else {}),
                                  "profile": profile, "normalized": normalize,
                                  "peak_only": peak_only,
                                  "period": [start, end],
                                  "timezone": "local building time"})


def _month_span(months: list[int]) -> str:
    """A month set named the way a reader would say it: "Sep-Dec"."""
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ordered = sorted({int(m) for m in months if 1 <= int(m) <= 12})
    if not ordered:
        return "no months"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return (names[ordered[0] - 1] if len(ordered) == 1
                else f"{names[ordered[0] - 1]}-{names[ordered[-1] - 1]}")
    return "/".join(names[m - 1] for m in ordered)


@tool("consumption")
def compare_temporal_profiles(
    ctx: ToolContext,
    building_ids: list[str],
    comparison: Literal["weekday_weekend", "summer_winter", "year_year",
                        "day_night", "named_days",
                        "named_months"] = "weekday_weekend",
    measure: Literal["mean_demand", "peak_hour", "profile_shape"] = "mean_demand",
    day_a: Literal["monday", "tuesday", "wednesday", "thursday", "friday",
                   "saturday", "sunday"] | None = None,
    day_b: Literal["monday", "tuesday", "wednesday", "thursday", "friday",
                   "saturday", "sunday"] | None = None,
    months_a: list[int] | None = None,
    months_b: list[int] | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    across: Literal["building", "cohort"] = "building",
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Compare one building across two periods: years, seasons, weekday/weekend.

    ``comparison="year_year"`` answers "did X use more electricity in 2017 than
    in 2016": it returns a column per year plus their ratio and percentage
    difference, so the comparison is in the result rather than left to the
    reader.

    ``comparison="named_days"`` with ``day_a`` and ``day_b`` contrasts two days
    of the week -- "how does Monday compare to Sunday". weekday_weekend cannot
    answer that: it pools Monday with the rest of the working week and Sunday
    with Saturday, and the question named two days.

    ``comparison="named_months"`` with ``months_a`` and ``months_b`` contrasts
    two stretches of the calendar that no season names -- the fall semester,
    September to December, against the spring semester, January to May. Give
    ``label_a`` and ``label_b`` to have the arms named in the result.

    ``measure="peak_hour"`` compares *when* demand peaks rather than how much
    of it there is, and reports the shift in hours rather than a ratio: "how
    does peak timing move between the semesters" is answered by 1, not by 1.08.
    The peak hour of an arm is the median of its daily peak hours, one per
    building-day. Taking the hour of the single largest reading in a term
    instead reads one hot afternoon off four months.

    A day's peak hour is the *earliest* hour at which that day's maximum is
    reached. About one building-day in ten reaches its maximum in more than one
    hour, and the choice is not cosmetic: for the education cohort the spring
    median is 12:00 taking the earliest and 13:00 taking the latest, which is
    the whole of the one-hour shift being measured. The earliest is when demand
    first reaches its high, and it is what the snapshot's own
    ``agg_daily.peak_hour_local`` records.

    ``measure="profile_shape"`` answers "does the daily profile change" rather
    than "is there more of it": each arm gets its 24-hour mean profile divided
    by its own mean, and the result is the Pearson correlation between the two
    shapes. Near 1 says the shape holds and only the level moved; the ratio of
    means cannot tell those apart, because a building that doubles its load
    without changing its rhythm has a ratio of 2 and a correlation of 1.

    ``across="cohort"`` pools every reading of the cohort into one comparison,
    which is what "weekday against weekend for site Fox" asks: the site has 137
    buildings and the per-building form answers with 137 ratios, leaving the
    site-level number to be worked out by hand.

    Contrast two temporal regimes: weekday/weekend, seasons, years, day/night,
    or two named days."""
    raw = load_series(ctx, building_ids=building_ids, start=start, end=end)
    if raw.empty:
        return empty("no measurements in the requested window")
    parts = with_local_parts(raw)
    ts = pd.to_datetime(parts["timestamp"])

    if comparison == "named_days":
        if not (day_a and day_b):
            return insufficient(
                "named_days",
                "comparison='named_days' needs day_a and day_b, for instance "
                "monday and sunday")
        weekday = ts.dt.day_name().str.lower()
        parts = parts[weekday.isin([day_a.lower(), day_b.lower()])]
        parts["group"] = weekday[parts.index]
    elif comparison == "weekday_weekend":
        parts["group"] = np.where(parts["is_weekend"], "weekend", "weekday")
    elif comparison == "summer_winter":
        parts["season"] = season_of(parts["month"])
        parts = parts[parts["season"].isin(["summer", "winter"])]
        parts["group"] = parts["season"]
    elif comparison == "named_months":
        if not (months_a and months_b):
            return insufficient(
                "named_months",
                "comparison='named_months' needs months_a and months_b, for "
                "instance [9, 10, 11, 12] and [1, 2, 3, 4, 5]")
        in_a = [int(m) for m in months_a]
        in_b = [int(m) for m in months_b]
        a_name = label_a or _month_span(in_a)
        b_name = label_b or _month_span(in_b)
        parts["group"] = np.where(parts["month"].isin(in_a), a_name,
                                  np.where(parts["month"].isin(in_b), b_name, ""))
        parts = parts[parts["group"] != ""]
    elif comparison == "year_year":
        parts["group"] = ts.dt.year.astype(str)
    else:  # day_night -- 07:00-19:00 local counts as day
        parts["group"] = np.where(parts["hour"].between(7, 18), "day", "night")

    if parts.empty:
        return empty(f"no data falls into either arm of {comparison}")
    if measure == "profile_shape":
        # The shape is the profile divided by its own mean, so two arms are
        # compared on rhythm rather than on size.
        if across == "cohort":
            parts = parts.assign(building_id="all buildings")
        hourly = (parts.groupby(["building_id", "group", "hour"])["value"]
                       .mean().reset_index())
        shapes = []
        for building, grp in hourly.groupby("building_id"):
            wide = grp.pivot(index="hour", columns="group", values="value")
            arms = [c for c in wide.columns]
            if len(arms) != 2 or wide.dropna().shape[0] < 3:
                continue
            paired = wide.dropna()
            normed = paired / paired.mean()
            row = {"building_id": building}
            for arm in arms:
                row[arm] = float(paired[arm].mean())
            row["shape_correlation"] = float(
                normed[arms[0]].corr(normed[arms[1]]))
            row["hours_compared"] = int(len(paired))
            shapes.append(row)
        if not shapes:
            return empty("neither arm has a full enough profile to compare")
        wide = pd.DataFrame(shapes)
        groups = [c for c in wide.columns
                  if c not in ("building_id", "shape_correlation", "hours_compared")]
    elif measure == "peak_hour":
        # One peak hour per building-day, then the typical one in each arm.
        # The daily peak has to be found per real building before a cohort is
        # pooled: relabelling first asks what hour the whole estate peaked in
        # on a given day, which is a fact about the largest building.
        usable = parts.dropna(subset=["value"])
        first_peak = usable.groupby(["building_id", "date", "group"])["value"].idxmax()
        peaks = usable.loc[first_peak, ["building_id", "group", "hour"]]
        if across == "cohort":
            peaks = peaks.assign(building_id="all buildings")
        agg = (peaks.groupby(["building_id", "group"])["hour"]
                    .agg(peak_hour="median", n_days="count").reset_index())
        wide = agg.pivot(index="building_id", columns="group",
                         values="peak_hour").reset_index()
    else:
        if across == "cohort":
            parts = parts.assign(building_id="all buildings")
        agg = (parts.groupby(["building_id", "group"])["value"]
                    .agg(mean_demand="mean", total="sum", n_observations="count")
                    .reset_index())
        wide = agg.pivot(index="building_id", columns="group",
                         values="mean_demand").reset_index()
    groups = [c for c in wide.columns if c != "building_id"]
    if comparison == "named_days" and len(groups) == 2:
        # The ratio is the one the question asked for -- Monday over Sunday --
        # not whichever of the two sorts first.
        groups = [day_a.lower(), day_b.lower()]
    if comparison == "named_months" and len(groups) == 2:
        named = [label_a or _month_span([int(m) for m in months_a]),
                 label_b or _month_span([int(m) for m in months_b])]
        if set(named) == set(groups):
            groups = named
    if measure == "profile_shape":
        pass                      # shape_correlation is already the comparison
    elif len(groups) == 2 and measure == "peak_hour":
        a, b = groups
        # Hours apart, not a ratio: 13:00 against 12:00 is a one-hour shift,
        # and 1.08 is not an answer to "how does the timing move".
        wide["shift_hours"] = wide[a] - wide[b]
    elif len(groups) == 2:
        a, b = groups
        wide["ratio"] = wide[a] / wide[b].replace(0, np.nan)
        wide["pct_difference"] = 100 * (wide[a] - wide[b]) / wide[b].replace(0, np.nan)
    # The comparison is the point of the call, so say so: the row holds each
    # arm's mean beside the ratio, and "weekday against weekend for site Fox"
    # was answered with the weekday mean, 157.36, where the ratio is 1.11.
    headline = None
    if len(wide) == 1 and measure == "profile_shape" \
            and pd.notna(wide["shape_correlation"].iloc[0]):
        headline = {"label": f"how alike the {groups[0]} and {groups[1]} daily "
                             f"profiles are in shape",
                    "value": round(float(wide["shape_correlation"].iloc[0]), 6),
                    "unit": "correlation"}
    elif len(wide) == 1 and len(groups) == 2 and measure == "peak_hour" \
            and pd.notna(wide["shift_hours"].iloc[0]):
        headline = {"label": f"hours the peak moves from {groups[1]} "
                             f"to {groups[0]}",
                    "value": round(float(wide["shift_hours"].iloc[0]), 6),
                    "unit": "hours"}
    elif len(wide) == 1 and len(groups) == 2 and pd.notna(wide["ratio"].iloc[0]):
        headline = {"label": f"{groups[0]} demand over {groups[1]} demand",
                    "value": round(float(wide["ratio"].iloc[0]), 6),
                    "unit": None}
    # The curves this comparison was made over, published beside the summary.
    # Every one of these comparisons reduces two daily profiles to a ratio, a
    # correlation or an hour, and the profiles themselves were discarded: a
    # question that asks to *compare the weekday and weekend load profile* is
    # asking to see them, and there was no frame anywhere in the run with an
    # hour axis to draw. Charting the summary drew nothing, so the answer
    # arrived as a ratio with neither a curve nor the table behind it.
    curves = None
    try:
        shape_parts = (parts.assign(building_id="all buildings")
                       if across == "cohort" else parts)
        curves = (shape_parts.groupby(["building_id", "group", "hour"])["value"]
                  .mean().reset_index()
                  .rename(columns={"value": "mean_kw", "group": "period"})
                  .sort_values(["building_id", "period", "hour"]))
    except Exception:
        # The profile is support, never the answer: a comparison that cannot
        # produce one still answers the question it was asked.
        curves = None
    evidence = {**({"headline": headline} if headline else {}),
                "comparison": comparison, "groups": groups,
                "measure": measure, "across": across,
                "ratio_is": f"{groups[0]} / {groups[1]}"
                            if len(groups) == 2
                            and measure != "peak_hour" else None}
    if curves is not None and not curves.empty:
        evidence["profile_handle"] = ctx.store.put(
            curves, name=f"profiles_{comparison}")
        evidence["profile_columns"] = {"x": "hour", "y": "mean_kw",
                                       "group": "period"}
        evidence["profile_rows"] = int(len(curves))
    return frame_result(wide, ctx.store, name=f"compare_{comparison}",
                        evidence=evidence)


def _hours_between(first: int, last: int) -> list[int]:
    """Every hour of the day in a window, the last hour included.

    A night usually crosses midnight -- 10 PM to 6 AM is hours 22, 23, 0 to 5
    -- so the window wraps rather than running out of range.
    """
    first, last = int(first) % 24, int(last) % 24
    if first <= last:
        return list(range(first, last + 1))
    return list(range(first, 24)) + list(range(0, last + 1))


#: The occupied core of the day, for the night-to-day contrast.
DAY_START_HOUR, DAY_LAST_HOUR = 9, 17


@tool("consumption")
def calculate_baseload(
    ctx: ToolContext,
    building_ids: list[str],
    method: Literal["night_mean", "percentile", "daily_min_mean"] = "night_mean",
    percentile: float = 0.05,
    across: Literal["building", "cohort"] = "building",
    night_start_hour: int | None = None,
    night_last_hour: int | None = None,
    decile_at_least: int | None = None,
    select: Literal["all", "unusual_high", "unusual_low"] = "all",
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Estimate baseload, plus its share of mean demand.

    night_start_hour and night_last_hour name the first and last hour of the
    window, in local time, and the last hour is part of it: 0 and 5 is
    00:00-05:59. "Midnight to 4 AM" ends before 4 AM, so it is
    night_last_hour=3 -- written as 4 it runs to 04:59 and takes in an hour the
    question excluded. The name says "last" rather than "end" because an end
    reads as the boundary you stop at, and that reading is wrong here.

    The three methods are three estimators of one quantity, the floor a
    building never drops below. percentile and daily_min_mean infer it from the
    shape of the demand because the unoccupied hours are not known; night_mean
    is for when they are. So naming a window says which estimator applies, and
    a window with method="percentile" is taken as night_mean over that window,
    reported in the evidence. The composition it asks for -- the 90th
    percentile of midnight to 4 AM -- is a peak inside the night rather than a
    floor, and is not a baseload at all.

    Left unset the window is 00:00-05:59 for night_mean and the other two
    estimators read every hour, as they must. The window matters: 00:00-03:59
    and 00:00-05:59 each put 29 office buildings in the top decile of baseload
    ratio, and only 21 are the same buildings.

    across says whose baseload is wanted. "The baseload of the healthcare
    facilities" is one figure over their pooled readings, across="cohort";
    "which offices hold a top-decile ratio" needs one row each,
    across="building". Pooling is not the average of the per-building
    figures -- it is the estimator applied to every reading at once, and a
    large building weighs more in it, which is what "the cohort's floor"
    means.

    decile_at_least keeps the buildings whose baseload_ratio falls in that
    decile or above, ranked within the cohort passed in, so 10 is the top
    tenth.

    Note that BDG2's cleaning nulls every zero, so a genuinely idle meter
    reads as missing rather than zero. Baseload here is therefore computed over
    observed readings only and is an upper bound for such buildings.
    """
    # Baseload scans raw hourly readings, so a repeat is worth avoiding. Serve
    # it from the knowledge store when every column of the result is there
    #; anything less falls through to the scan.
    wanted = {c: f"baseload:{method}:{c}"
              for c in ("baseload", "baseload_ratio", "mean_demand",
                        "n_observations")}
    # The cached figures were computed for the default night, and the cache key
    # does not carry the window: serving them for a question that named
    # midnight to 4 AM would answer a different question with a warm result.
    window = (None if night_start_hour is None and night_last_hour is None
              else (0 if night_start_hour is None else int(night_start_hour),
                    5 if night_last_hour is None else int(night_last_hour)))
    # A named window is the definition of the floor, so it settles the
    # estimator too. Planners reach for percentile=0.9 on "top-decile baseload
    # ratio", conflating the rank bucket with a percentile of demand, and that
    # measures each building's 90th-busiest night hour.
    overridden = None if window is None or method == "night_mean" else method
    if overridden:
        method = "night_mean"
    default_night = window in (None, (0, 5))
    cached, why = (try_cached(ctx, "PerformanceScore", wanted, building_ids,
                             sort_by="baseload", ascending=False,
                             params={"method": method, "start": start, "end": end,
                                     "percentile": percentile})
                   if default_night
                   else (None, {"cache": "skipped",
                                "reason": "a night window other than the "
                                          "default is not what was cached"}))
    if cached is not None:
        cached["n_observations"] = cached["n_observations"].astype(int)
        return frame_result(cached, ctx.store, name="baseload",
                            evidence={"method": method, **why})

    raw = load_series(ctx, building_ids=building_ids, start=start, end=end)
    if raw.empty:
        return empty("no measurements in the requested window")
    parts = with_local_parts(raw)
    rows = []
    # The night, for the estimator that reads one. Settled once, outside the
    # loop, because the evidence below reports it too.
    hours = window or ((0, 5) if method == "night_mean" else None)
    grouped = (parts.groupby("building_id") if across == "building"
               else [("all buildings", parts)])
    _walk, _cov = budgeted(ctx, grouped)
    for building, grp in _walk:
        values = grp["value"].dropna()
        if values.empty:
            continue
        # The ratio's denominator is the building's mean over every hour in the
        # requested period, whatever window the numerator was taken from: the
        # question is what share of an average hour runs overnight.
        mean_demand = float(values.mean())
        inside = (grp if hours is None
                  else grp.loc[grp["hour"].isin(_hours_between(*hours))])
        sample = inside["value"].dropna()
        if sample.empty:
            base = np.nan
        elif method == "night_mean":
            base = float(sample.mean())
        elif method == "percentile":
            base = float(sample.quantile(percentile))
        else:
            base = float(inside.groupby("date")["value"].min().mean())
        # Night against DAY, as well as night against the overall mean. The
        # overall mean contains the night hours themselves, so that ratio is
        # compressed towards 1 -- at Bull it spans 0.59 to 1.18 and no
        # building clears an IQR fence, which makes it useless for deciding
        # who is unusual. Night over day separates the same buildings clearly.
        # "Day" here is the occupied core, 09:00-17:59, not simply every hour
        # that is not night: the evening shoulder sits between the two levels
        # and averaging it into the denominator pulls the ratio back towards
        # the middle, which is what stopped any building at Bull clearing the
        # fence when the whole non-night clock was used.
        daytime = grp.loc[grp["hour"].isin(range(DAY_START_HOUR,
                                                 DAY_LAST_HOUR + 1))]
        day = daytime["value"].dropna()
        day_mean = float(day.mean()) if not day.empty else np.nan
        rows.append({"building_id": building, "baseload": base,
                     "mean_demand": mean_demand,
                     "day_mean": day_mean,
                     "baseload_ratio": base / mean_demand if mean_demand else np.nan,
                     "night_day_ratio": (base / day_mean
                                         if day_mean and day_mean == day_mean
                                         else np.nan),
                     "n_observations": int(values.size)})
    if not rows:
        return empty("no building had usable readings")
    out = pd.DataFrame(rows)
    kept = None
    if decile_at_least is not None and len(out) >= 10:
        # Deciles as rank buckets over the cohort, the way the size deciles
        # work: "top-decile baseload ratio among offices" is the tenth bucket
        # of the offices, not a ratio above some absolute line.
        ranked = out.dropna(subset=["baseload_ratio"]).sort_values("baseload_ratio")
        # NTILE's own division, not qcut's: with 296 offices the ten buckets
        # are six of thirty and four of twenty-nine, the remainder going to the
        # earlier ones, so the top decile holds 29. qcut put 30 there and the
        # extra building is a real building either way -- the boundary has to
        # be the same boundary the reference drew.
        total, buckets = len(ranked), 10
        larger, size = total % buckets, total // buckets
        def _bucket(position: int) -> int:
            edge = larger * (size + 1)
            if position < edge:
                return position // (size + 1) + 1
            return larger + (position - edge) // size + 1
        ranked["decile"] = [_bucket(i) for i in range(total)]
        kept = int(decile_at_least)
        out = (ranked[ranked["decile"] >= kept]
               .sort_values("baseload_ratio", ascending=False)
               .reset_index(drop=True))
    # "Which buildings have UNUSUALLY high night consumption" is not a request
    # for 123 rows of ratio, and it is not a top-N either: how many buildings
    # are unusual is the thing being asked. The cohort's own upper IQR fence
    # answers it -- a fixed ratio would call every data centre an anomaly and
    # a fixed count would return ten whether or not any stand out.
    fence = None
    ratio_column = ("night_day_ratio"
                    if "night_day_ratio" in out
                    and out["night_day_ratio"].notna().sum() >= 4
                    else "baseload_ratio")
    if select != "all" and len(out) >= 4:
        ratios = out[ratio_column].astype(float).dropna()
        if len(ratios) >= 4:
            q1, q3 = ratios.quantile(0.25), ratios.quantile(0.75)
            span = 1.5 * (q3 - q1)
            if select == "unusual_high":
                fence = float(q3 + span)
                out = out[out[ratio_column] > fence]
                out = out.sort_values(ratio_column, ascending=False)
            else:
                fence = float(q1 - span)
                out = out[out[ratio_column] < fence]
                out = out.sort_values(ratio_column)
            out = out.reset_index(drop=True)
    return frame_result(out, ctx.store, name="baseload",
                        evidence={"method": method,
                                  "method_asked_for": overridden,
                                  "selection": None if select == "all" else {
                                      "rule": f"{ratio_column} beyond the cohort's "
                                              f"{'upper' if select == 'unusual_high' else 'lower'} "
                                              f"IQR fence",
                                      "fence": None if fence is None else round(fence, 4),
                                      "cohort_ranked": len(rows),
                                      "selected": len(out)},
                                  "decile_at_least": kept,
                                  "buildings_ranked": len(rows) if kept else None,
                                  "percentile": percentile if method == "percentile" else None,
                                  "hours": (
                                      "every hour" if hours is None else
                                      f"{hours[0]:02d}:00-{hours[1]:02d}:59 local"),
                                  "caveat": "masked zeros are excluded; baseload is an "
                                            "upper bound where meters idle at zero"})


@tool("consumption")
def analyze_operating_pattern(
    ctx: ToolContext,
    building_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    threshold: float = 0.25,
    days: Literal["all", "weekday", "weekend"] = "all",
    business_start_hour: int | None = None,
    business_last_hour: int | None = None,
    business_days: Literal["all", "weekday", "weekend"] = "weekday",
    report: Literal["schedule", "start", "end", "after_hours"] = "schedule",
) -> ToolResult:
    """Infer start-up, shutdown, after-hours share and morning ramp.

    An hour counts as operating when its mean demand exceeds the overnight
    floor -- the mean of 00:00 to 04:59, when an office is least likely to be
    occupied -- by ``threshold`` of the day's range. This is an operational
    signature, not an occupancy claim.

    The baseline is the floor rather than the single lowest hour because one
    anomalously low hour drags the minimum down and inflates the range, and
    because a schedule inferred from consumption is read against the load a
    building carries when nobody is there.

    A quarter of the range rather than a half: a half misses the shoulder
    hours when a building is occupied but not yet at full load, and for
    Cockatoo_office_Alton it puts the day at 14 hours where a quarter gives
    17, ending at 18:00 rather than 21:00.

    ``days="weekday"`` reads the schedule off Monday to Friday only. A
    building that opens at 07:00 on Tuesday does not open at all on Sunday,
    and pooling the week averages the two into a ramp that happens on neither.

    ``business_start_hour`` and ``business_last_hour`` state the working day
    instead of inferring it, and ``after_hours_share`` is then the share of
    consumption falling outside it. "7am to 7pm" is 7 and 18: the last hour is
    part of the window and 7pm is where it ends. ``business_days`` says which
    days the window applies on and defaults to weekdays, so the whole weekend
    counts as after hours -- which is the difference between 52.8% and 45.0%
    for Bear_education_Alfredo, whose inferred schedule opens at 08:00 and
    runs every day of the week.

    ``business_days`` is not ``days``. ``days`` narrows what the tool looks at;
    the business window narrows what counts as open inside it. An after-hours
    share has to see the weekend to count it.

    ``report`` says which figure the caller wants named as the answer:
    "schedule" the length of the day, "start" the hour it opens, "end" the
    hour it closes, "after_hours" the percentage drawn outside hours. Note
    that ``ramp_hour`` is a different thing again -- the hour demand climbs
    fastest, which for Fox_office_Alice is 04:00 while the building does not
    cross its occupied threshold until 06:00. "When does it start ramping up"
    is start_hour, the first hour above the floor, not the steepest one.
    """
    raw = load_series(ctx, building_ids=building_ids, start=start, end=end)
    if raw.empty:
        return empty("no measurements in the requested window")
    parts = with_local_parts(raw)
    if days == "weekday":
        parts = parts[~parts["is_weekend"]]
    elif days == "weekend":
        parts = parts[parts["is_weekend"]]
    if parts.empty:
        return empty(f"no {days} readings in the requested window")
    rows = []
    window = (None if business_start_hour is None and business_last_hour is None
              else (0 if business_start_hour is None else int(business_start_hour),
                    23 if business_last_hour is None else int(business_last_hour)))
    _walk, _cov = budgeted(ctx, parts.groupby("building_id"))
    for building, grp in _walk:
        hourly = grp.groupby("hour")["value"].mean()
        if hourly.dropna().empty:
            continue
        lo, hi = float(hourly.min()), float(hourly.max())
        if not np.isfinite(lo) or hi <= lo:
            continue
        overnight = hourly[hourly.index.isin(_hours_between(0, 4))].dropna()
        floor = float(overnight.mean()) if not overnight.empty else lo
        # "Exceeds" is what the docstring has always said; the code tested
        # >=, which counts an hour sitting exactly on the cut as open.
        active = hourly[hourly > floor + threshold * (hi - lo)].index.tolist()
        weekday = grp[~grp["is_weekend"]]["value"].mean()
        weekend = grp[grp["is_weekend"]]["value"].mean()
        if window is None:
            outside = ~grp["hour"].isin(active)
        else:
            # A stated working day: inside it means the right hours *and* the
            # right days, so everything else -- evenings, nights, the whole
            # weekend -- is after hours.
            inside = grp["hour"].isin(_hours_between(*window))
            if business_days == "weekday":
                inside &= ~grp["is_weekend"]
            elif business_days == "weekend":
                inside &= grp["is_weekend"]
            outside = ~inside
        after = grp.loc[outside, "value"].sum()
        total = grp["value"].sum()
        ramp = float(hourly.diff().max()) if len(hourly) > 1 else np.nan
        rows.append({
            "building_id": building,
            "peak_hour": int(hourly.idxmax()),
            "peak_mean_demand": float(hi),
            "start_hour": int(min(active)) if active else None,
            "end_hour": int(max(active)) if active else None,
            "operating_hours": len(active),
            "morning_ramp_kw_per_h": ramp,
            "ramp_hour": int(hourly.diff().idxmax()) if len(hourly) > 1 else None,
            "after_hours_share": float(after / total) if total else np.nan,
            "after_hours_share_pct": float(100 * after / total) if total else np.nan,
            "weekend_weekday_ratio": float(weekend / weekday) if weekday else np.nan,
        })
    if not rows:
        return empty("no building had a usable daily shape")
    # Ten columns describe the schedule; one of them is the answer. Asked what
    # a building's apparent schedule is, a run that reports the row reports
    # everything except an answer, so say which figure it is -- but only for
    # one building, where "the schedule" names one thing.
    named = {"schedule": ("operating_hours",
                          "hours a day the building is apparently open", "hours"),
             "start": ("start_hour",
                       "hour the building's demand starts rising", "hour of day"),
             "end": ("end_hour",
                     "hour the building's demand falls back", "hour of day"),
             "after_hours": ("after_hours_share_pct",
                             "share of consumption drawn outside hours",
                             "%")}[report]
    column, label, unit = named
    figure = rows[0].get(column) if len(rows) == 1 else None
    headline = ({"label": label,
                 "value": (round(float(figure), 3) if unit == "%" else int(figure)),
                 "unit": unit}
                if figure is not None and not pd.isna(figure) else None)
    return frame_result(pd.DataFrame(rows), ctx.store, name="operating_pattern",
                        evidence={**({"headline": headline} if headline else {}),
                                  "threshold": threshold, "days": days,
                                  "reported": column,
                                  "business_hours": (
                                      None if window is None else
                                      f"{window[0]:02d}:00-{window[1]:02d}:59 "
                                      f"local, {business_days}"),
                                  "definition": "operating hour = mean demand above "
                                                "the 00:00-04:59 floor by more than "
                                                f"{threshold:.0%} of the daily range",
                                  "timezone": "local building time"})


@tool("consumption")
def analyze_consumption_trend(
    ctx: ToolContext,
    building_ids: list[str],
    grain: Literal["monthly", "yearly"] = "monthly",
    min_coverage: float = 0.9,
) -> ToolResult:
    """Long-run trend: slope, total change and period-over-period comparison.

    Periods below ``min_coverage`` are dropped rather than compared against
    complete ones.
    """
    view = {"monthly": "agg_monthly", "yearly": "agg_yearly"}[grain]
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    params.append(min_coverage)
    df = ctx.db.df(f"""
        SELECT building_id, period_start, total_kwh, coverage_fraction
        FROM {view} WHERE {where} AND coverage_fraction >= ?
        ORDER BY building_id, period_start
    """, params)
    if df.empty:
        return empty(f"no {grain} periods meet coverage >= {min_coverage}")
    rows = []
    _walk, _cov = budgeted(ctx, df.groupby("building_id"))
    for building, grp in _walk:
        grp = grp.sort_values("period_start")
        y = grp["total_kwh"].to_numpy(dtype=float)
        if len(y) < 3:
            continue
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "building_id": building, "periods": len(y),
            "first_period_kwh": float(y[0]), "last_period_kwh": float(y[-1]),
            "slope_kwh_per_period": float(slope),
            "pct_change_first_to_last": float(100 * (y[-1] - y[0]) / y[0]) if y[0] else np.nan,
            "mean_kwh": float(y.mean()),
        })
    if not rows:
        return empty(f"no building has 3+ {grain} periods above the coverage floor")
    return frame_result(pd.DataFrame(rows), ctx.store, name=f"trend_{grain}",
                        evidence={"grain": grain, "min_coverage": min_coverage,
                                  "method": "ordinary least squares on period totals"})


@tool("consumption")
def count_step_changes(
    ctx: ToolContext,
    building_ids: list[str],
    grain: Literal["daily", "weekly", "monthly"] = "daily",
    change_at_least_pct: float = 20.0,
    direction: Literal["increase", "decrease", "either"] = "increase",
    start: str | None = None,
    end: str | None = None,
    # Complete days only. A partial day's total is low for a reason that has
    # nothing to do with consumption, and it inflates the *next* day's
    # percentage change: on Cockatoo_office_Alton the 15-hour day of
    # 2017-07-19 makes 07-20 look like a rise. Counting only days with all 24
    # hourly readings -- and only against the calendar day before them --
    # gives 54 where keeping partial days gives 65 and bridging gaps as well
    # gives 66.
    min_coverage: float = 1.0,
) -> ToolResult:
    """How often consumption jumped from one period to the next, and when.

    "How many times did daily consumption increase by more than 20% compared
    with the previous day" is a count of ordinary period-over-period moves,
    and none of the tools that sound close answer it. detect_change_points
    finds the single persistent regime shift in a series -- at most one per
    building, by construction -- and detect_anomalies with first_difference
    scores a change in standard deviations of the building's own changes,
    which is a different threshold on a different scale: a 20% rise is 0.8
    sigma in one building and 4 sigma in another. Asked for a count of 20%
    days, both return a number, and neither returns this one.

    A period below ``min_coverage`` is dropped rather than compared against a
    complete one -- a day with six hours of readings is not a 75% fall -- and
    dropping it also breaks the adjacency, so the day after is not compared
    across the hole either. That second half is the one it is easy to leave
    out, and leaving it out does not look like a bug: the count simply comes
    out higher. For Cockatoo_office_Alton over 2016-17 it is the difference
    between 57 and 59, the two extra being 2017-01-25 against 2017-01-22 and
    2017-07-20 against 2017-07-15 -- three and five days apart, each counted
    as "the previous day".
    """
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    if start:
        where += " AND period_start >= ?"; params.append(start)
    if end:
        where += " AND period_start < ?"; params.append(end)
    params.append(min_coverage)
    if grain == "weekly":
        # No weekly aggregate exists and none is needed: a week is exactly
        # seven days, so it rolls up from the daily one without loss and
        # cannot drift from it. Coverage is the week's observed hours over
        # its 168, which is the same rule the stored grains use -- a week
        # missing two whole days is 71% covered, not complete.
        df = ctx.db.df(f"""
            SELECT building_id, DATE_TRUNC('week', period_start) AS period_start,
                   SUM(total_kwh) AS total_kwh
            FROM agg_daily WHERE {where} AND grain = 'daily'
            GROUP BY 1, 2
            HAVING SUM(n_observations) >= ? * 168
            ORDER BY 1, 2
        """, params)
    else:
        view = {"daily": "agg_daily", "monthly": "agg_monthly"}[grain]
        df = ctx.db.df(f"""
            SELECT building_id, period_start, total_kwh
            FROM {view} WHERE {where} AND coverage_fraction >= ?
            ORDER BY building_id, period_start
        """, params)
    if df.empty:
        return empty(f"no {grain} periods meet coverage >= {min_coverage}")
    step = {"daily": pd.Timedelta(days=1),
            "weekly": pd.Timedelta(days=7)}.get(grain)
    rows, counts = [], []
    _walk, _cov = budgeted(ctx, df.groupby("building_id"))
    for building, grp in _walk:
        grp = grp.sort_values("period_start").reset_index(drop=True)
        previous = grp["total_kwh"].shift(1)
        change = 100.0 * (grp["total_kwh"] - previous) / previous.replace(0, np.nan)
        stamps = pd.to_datetime(grp["period_start"])
        # Two periods either side of a dropped one are not adjacent, and the
        # move between them is not a period-over-period move.
        adjacent = (stamps.diff() == step) if step is not None else (
            stamps.dt.to_period("M").astype(int).diff() == 1)
        if direction == "increase":
            hit = change > change_at_least_pct
        elif direction == "decrease":
            hit = change < -change_at_least_pct
        else:
            hit = change.abs() > change_at_least_pct
        hit = hit & adjacent & change.notna()
        counts.append({"building_id": building, "n_changes": int(hit.sum()),
                       "n_periods_compared": int((adjacent & change.notna()).sum())})
        for position in grp.index[hit]:
            rows.append({"building_id": building,
                         "period_start": str(stamps[position])[:10],
                         "previous_kwh": float(previous[position]),
                         "total_kwh": float(grp["total_kwh"][position]),
                         "change_pct": float(change[position])})
    if not counts:
        return insufficient(
            "no_periods",
            f"no building has two adjacent {grain} periods above the "
            f"coverage floor")
    word = {"increase": "rose", "decrease": "fell", "either": "moved"}[direction]
    noun = {"daily": "day", "weekly": "week", "monthly": "month"}[grain]
    evidence = {"grain": grain, "direction": direction,
                "threshold_pct": change_at_least_pct,
                "min_coverage": min_coverage,
                "definition": f"a {noun} whose total {word} more than "
                              f"{change_at_least_pct:g}% against the period "
                              f"immediately before it",
                "n_changes": int(sum(c["n_changes"] for c in counts)),
                "per_building": counts}
    if not rows:
        return ok({"changes": [], "n_changes": 0}, evidence=evidence)
    return frame_result(pd.DataFrame(rows), ctx.store, name="step_changes",
                        evidence=evidence)
