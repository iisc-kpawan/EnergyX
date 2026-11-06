"""Weather, Climate & Spatio-Temporal Analytics tools (7).

Covers task categories 6 and 8.

Weather is joined at site level and always in local building time -- comparing
peak timing across Tempe and London in UTC would be meaningless. Air temperature
is complete in this snapshot (100%), but ``cloud_coverage`` (48%) and
``precip_depth_6hr`` (6%) are sparse, so tools report the sample they actually
used rather than assuming coverage.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..knowledge.cache import try_cached
from ._common import in_clause
from .base import (Status, ToolResult, empty, frame_result,
                   insufficient)
from .registry import ToolContext, tool

WEATHER_VARS = ("air_temperature", "dew_temperature", "wind_speed",
                "cloud_coverage", "sea_level_pressure", "precip_depth_1hr")
MIN_POINTS = 100
# Buildings per join. Small enough that a deadline can interrupt between
# batches, large enough that the per-query overhead stays negligible.
JOIN_BATCH = 250


#: Time-of-use controls a question can ask the weather relationship to hold
#: constant. Derived from the joined timestamp, so they cost no extra query.
CONTROL_TERMS = {
    "hour_of_day": lambda ts: ts.dt.hour,
    "day_of_week": lambda ts: ts.dt.dayofweek,
    "month": lambda ts: ts.dt.month,
}


def _design(frame: pd.DataFrame, controls: list[str]) -> np.ndarray:
    """Intercept, then one dummy column per control level bar the first.

    Dropping one level per control is what keeps the design from being
    singular; with every level present the intercept is a linear combination
    of them and the fit is not identified.
    """
    columns = [np.ones(len(frame))]
    stamps = pd.to_datetime(frame["ts"])
    for name in controls:
        levels = CONTROL_TERMS[name](stamps)
        for value in sorted(levels.unique())[1:]:
            columns.append((levels == value).to_numpy(float))
    return np.column_stack(columns)


def _partial_temperature_fit(frame: pd.DataFrame, controls: list[str]) -> dict:
    """Weather slope with the named time terms held constant.

    A raw temperature correlation on hourly data mostly measures the working
    day: consumption and temperature both rise in the morning, so a building
    with no cooling at all still correlates. Regressing both on the time terms
    first and relating what is left is the Frisch-Waugh-Lovell construction --
    the slope is identical to the multivariate one, and the correlation of the
    residuals is the partial correlation, which is the quantity "after
    controlling for hour-of-day and day-of-week" names.
    """
    y = frame["value"].to_numpy(float)
    x = frame["wvalue"].to_numpy(float)
    design = _design(frame, controls)
    # Residualise both sides against the controls, then relate the residuals.
    coef_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    coef_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    ry = y - design @ coef_y
    rx = x - design @ coef_x
    denom = float((rx ** 2).sum())
    if denom <= 0:
        return {"slope": float("nan"), "partial_correlation": float("nan"),
                "r_squared": float("nan"), "partial_r_squared": float("nan")}
    slope = float((rx * ry).sum() / denom)
    partial_r = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_controls = float((ry ** 2).sum())
    ss_full = float(((ry - slope * rx) ** 2).sum())
    return {
        "slope": slope,
        "partial_correlation": partial_r,
        # What the controls alone explain, and what temperature adds on top.
        "r_squared": 1 - ss_full / ss_tot if ss_tot else float("nan"),
        "controls_r_squared": 1 - ss_controls / ss_tot if ss_tot else float("nan"),
        "partial_r_squared": ((ss_controls - ss_full) / ss_controls
                              if ss_controls > 0 else float("nan")),
    }


def _joined(ctx: ToolContext, building_ids: list[str], variable: str,
            start: str | None, end: str | None, grain: str = "daily") -> pd.DataFrame:
    """Electricity joined to a site weather variable, at hourly or daily grain."""
    params: list = []
    where = "1=1" + in_clause("e.building_id", building_ids, params)
    if start:
        where += " AND e.timestamp_local >= ?"; params.append(start)
    if end:
        where += " AND e.timestamp_local < ?"; params.append(end)
    bucket = ("e.timestamp_local" if grain == "hourly"
              else "CAST(e.timestamp_local AS DATE)")
    wbucket = ("w.timestamp_local" if grain == "hourly"
               else "CAST(w.timestamp_local AS DATE)")
    agg = "SUM(e.value)" if grain == "daily" else "AVG(e.value)"
    return ctx.db.df(f"""
        WITH elec AS (
            SELECT e.building_id, e.site_id, {bucket} AS ts, {agg} AS value
            FROM electricity e WHERE {where}
            GROUP BY 1, 2, 3
        ), wx AS (
            SELECT w.site_id, {wbucket} AS ts, AVG(w.{variable}) AS wvalue
            FROM weather w GROUP BY 1, 2
        )
        SELECT elec.building_id, elec.site_id, elec.ts, elec.value, wx.wvalue
        FROM elec JOIN wx ON wx.site_id = elec.site_id AND wx.ts = elec.ts
        WHERE elec.value IS NOT NULL AND wx.wvalue IS NOT NULL
        ORDER BY elec.building_id, elec.ts
    """, params or None)


def _weather_in_top_hours(ctx: ToolContext, building_ids: list[str],
                          variables: list[str], top_share_pct: float,
                          months: list[int] | None, start: str | None,
                          end: str | None) -> ToolResult:
    """Weather over the hours a building was at or above its demand cutoff."""
    cutoff = 1.0 - float(top_share_pct) / 100.0
    # Placeholders one per value rather than unnest: DuckDB will not bind an
    # unnest inside the CTE that carries the percentile window.
    params: list = [cutoff, *list(building_ids)]
    where = ("e.quality_flag = 'OK' AND e.building_id IN ("
             + ", ".join("?" for _ in building_ids) + ")")
    filters = ""
    if months:
        filters += (" AND EXTRACT(month FROM e.timestamp_local) IN ("
                    + ", ".join("?" for _ in months) + ")")
        params += [int(m) for m in months]
    if start:
        filters += " AND e.timestamp_local >= ?"; params.append(start)
    if end:
        filters += " AND e.timestamp_local < ?"; params.append(end)
    selects = ", ".join(
        f"AVG(w.{v}) AS mean_{v}, MIN(w.{v}) AS min_{v}, "
        f"MAX(w.{v}) AS max_{v}, MEDIAN(w.{v}) AS median_{v}" for v in variables)
    frame = ctx.db.df(f"""
        WITH h AS (
            SELECT e.building_id, e.site_id, e.timestamp_utc, e.value,
                   QUANTILE_CONT(e.value, ?) OVER (PARTITION BY e.building_id)
                       AS cutoff_value
            FROM electricity e WHERE {where}{filters})
        SELECT h.building_id, COUNT(*) AS hours, {selects}
        FROM h JOIN weather w
          ON w.site_id = h.site_id AND w.timestamp_utc = h.timestamp_utc
        WHERE h.value >= h.cutoff_value
        GROUP BY 1 ORDER BY 1
    """, params)
    if frame.empty:
        return empty("no weather observations in those hours")
    headline = None
    if len(frame) == 1 and f"mean_{variables[0]}" in frame.columns:
        headline = {"label": f"mean {variables[0].replace('_', ' ')} during the "
                             f"top {top_share_pct:g}% of demand hours",
                    "value": round(float(frame[f"mean_{variables[0]}"].iloc[0]), 4),
                    "unit": "degC" if "temperature" in variables[0] else None}
    return frame_result(frame, ctx.store, name="weather_in_top_hours",
                        evidence={**({"headline": headline} if headline else {}),
                                  "variables": variables,
                                  "during": f"hours at or above the "
                                            f"{100 - top_share_pct:g}th demand "
                                            f"percentile",
                                  "months": months, "period": [start, end]})


@tool("weather_spatial")
def calculate_weather_statistics(
    ctx: ToolContext,
    site_ids: list[str] | None = None,
    variables: list[Literal["air_temperature", "dew_temperature", "wind_speed",
                            "wind_direction", "cloud_coverage",
                            "precip_depth_1hr", "sea_level_pressure"]] | None = None,
    building_ids: list[str] | None = None,
    during: Literal["all_hours", "top_demand_hours"] = "all_hours",
    top_share_pct: float = 5.0,
    months: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Mean, min, max and median of a weather variable, per site.

    This answers "what was the average air temperature at site X" and "what was
    the highest wind speed recorded at site Y". fetch_weather returns the raw
    series for a later step; this aggregates it, so a weather statistic is not
    answered from electricity.

    ``during="top_demand_hours"`` answers the other half of that pairing: what
    the weather was doing while a building was at its busiest. The hours are
    the building's own -- those at or above its (100 - top_share_pct) demand
    percentile in the window asked about -- so "the average temperature during
    the top 5% of demand hours" is one call rather than a join the plan has to
    assemble. At or above a threshold, not a bucket holding exactly one
    twentieth of the rows: with ties at the boundary the two differ, and a
    threshold is what the phrase means.
    """
    chosen = [v for v in (variables or ["air_temperature"])]
    if during == "top_demand_hours":
        if not building_ids:
            return insufficient(
                "building_ids",
                "the busiest hours are a building's own; name the building")
        return _weather_in_top_hours(ctx, building_ids, chosen, top_share_pct,
                                     months, start, end)
    params: list = []
    where = "1=1"
    # A cohort of buildings names the sites whose weather is wanted. This
    # argument was accepted and then used only by the top-demand-hours branch,
    # so "the highest wind speed at site Bull", planned as find_buildings then
    # this tool with those buildings, returned one row for each of the
    # nineteen sites with Bull among them -- the filter silently discarded and
    # the answer nineteen times wider than the question.
    if not site_ids and building_ids:
        resolved = ctx.db.df(
            "SELECT DISTINCT site_id FROM buildings "
            "WHERE building_id IN (SELECT unnest(?)) AND site_id IS NOT NULL",
            [list(building_ids)])
        site_ids = resolved["site_id"].tolist() if not resolved.empty else None
        if not site_ids:
            return empty("none of those buildings resolve to a site with "
                         "weather observations")
    if site_ids:
        params.append(list(site_ids))
        where += " AND site_id IN (SELECT unnest(?))"
    if months:
        where += " AND EXTRACT(month FROM timestamp_local) IN (SELECT unnest(?))"
        params.append([int(m) for m in months])
    if start:
        where += " AND timestamp_local >= ?"; params.append(start)
    if end:
        where += " AND timestamp_local < ?"; params.append(end)
    selects = []
    for v in chosen:
        selects += [f"AVG({v}) AS mean_{v}", f"MIN({v}) AS min_{v}",
                    f"MAX({v}) AS max_{v}", f"MEDIAN({v}) AS median_{v}",
                    f"COUNT({v}) AS n_{v}"]
    df = ctx.db.df(f"""
        SELECT site_id, {', '.join(selects)}
        FROM weather WHERE {where} GROUP BY site_id ORDER BY site_id
    """, params or None)
    if df.empty:
        return empty("no weather observations in the requested window")
    return frame_result(df, ctx.store, name="weather_statistics",
                        evidence={"variables": chosen, "period": [start, end],
                                  "sites": len(df),
                                  "site_ids": list(site_ids) if site_ids else None,
                                  "unit": "degC" if "temperature" in chosen[0]
                                          else None})


def _hourly_contrast(ctx: ToolContext, building_ids: list[str], variable: str,
                     start: str | None, end: str | None,
                     basis: str = "median") -> ToolResult:
    """Mean hourly consumption in cool hours against warm ones.

    Distinct from the daily contrast beside it, which asks whether a *day*
    whose peak passed a line used more than the rest. "How much does mean
    hourly consumption differ between cool and warm hours" pairs each hour
    with the temperature at that hour, which is a different split and a
    different number: the daily form cannot express it because a day is warm
    or cool as a whole.

    The line comes from the data -- the median of the site's own hours over
    the period -- because a fixed temperature is cool in one climate and warm
    in another.
    """
    joined = _joined(ctx, building_ids, variable, start, end, "hourly")
    if joined.empty:
        return insufficient("no_overlap",
                            f"no overlapping electricity and {variable} hours")
    rows = []
    for building, grp in joined.groupby("building_id"):
        temps = pd.to_numeric(grp["wvalue"], errors="coerce")
        use = pd.to_numeric(grp["value"], errors="coerce")
        ok = temps.notna() & use.notna()
        temps, use = temps[ok], use[ok]
        if len(temps) < MIN_POINTS:
            continue
        if basis == "tercile":
            low_cut, high_cut = temps.quantile(1 / 3), temps.quantile(2 / 3)
        else:
            low_cut = high_cut = temps.median()
        # A median split is a partition: an hour sitting exactly on the line
        # belongs to one side, not both. Counting it twice put 4,393 hours in
        # the warm half where 4,245 belong, and 8,906 hours in a year of
        # 8,758. Terciles are a genuine gap -- the middle third is in neither
        # half by design -- so only the median case closes at the boundary.
        if basis == "tercile":
            cool, warm = use[temps <= low_cut], use[temps >= high_cut]
        else:
            cool, warm = use[temps <= low_cut], use[temps > high_cut]
        if cool.empty or warm.empty:
            continue
        cool_mean, warm_mean = float(cool.mean()), float(warm.mean())
        rows.append({
            "building_id": building,
            "split_at": round(float(low_cut), 3) if basis != "tercile" else None,
            "cool_below": round(float(low_cut), 3),
            "warm_above": round(float(high_cut), 3),
            "n_cool_hours": int(len(cool)), "n_warm_hours": int(len(warm)),
            "cool_mean_kwh": round(cool_mean, 4),
            "warm_mean_kwh": round(warm_mean, 4),
            "difference_kwh": round(warm_mean - cool_mean, 4),
            "difference_pct": (round(100.0 * (warm_mean - cool_mean) / cool_mean, 3)
                               if cool_mean else None),
        })
    if not rows:
        return insufficient("too_few_points",
                            f"no building has {MIN_POINTS}+ paired hours")
    frame = pd.DataFrame(rows).sort_values("difference_kwh", ascending=False)
    top = frame.iloc[0]
    headline = {
        "label": (f"{top['building_id']}: mean hourly use "
                  f"{top['cool_mean_kwh']:,.2f} kWh in cool hours vs "
                  f"{top['warm_mean_kwh']:,.2f} in warm "
                  f"({top['difference_kwh']:+,.2f} kWh, "
                  f"{top['difference_pct']:+.1f}%)"),
        "value": float(top["difference_kwh"]), "unit": "kWh"}
    return frame_result(frame, ctx.store, name="hourly_temperature_contrast",
                        evidence={"headline": headline, "variable": variable,
                                  "basis": basis, "grain": "hourly",
                                  "period": [start, end],
                                  "buildings": int(len(frame))})


def _threshold_contrast(ctx: ToolContext, building_ids: list[str], variable: str,
                        threshold: float, months: list[int] | None,
                        start: str | None, end: str | None) -> ToolResult:
    """Daily consumption on days the weather passed a line, against the rest."""
    if not building_ids:
        return insufficient("building_ids", "name the building to contrast")
    # Parameters in the order the statement binds them: the weather CTE first,
    # then the four threshold comparisons, then the consumption filters.
    weather_params: list = []
    main_params: list = list(building_ids)
    where = "a.grain = 'daily' AND a.building_id IN (" + ", ".join(
        "?" for _ in building_ids) + ")"
    wwhere = "1 = 1"
    if months:
        placeholders = ", ".join("?" for _ in months)
        wwhere += f" AND EXTRACT(month FROM w.timestamp_local) IN ({placeholders})"
        weather_params += [int(m) for m in months]
        where += f" AND EXTRACT(month FROM a.period_start) IN ({placeholders})"
        main_params += [int(m) for m in months]
    if start:
        where += " AND a.period_start >= ?"; main_params.append(start)
    if end:
        where += " AND a.period_start < ?"; main_params.append(end)
    frame = ctx.db.df(f"""
        WITH d AS (
            SELECT w.site_id, CAST(w.timestamp_local AS DATE) AS day,
                   MAX(w.{variable}) AS peak_value
            FROM weather w WHERE {wwhere} GROUP BY 1, 2)
        SELECT a.building_id,
               COUNT(*) FILTER (WHERE d.peak_value > ?)  AS days_above,
               COUNT(*) FILTER (WHERE d.peak_value <= ?) AS days_below,
               AVG(a.total_kwh) FILTER (WHERE d.peak_value > ?)  AS mean_kwh_above,
               AVG(a.total_kwh) FILTER (WHERE d.peak_value <= ?) AS mean_kwh_below
        FROM agg_daily a
        JOIN d ON d.site_id = a.site_id AND d.day = CAST(a.period_start AS DATE)
        WHERE {where}
        GROUP BY 1 ORDER BY 1
    """, [*weather_params, threshold, threshold, threshold, threshold,
          *main_params])
    if frame.empty:
        return empty("no days matched the window")
    frame["ratio_above_to_below"] = (frame["mean_kwh_above"]
                                     / frame["mean_kwh_below"].replace(0, np.nan))
    headline = None
    if len(frame) == 1 and frame["ratio_above_to_below"].notna().iloc[0]:
        headline = {"label": f"mean daily consumption on days above "
                             f"{threshold:g} against the other days",
                    "value": round(float(frame["ratio_above_to_below"].iloc[0]), 4),
                    "unit": "ratio"}
    above = int(frame["days_above"].sum())
    return frame_result(frame, ctx.store, name="threshold_contrast",
                        evidence={**({"headline": headline} if headline else {}),
                                  "variable": variable, "threshold": threshold,
                                  "days_above": above,
                                  "months": months, "period": [start, end],
                                  # A contrast with nothing on one side is not
                                  # a contrast, and saying so is the answer.
                                  "caveat": (None if above else
                                             f"no day reached {threshold:g}")})


@tool("weather_spatial")
def analyze_weather_relationship(
    ctx: ToolContext,
    building_ids: list[str],
    variable: Literal["air_temperature", "dew_temperature", "wind_speed",
                      "cloud_coverage"] = "air_temperature",
    # "correlation" is Pearson, which is what the word means unqualified here.
    # "spearman" ranks both series first: it answers whether consumption moves
    # monotonically with the weather rather than linearly, which is the right
    # question for a variable like dew point whose effect saturates. The two
    # differ enough to change which buildings clear a 0.75 threshold.
    method: Literal["correlation", "spearman", "regression",
                    "threshold_contrast"] = "correlation",
    controls: list[Literal["hour_of_day", "day_of_week", "month"]] | None = None,
    grain: Literal["hourly", "daily"] = "daily",
    threshold: float | Literal["median", "tercile"] | None = None,
    months: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Strength and direction of the weather-electricity relationship.

    ``method="threshold_contrast"`` answers "did consumption spike during the
    heatwave": mean daily consumption on the days whose peak weather value
    passed ``threshold``, against the other days in the same window, and the
    ratio between them. A correlation cannot answer that question -- it
    measures the whole range, and a building that only responds above 35C has
    a weak one -- and a plan that fetches both series leaves the contrast to
    be computed by a reader.
    """
    if method == "threshold_contrast":
        if threshold is None:
            return insufficient("threshold",
                                "a contrast needs the line it contrasts across")
        # "Cool hours versus warm hours" names a split without naming a
        # temperature, and the line that answers it comes from the data: the
        # median of the site's own hours over the period asked about. A fixed
        # number cannot serve, because 20 C is a cool hour in Austin and a warm
        # one in Cardiff.
        if isinstance(threshold, str):
            observed = _joined(ctx, building_ids, variable, start, end, "hourly")
            if observed.empty:
                return insufficient(
                    "no_overlap",
                    f"no overlapping electricity and {variable} observations to "
                    f"take a {threshold} from")
            return _hourly_contrast(ctx, building_ids, variable, start, end,
                                    basis=threshold)
        return _threshold_contrast(ctx, building_ids, variable, float(threshold),
                                   months, start, end)
    if variable not in WEATHER_VARS:
        return insufficient("unknown_variable", f"valid: {sorted(WEATHER_VARS)}")

    # Serve from the knowledge layer when every requested building already has
    # this exact quantity computed.
    # The computed path always ranks by correlation, so the cache must carry it
    # even when the caller asked for a regression.
    wanted = {"correlation": f"weather_sensitivity:{variable}:correlation"}
    if method == "regression":
        wanted["slope"] = f"weather_sensitivity:{variable}:slope"
    # Every parameter that changes the number belongs in the key. It held
    # only the variable and the method, so an hourly correlation over 2017 was
    # served a cached daily one over the whole record -- 0.3905 where the
    # answer is 0.4557 -- and the `computed_at` column was the only sign.
    # Controls make it a different quantity again, so a controlled fit is
    # never served from a cache of uncontrolled ones.
    cached, why = (None, {}) if controls else try_cached(
        ctx, "PerformanceScore", wanted, building_ids,
        sort_by="correlation", ascending=False,
        params={"variable": variable, "method": method, "grain": grain,
                "start": start, "end": end, "months": tuple(months or ())})
    if cached is not None:
        return frame_result(cached, ctx.store, name="weather_relationship",
                            evidence={"variable": variable, "method": method,
                                      **why})

    # Join in batches so a large cohort can report what it finished rather than
    # running past its budget in one unbounded query. A portfolio-wide request
    # was taking over eight minutes as a single join.
    frames, covered, ran_out = [], 0, False
    for offset in range(0, len(building_ids), JOIN_BATCH):
        if ctx.out_of_time and frames:
            ran_out = True
            break
        batch = list(building_ids)[offset:offset + JOIN_BATCH]
        part = _joined(ctx, batch, variable, start, end, grain)
        covered += len(batch)
        if not part.empty:
            frames.append(part)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return insufficient("no_overlap",
                            f"no overlapping electricity and {variable} observations")
    rows = []
    for building, grp in df.groupby("building_id"):
        if len(grp) < MIN_POINTS:
            continue
        x = grp["wvalue"].to_numpy(float)
        y = grp["value"].to_numpy(float)
        if method == "spearman":
            # Ranks, then the ordinary correlation of the ranks. Ties take the
            # average rank, which is what scipy.stats.spearmanr does and what
            # a reference implementation will have used.
            x = pd.Series(x).rank(method="average").to_numpy(float)
            y = pd.Series(y).rank(method="average").to_numpy(float)
        r = float(np.corrcoef(x, y)[0, 1])
        entry = {"building_id": building, "n_points": int(len(grp)),
                 "correlation": r, "correlation_kind": (
                     "spearman" if method == "spearman" else "pearson")}
        if method == "regression" and controls:
            # "After controlling for hour-of-day and day-of-week" is a
            # different quantity from the raw slope, and on hourly data it is
            # usually a much smaller one: without the controls the fit is
            # partly measuring the working day.
            entry.update(_partial_temperature_fit(grp, controls))
            entry["correlation"] = entry.get("partial_correlation", r)
            entry["raw_correlation"] = r
        elif method == "regression":
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            entry.update(slope=float(slope), intercept=float(intercept),
                         r_squared=1 - ss_res / ss_tot if ss_tot else np.nan)
        rows.append(entry)
    if not rows:
        return insufficient("too_few_points",
                            f"no building has {MIN_POINTS}+ paired observations")
    out = pd.DataFrame(rows)
    # "Strongest sensitivity" is a magnitude: a building that draws hard as it
    # gets colder is as weather-driven as one that draws hard as it warms, and
    # ranking on the signed figure buried every heating-led building below
    # every cooling-led one. With controls the share of variance temperature
    # explains is what "strongest" means, which is the partial R-squared.
    if "partial_r_squared" in out.columns:
        out = out.sort_values("partial_r_squared", ascending=False)
    else:
        out = out.assign(_m=out["correlation"].abs()).sort_values(
            "_m", ascending=False).drop(columns="_m")
    # Say when the cohort was not finished, and how much of it was. A partial
    # answer that does not admit it is worse than a slow one.
    evidence = {"variable": variable, "method": method,
                "controls": list(controls or []),
                "grain": grain, "min_points": MIN_POINTS, **why,
                "buildings_requested": len(building_ids),
                "buildings_covered": covered,
                "slope_unit": f"kWh per unit {variable}"
                              if method == "regression" else None}
    result = frame_result(out, ctx.store, name="weather_relationship",
                          evidence=evidence)
    if ran_out:
        evidence["truncated_by_time"] = True
        result.status = Status.PARTIAL
        result.message = (f"time budget reached after {covered} of "
                          f"{len(building_ids)} buildings")
    return result


def _changepoint(x: np.ndarray, y: np.ndarray, mode: str) -> dict:
    """Grid-search a piecewise-linear balance temperature (3-parameter model).

    Whole degrees. The response surface is flat near its minimum, so a finer
    grid moves the answer without improving the fit, and a balance point
    claimed to a hundredth of a degree is precision the data does not carry.
    """
    span = np.quantile(x, [0.05, 0.95])
    candidates = np.arange(np.floor(span[0]), np.ceil(span[1]) + 1, 1.0)
    if candidates.size == 0:
        candidates = np.quantile(x, np.linspace(0.05, 0.95, 37))
    best = None
    for cp in candidates:
        if mode == "cooling":
            feature = np.maximum(0.0, x - cp)
        elif mode == "heating":
            feature = np.maximum(0.0, cp - x)
        else:
            feature = np.column_stack([np.maximum(0.0, cp - x),
                                       np.maximum(0.0, x - cp)])
        design = np.column_stack([np.ones_like(x), feature])
        coef, residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        sse = float(((y - pred) ** 2).sum())
        if best is None or sse < best["sse"]:
            sst = float(((y - y.mean()) ** 2).sum())
            best = {"change_point": float(cp), "baseline": float(coef[0]),
                    "sse": sse, "r_squared": 1 - sse / sst if sst else np.nan,
                    "slopes": [float(c) for c in np.atleast_1d(coef[1:])]}
    return best or {}


@tool("weather_spatial")
def fit_temperature_response(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    model: Literal["cooling", "heating", "both"] = "cooling",
    across: Literal["building", "site"] = "building",
    site_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Fit a change-point temperature-response model per building, or per site.

    Returns the balance temperature and the sensitivity slope -- the quantities
    the building-energy literature calls the cooling/heating change point.

    ``across="site"`` fits one model to the site's pooled daily consumption,
    which is what "the cooling change-point temperature for site Robin" asks
    for. Fitting each of its buildings and handing back the list answers a
    different question: a site has one balance point, and the median of its
    buildings' is not it.
    """
    if across == "site" and not (site_ids or building_ids):
        return insufficient("site_ids", "name the site to fit")
    cached, why = try_cached(ctx, "ChangePoint",
                             {"change_point_c": f"piecewise_changepoint:{model}"},
                             building_ids or [], params={"model": model})
    if across == "site":
        cached = None
    if cached is not None:
        return frame_result(cached, ctx.store, name="temperature_response",
                            evidence={"model": model, **why})

    if across == "site":
        params: list = []
        where = "1=1"
        if site_ids:
            where += " AND a.site_id IN (SELECT unnest(?))"; params.append(list(site_ids))
        elif building_ids:
            where += " AND a.building_id IN (SELECT unnest(?))"
            params.append(list(building_ids))
        if start:
            where += " AND a.period_start >= ?"; params.append(start)
        if end:
            where += " AND a.period_start < ?"; params.append(end)
        df = ctx.db.df(f"""
            WITH w AS (SELECT site_id, CAST(timestamp_local AS DATE) AS day,
                              AVG(air_temperature) AS wvalue
                       FROM weather WHERE air_temperature IS NOT NULL
                       GROUP BY 1, 2)
            SELECT a.site_id AS building_id, CAST(a.period_start AS DATE) AS ts,
                   SUM(a.total_kwh) AS value, ANY_VALUE(w.wvalue) AS wvalue
            FROM agg_daily a
            JOIN w ON w.site_id = a.site_id
                  AND w.day = CAST(a.period_start AS DATE)
            WHERE a.grain = 'daily' AND {where}
            GROUP BY 1, 2""", params or None)
    else:
        df = _joined(ctx, building_ids or [], "air_temperature", start, end, "daily")
    if df.empty:
        return insufficient("no_overlap",
                            "no overlapping electricity and temperature observations")
    rows = []
    for building, grp in df.groupby("building_id"):
        if len(grp) < MIN_POINTS:
            continue
        fit = _changepoint(grp["wvalue"].to_numpy(float),
                           grp["value"].to_numpy(float), model)
        if not fit:
            continue
        entry = {("site_id" if across == "site" else "building_id"): building,
                 "n_days": int(len(grp)),
                 "change_point_c": fit["change_point"],
                 "baseline_kwh_per_day": fit["baseline"],
                 "r_squared": fit["r_squared"]}
        if model == "both":
            entry["heating_slope"], entry["cooling_slope"] = fit["slopes"][:2]
        else:
            entry[f"{model}_slope_kwh_per_degc"] = fit["slopes"][0]
        rows.append(entry)
    if not rows:
        return insufficient("too_few_days",
                            f"no building has {MIN_POINTS}+ paired days")
    out = pd.DataFrame(rows)
    sort_col = ("cooling_slope" if model == "both"
                else f"{model}_slope_kwh_per_degc")
    # One fit is one number: the balance point is what the question asks for,
    # and a one-row table leaves it to be read off.
    headline = ({"label": f"{model} change-point temperature",
                 "value": float(out["change_point_c"].iloc[0]), "unit": "degC"}
                if len(out) == 1 else None)
    return frame_result(out.sort_values(sort_col, ascending=False), ctx.store,
                        name="temperature_response",
                        evidence={**({"headline": headline} if headline else {}),
                                  "across": across,
                                  "model": f"3-parameter change-point ({model})",
                                  "grain": "daily totals vs daily mean temperature",
                                  **why,
                                  "search": "37 candidate balance points over the "
                                            "5th-95th temperature percentiles"})


@tool("weather_spatial")
def weather_normalize_consumption(
    ctx: ToolContext,
    building_ids: list[str],
    baseline_start: str,
    baseline_end: str,
    target_start: str,
    target_end: str,
    model: Literal["cooling", "heating", "both"] = "both",
) -> ToolResult:
    """Compare actual target-period use against what baseline behaviour predicts.

    Fits a change-point model on the baseline period, applies it to the target
    period's weather, and reports the residual. A positive residual means the
    building used more than its own historical weather response explains.
    """
    base = _joined(ctx, building_ids, "air_temperature", baseline_start,
                   baseline_end, "daily")
    target = _joined(ctx, building_ids, "air_temperature", target_start,
                     target_end, "daily")
    if base.empty or target.empty:
        return insufficient("no_overlap",
                            "baseline or target period has no paired observations")
    rows = []
    for building, bgrp in base.groupby("building_id"):
        tgrp = target[target["building_id"] == building]
        if len(bgrp) < 30 or len(tgrp) < 7:
            continue
        x = bgrp["wvalue"].to_numpy(float)
        fit = _changepoint(x, bgrp["value"].to_numpy(float), model)
        if not fit:
            continue
        cp = fit["change_point"]
        tx = tgrp["wvalue"].to_numpy(float)
        if model == "cooling":
            feature = np.maximum(0.0, tx - cp)[:, None]
        elif model == "heating":
            feature = np.maximum(0.0, cp - tx)[:, None]
        else:
            feature = np.column_stack([np.maximum(0.0, cp - tx),
                                       np.maximum(0.0, tx - cp)])
        predicted = fit["baseline"] + feature @ np.array(fit["slopes"])
        actual_total = float(tgrp["value"].sum())
        predicted_total = float(predicted.sum())
        rows.append({
            "building_id": building,
            "baseline_days": int(len(bgrp)), "target_days": int(len(tgrp)),
            "change_point_c": cp,
            "actual_kwh": actual_total,
            "weather_normalized_expected_kwh": predicted_total,
            "residual_kwh": actual_total - predicted_total,
            "residual_pct": (100 * (actual_total - predicted_total) / predicted_total
                             if predicted_total else np.nan),
            "baseline_r_squared": fit["r_squared"],
        })
    if not rows:
        return insufficient("too_few_days",
                            "no building has 30+ baseline and 7+ target days")
    out = pd.DataFrame(rows).sort_values("residual_pct", ascending=False)
    return frame_result(out, ctx.store, name="weather_normalized",
                        evidence={"baseline": [baseline_start, baseline_end],
                                  "target": [target_start, target_end],
                                  "model": model,
                                  "interpretation": "positive residual = used more "
                                                    "than its own weather response "
                                                    "predicts"})


@tool("weather_spatial")
def aggregate_geographically(
    ctx: ToolContext,
    level: Literal["site", "city", "country"] = "city",
    metric: Literal["eui", "total_kwh", "mean_demand", "peak_demand"] = "eui",
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
    year: int | None = None,
    min_coverage: float | None = None,
    require_capability: bool = True,
) -> ToolResult:
    """Roll a metric up to site, city or country level.

    A rollup of a *total* is a sum and a rollup of an intensity is a middle.
    "Which city has the highest aggregate total consumption" is answered by
    the sum over its buildings; ordering by the median instead put a city of
    ten large buildings above one with a hundred, which is a question about
    the typical building and not the one that was asked. Both are reported,
    and the one the metric calls for decides the order.

    ``min_coverage`` defaults to 0.9 for an intensity, whose denominator is a
    year the building has to have had, and to 0 for a total, where a floor
    removes population members rather than unfit ones. Coverage is of the
    period rather than of its worst month, as it is everywhere else: a
    building 97% complete over a year was being dropped for one month at 71%.

    ``require_capability`` applies the same ``can_eui`` predicate that
    ``find_buildings(requires=["eui_analysis"])`` applies upstream. Without it a
    building with a well-covered target year but a poorly-covered snapshot
    period enters a geographic rollup while being excluded from a per-building
    ranking of the same metric -- an inconsistency that silently changes which
    city comes top. The predicate used is always reported in the evidence.
    """
    key = {"site": "s.site_id", "city": "s.city", "country": "s.country_canonical"}[level]
    sums = metric in ("total_kwh",)
    if min_coverage is None:
        min_coverage = 0.0 if sums else 0.9
    capability_join = ""
    predicate_parts = [f"{key} IS NOT NULL",
                       f"period coverage >= {min_coverage}"]
    if require_capability and metric in ("eui",):
        capability_join = "JOIN capabilities c ON c.building_id = a.building_id"
        predicate_parts.append("can_eui")
    params: list = []
    where = f"{key} IS NOT NULL"
    if capability_join:
        where += " AND c.can_eui"
    if building_type:
        where += " AND b.building_type_canonical = ?"; params.append(building_type)
    if year:
        where += " AND EXTRACT(year FROM a.period_start) = ?"; params.append(year)
    params.append(min_coverage)
    value = {"eui": "SUM(a.total_kwh) / ANY_VALUE(b.floor_area_m2)",
             "total_kwh": "SUM(a.total_kwh)",
             "mean_demand": "AVG(a.mean_demand)",
             "peak_demand": "MAX(a.peak_demand)"}[metric]
    # Coverage of the period, not of its worst month -- the same rule the
    # performance frame uses, for the same reason.
    expected = ("8784.0" if year in (2016, 2020, 2024)
                else "8760.0" if year
                else "SUM(a.n_observations / NULLIF(a.coverage_fraction, 0))")
    order_by = "total_value" if sums else "median_value"
    df = ctx.db.df(f"""
        WITH per_building AS (
            SELECT {key} AS geo, a.building_id, {value} AS metric_value
            FROM agg_monthly a
            JOIN buildings b ON b.building_id = a.building_id
            JOIN sites s ON s.site_id = b.site_id
            {capability_join}
            WHERE {where}
            GROUP BY 1, 2
            HAVING SUM(a.n_observations) / NULLIF({expected}, 0) >= ?
        )
        SELECT geo AS {level}, COUNT(*) AS buildings,
               ROUND(SUM(metric_value), 2)    AS total_value,
               ROUND(MEDIAN(metric_value), 2) AS median_value,
               ROUND(AVG(metric_value), 2)    AS mean_value,
               ROUND(QUANTILE_CONT(metric_value, 0.25), 2) AS p25,
               ROUND(QUANTILE_CONT(metric_value, 0.75), 2) AS p75
        FROM per_building WHERE metric_value IS NOT NULL
        GROUP BY 1 ORDER BY {order_by} DESC
    """, params)
    if df.empty:
        return empty(f"no {level} has scorable buildings for {metric}")
    headline = {"label": f"{level} with the highest "
                         f"{'total' if sums else 'median'} "
                         f"{metric.replace('_', ' ')}",
                "value": str(df[level].iloc[0]), "unit": None}
    return frame_result(df, ctx.store, name=f"geo_{level}_{metric}",
                        evidence={"headline": headline,
                                  "level": level, "metric": metric,
                                  "ordered_by": order_by,
                                  "building_type": building_type, "year": year,
                                  "min_coverage": min_coverage,
                                  "predicate": " AND ".join(predicate_parts)})


@tool("weather_spatial")
def compare_geographies(
    ctx: ToolContext,
    level: Literal["site", "city", "country"],
    values: list[str],
    metric: Literal["eui", "total_kwh", "mean_demand", "peak_demand"] = "eui",
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
    year: int | None = None,
) -> ToolResult:
    """Compare named sites, cities or countries on one metric."""
    result = aggregate_geographically(ctx, level=level, metric=metric,
                                      building_type=building_type, year=year)
    if not result.ok:
        return result
    df = pd.DataFrame(result.data) if result.data else ctx.store.get(result.handle)
    subset = df[df[level].isin(values)]
    if subset.empty:
        return empty(f"none of {values} present at {level} level")
    missing = sorted(set(values) - set(subset[level]))
    evidence = {**result.evidence, "requested": values, "missing": missing}
    return frame_result(subset, ctx.store, name=f"compare_{level}", evidence=evidence)


@tool("weather_spatial")
def analyze_spatiotemporal_patterns(
    ctx: ToolContext,
    level: Literal["site", "city", "country"] = "city",
    pattern: Literal["peak_timing", "seasonality", "load_profile",
                     "weather_response"] = "peak_timing",
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
) -> ToolResult:
    """How load behaviour varies across geography, in local time."""
    key = {"site": "s.site_id", "city": "s.city", "country": "s.country_canonical"}[level]
    params: list = []
    filt = f"{key} IS NOT NULL"
    if building_type:
        filt += " AND b.building_type_canonical = ?"; params.append(building_type)

    if pattern == "peak_timing":
        df = ctx.db.df(f"""
            SELECT {key} AS geo, COUNT(DISTINCT p.building_id) AS buildings,
                   ARG_MAX(p.hour_of_day, p.mean_demand) AS peak_hour_local
            FROM profile_daily p
            JOIN buildings b ON b.building_id = p.building_id
            JOIN sites s ON s.site_id = b.site_id
            WHERE {filt} GROUP BY 1 ORDER BY 1
        """, params or None)
    elif pattern == "load_profile":
        df = ctx.db.df(f"""
            SELECT {key} AS geo, p.hour_of_day,
                   ROUND(AVG(p.mean_demand), 2) AS mean_demand,
                   COUNT(DISTINCT p.building_id) AS buildings
            FROM profile_daily p
            JOIN buildings b ON b.building_id = p.building_id
            JOIN sites s ON s.site_id = b.site_id
            WHERE {filt} GROUP BY 1, 2 ORDER BY 1, 2
        """, params or None)
    elif pattern == "seasonality":
        df = ctx.db.df(f"""
            SELECT {key} AS geo, EXTRACT(month FROM a.period_start)::INT AS month,
                   ROUND(AVG(a.mean_demand), 2) AS mean_demand,
                   COUNT(DISTINCT a.building_id) AS buildings
            FROM agg_monthly a
            JOIN buildings b ON b.building_id = a.building_id
            JOIN sites s ON s.site_id = b.site_id
            WHERE {filt} AND a.coverage_fraction >= 0.9
            GROUP BY 1, 2 ORDER BY 1, 2
        """, params or None)
    else:  # weather_response
        df = ctx.db.df(f"""
            WITH daily AS (
                SELECT {key} AS geo, e.building_id,
                       CAST(e.timestamp_local AS DATE) AS d,
                       SUM(e.value) AS kwh
                FROM electricity e
                JOIN buildings b ON b.building_id = e.building_id
                JOIN sites s ON s.site_id = b.site_id
                WHERE e.quality_flag = 'OK' AND {filt} GROUP BY 1, 2, 3
            ), wx AS (
                SELECT s2.site_id, {key.replace('s.', 's2.')} AS geo,
                       CAST(w.timestamp_local AS DATE) AS d,
                       AVG(w.air_temperature) AS t
                FROM weather w JOIN sites s2 ON s2.site_id = w.site_id
                GROUP BY 1, 2, 3
            )
            SELECT daily.geo, COUNT(*) AS observations,
                   ROUND(CORR(daily.kwh, wx.t), 3) AS temp_correlation
            FROM daily JOIN wx ON wx.geo = daily.geo AND wx.d = daily.d
            WHERE daily.kwh IS NOT NULL AND wx.t IS NOT NULL
            GROUP BY 1 ORDER BY temp_correlation DESC
        """, params or None)

    if df is None or df.empty:
        return empty(f"no {pattern} pattern computable at {level} level")
    return frame_result(df.rename(columns={"geo": level}), ctx.store,
                        name=f"spatiotemporal_{pattern}",
                        evidence={"level": level, "pattern": pattern,
                                  "building_type": building_type,
                                  "timezone": "local building time"})


@tool("weather_spatial")
def prepare_geospatial_analysis(
    ctx: ToolContext,
    level: Literal["site", "city", "country"] = "site",
    metric: Literal["eui", "total_kwh", "mean_demand"] = "eui",
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
    year: int | None = None,
) -> ToolResult:
    """Coordinates plus metric values, ready for mapping.

    Buildings in anonymous sites carry no geography and are excluded here -- by
    predicate, not silently.
    """
    rolled = aggregate_geographically(ctx, level=level, metric=metric,
                                      building_type=building_type, year=year)
    if not rolled.ok:
        return rolled
    df = pd.DataFrame(rolled.data) if rolled.data else ctx.store.get(rolled.handle)
    coord_key = {"site": "site_id", "city": "city", "country": "country_canonical"}[level]
    coords = ctx.db.df(f"""
        SELECT {coord_key} AS geo, AVG(latitude) AS latitude,
               AVG(longitude) AS longitude
        FROM sites WHERE {coord_key} IS NOT NULL AND latitude IS NOT NULL
        GROUP BY 1
    """)
    merged = df.merge(coords, left_on=level, right_on="geo", how="left").drop(
        columns=["geo"], errors="ignore")
    without = int(merged["latitude"].isna().sum())
    return frame_result(merged, ctx.store, name=f"geospatial_{level}",
                        evidence={**rolled.evidence,
                                  "rows_without_coordinates": without,
                                  "note": "anonymous sites have no coordinates"})
