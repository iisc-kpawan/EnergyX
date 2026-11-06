"""Building Performance & Peer Benchmarking tools (6).

Covers task categories 5 and 7.

Two things are deliberately explicit here because they are the usual source of
irreproducible benchmarking results: the **eligibility predicate** that defines
who is in the cohort, and the **outlier rule**. Median 2017 electricity EUI in
this snapshot is plausible per type (Office 112, Education 102 kWh/m2/yr), but
4.1% of eligible buildings exceed 500 kWh/m2/yr and the maximum is 4,360 -- so a
naive "top 5 by EUI" returns data-quality artefacts unless outliers are handled.
Every tool that ranks therefore exposes ``exclude_outliers_above``.
"""
from __future__ import annotations

import calendar
from typing import Literal

import numpy as np
import pandas as pd

from ._common import STRUCTURAL, in_clause
from .base import (PREVIEW_ROWS, Status, ToolResult, empty, frame_result,
                   insufficient, ok)
from .registry import ToolContext, tool

METRICS = {
    "eui": ("total_kwh / floor_area_m2", "kWh/m2"),
    "peak_per_area": ("peak_demand / floor_area_m2", "kW/m2"),
    # The same quantity in the unit the field actually writes it in. Peak
    # power density is quoted in W/m2 -- a lighting allowance is 8 W/m2, a
    # data hall 1000 -- and kW/m2 puts every building between 0.01 and 0.2,
    # where a threshold of 50 matches nothing and looks like an empty answer
    # rather than a unit error. The two differ by exactly 1000.
    "peak_power_density": ("1000 * peak_demand / floor_area_m2", "W/m2"),
    # What a square metre draws per hour the building is actually in use, as
    # opposed to per hour on the calendar. Two buildings with the same EUI are
    # not alike if one runs eight hours a day and the other sixteen.
    #
    # The divisor is inferred, not recorded, and the inference travels with
    # the number: a building that never turns off reads as operating 24 hours
    # and its intensity falls accordingly, which is a statement about the
    # meter rather than about the floorspace. operating_hours_per_day is
    # reported beside the quotient so the reader can see what it rests on.
    "eui_per_operating_hour": (
        "total_kwh / floor_area_m2 / (operating_hours_per_day * 365)",
        "kWh/m2 per operating hour"),
    "per_occupant": ("total_kwh / occupants", "kWh/person"),
    "total_kwh": ("total_kwh", "kWh"),
    "mean_demand": ("mean_demand", "kW"),
    "peak_demand": ("peak_demand", "kW"),
    # How flat a building runs, and how spiky. Neither needs floor area or
    # occupancy, so they compare cohorts that have no metadata in common --
    # which is what "compare education with office" asks when it asks about
    # shape rather than size.
    "load_factor": ("mean_demand / peak_demand", "ratio"),
    "peak_to_average_ratio": ("peak_demand / mean_demand", "ratio"),
    "baseload_per_area": ("percentile of hourly demand / floor_area_m2",
                          "kW/m2"),
    # The overnight floor per square metre. Distinct from baseload_per_area,
    # which is a low percentile of the whole day: this is the level between
    # midnight and 05:00, which is what "normalised overnight baseload" names,
    # and it is quoted in watts because that is how power density is read.
    "daily_load_factor": ("mean over peak within each day, averaged across "
                          "days", "ratio"),
    "seasonal_cv": ("coefficient of variation of a building's twelve monthly "
                    "mean demands, averaged across years", "ratio"),
    "overnight_baseload_per_area": ("1000 * mean demand 00:00-04:59 / "
                                    "floor_area_m2", "W/m2"),
}


def _performance_frame(ctx: ToolContext, building_ids: list[str] | None,
                       year: int | None, min_coverage: float,
                       metric: str | None = None) -> pd.DataFrame:
    """Per-building annual totals joined to metadata, coverage-filtered.

    A metric whose divisor is not in the catalog -- operating hours are
    inferred from the profile, not recorded -- brings its own column in
    here, so that every route that builds a frame can rank on it rather
    than only the one tool that first needed it.
    """
    params: list = []
    where = "1=1" + in_clause("a.building_id", building_ids, params)
    if year:
        where += " AND EXTRACT(year FROM a.period_start) = ?"; params.append(year)
    # Coverage of the period, not of its worst month. MIN(coverage_fraction)
    # disqualified a building that is 97% complete over 2017 because one month
    # dipped to 71%: eight of the ten highest kWh/person buildings are exactly
    # that shape, and a floor named "min_coverage" reads as a bar on the span
    # being measured rather than on every month inside it.
    #
    # The denominator is the calendar year when one is named, so months with
    # no rows at all still count against a building, and otherwise the hours
    # the months present were expected to hold.
    expected = ("8784.0" if year in (2016, 2020, 2024)
                else "8760.0" if year
                else "SUM(a.n_observations / NULLIF(a.coverage_fraction, 0))")
    params.append(min_coverage)
    frame = ctx.db.df(f"""
        SELECT a.building_id,
               ANY_VALUE(b.building_type_canonical) AS building_type,
               -- Carried so `across` can group by them. They cost nothing to
               -- select and grouping by a column the frame does not hold is a
               -- KeyError at the end of a long scan.
               ANY_VALUE(b.building_subtype) AS building_subtype,
               ANY_VALUE(b.industry) AS industry,
               ANY_VALUE(b.subindustry) AS subindustry,
               ANY_VALUE(b.floor_area_m2) AS floor_area_m2,
               ANY_VALUE(b.occupants) AS occupants,
               ANY_VALUE(b.year_built) AS year_built,
               ANY_VALUE(s.city) AS city,
               ANY_VALUE(s.country_canonical) AS country,
               ANY_VALUE(b.site_id) AS site_id,
               SUM(a.total_kwh) AS total_kwh,
               AVG(a.mean_demand) AS mean_demand,
               MAX(a.peak_demand) AS peak_demand,
               SUM(a.n_observations) AS observed_hours,
               MIN(a.coverage_fraction) AS min_coverage_fraction,
               SUM(a.n_observations) / NULLIF({expected}, 0) AS period_coverage
        FROM agg_monthly a
        JOIN buildings b ON b.building_id = a.building_id
        JOIN sites s ON s.site_id = b.site_id
        WHERE {where}
        GROUP BY a.building_id
        HAVING SUM(a.n_observations) / NULLIF({expected}, 0) >= ?
    """, params)
    if metric == "eui_per_operating_hour" and not frame.empty:
        frame = frame.merge(ctx.db.df(OPERATING_HOURS_SQL),
                            on="building_id", how="left")
    return frame


# A whole building metered at under 100 m2 is a metadata error, not a small
# building: the median floor area in this snapshot is 5,074 m2 and only 16 of
# 1,657 buildings fall below this floor. Area-normalised metrics divide by that
# number, so a wrong denominator -- not unusual consumption -- is what produces
# the extreme values (Bear_parking_Bridget: a parking structure recorded as
# 36 m2, 4,015 kWh/m2, at coverage 0.994). Flag the denominator; never silently
# drop the row.
MIN_PLAUSIBLE_FLOOR_AREA_M2 = 100.0

AREA_NORMALISED = ("eui", "peak_per_area", "peak_power_density",
                   "eui_per_operating_hour", "baseload_per_area",
                   "overnight_baseload_per_area")

# Hours a day a building is apparently in use, inferred from its own daily
# profile: the hours whose mean demand clears the overnight floor by a quarter
# of the day's range. The same rule analyze_operating_pattern applies, read
# here off the precomputed profile so a portfolio-wide question does not
# rescan every reading.
# Counted rather than filtered, so a flat profile reports zero hours instead
# of dropping out of every ranking that joins these hours. The comparison
# carries a tolerance because on a flat profile its two sides are equal to
# within floating point, and parallel sums do not reproduce exactly.
OPERATING_HOURS_SQL = """
    WITH p AS (SELECT building_id, hour_of_day AS h, SUM(mean_demand) AS m
               FROM profile_daily GROUP BY 1, 2),
         f AS (SELECT building_id, AVG(m) AS floor FROM p
               WHERE h BETWEEN 0 AND 4 GROUP BY 1),
         r AS (SELECT building_id, MIN(m) AS lo, MAX(m) AS hi FROM p GROUP BY 1)
    SELECT p.building_id,
           COUNT(*) FILTER (
               WHERE p.m > f.floor + 0.25 * (r.hi - r.lo)
                     + 1e-9 * ABS(NULLIF(r.hi, 0))) AS operating_hours_per_day
    FROM p JOIN f USING (building_id) JOIN r USING (building_id)
    GROUP BY 1
"""

# Metrics that are a maximum rather than a sum. A coverage floor is a fitness
# bar for comparing totals -- a part-year understates a year's consumption --
# but a peak is an observed extreme: a short record can only understate it, so
# a building seen above a threshold is above it whatever else is missing.
# Screening education buildings at 0.9 dropped 167 of 589 and 29 of the 144
# that exceed 50 W/m2.
EXTREME_METRICS = ("peak_demand", "peak_per_area", "peak_power_density")

# An intensity is a per-year figure: kWh per square metre per year, kWh per
# person per year. Computed over a span of unknown length it is not that
# measure, just a bigger number -- and over this snapshot's eighteen months it
# reorders the ranking, putting a building at 13,638 kWh/person above one at
# 15,794. So a year is supplied when none is named. The last calendar year the
# snapshot covers end to end; an explicit year always wins.
PER_YEAR_METRICS = ("eui", "per_occupant", "eui_per_operating_hour")
LAST_FULL_YEAR = 2017


def _questionable_denominator(df: pd.DataFrame, metric: str) -> pd.Series:
    """Rows whose normalised metric rests on an implausible denominator."""
    if metric in AREA_NORMALISED:
        return df["floor_area_m2"] < MIN_PLAUSIBLE_FLOOR_AREA_M2
    if metric == "per_occupant":
        return df["occupants"] <= 0
    return pd.Series(False, index=df.index)


# An annual intensity divides a year's consumption, so it needs enough of a
# year to divide. The snapshot already decides that per building -- can_eui is
# "has data, has a floor area, and has enough record" -- and a plan that
# forgets the filter should not get a different answer from one that remembers
# it. Panther_office_Lois has a floor area and an apparent 2017 EUI of 1,101,
# which put it third among offices; its record is too short for the figure to
# mean anything, and the snapshot says so.
#
# A peak per square metre is deliberately not in this map. It divides a
# maximum, not a sum: a short record can only understate a peak, so a building
# observed above a threshold is above it whatever else is missing.
ELIGIBILITY = {"eui": "can_eui", "eui_per_operating_hour": "can_eui",
               "baseload_per_area": "can_eui",
               "overnight_baseload_per_area": "can_eui",
               "per_occupant": "can_occupancy_normalisation"}




def _eligible_for(ctx: ToolContext, metric: str,
                  building_ids: list[str] | None,
                  eligibility: str = "fit") -> tuple[list[str] | None, int]:
    """The requested cohort narrowed to whom the metric applies, and how many left."""
    gate = ELIGIBILITY.get(metric)
    if not gate:
        return building_ids, 0
    if eligibility == "structural" and gate in STRUCTURAL:
        # Everyone the metric can be computed for, fit or not. The caller has
        # taken responsibility for the fitness bar.
        eligible = ctx.db.df(
            f"SELECT building_id FROM buildings WHERE {STRUCTURAL[gate]}"
        )["building_id"]
        keep = set(eligible)
        if building_ids is None:
            return sorted(keep), 0
        kept = [b for b in building_ids if b in keep]
        return kept, len(building_ids) - len(kept)
    eligible = ctx.db.df(
        f"SELECT building_id FROM capabilities WHERE {gate}")["building_id"]
    # An omitted cohort means every building, and every building still has to
    # be eligible. Screening only a named cohort meant a portfolio-wide
    # question was answered over buildings a per-cohort question would have
    # excluded -- 1,571 buildings banded on overnight intensity where 1,254
    # can carry the division.
    if building_ids is None:
        return list(eligible), 0
    kept = [b for b in building_ids if b in set(eligible)]
    return kept, len(building_ids) - len(kept)


def _name_metric_column(out: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rename ``metric_value`` to the metric without duplicating a label.

    Most metrics are derived and their name is free, but the pass-through ones
    -- total_kwh, mean_demand, peak_demand -- are already columns of the frame.
    Renaming blindly left two columns with the same label, and the next lookup
    of it raised "The column label 'total_kwh' is not unique". The raw column
    goes, because under a coverage adjustment ``metric_value`` is the adjusted
    figure and the raw one is the stale copy.
    """
    if "metric_value" not in out.columns:
        return out
    if metric in out.columns:
        out = out.drop(columns=[metric])
    return out.rename(columns={"metric_value": metric})


def _apply_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df.copy()
    if metric == "load_factor":
        out["metric_value"] = out["mean_demand"] / out["peak_demand"].replace(0, np.nan)
    elif metric == "peak_to_average_ratio":
        out["metric_value"] = out["peak_demand"] / out["mean_demand"].replace(0, np.nan)
    elif metric == "eui":
        out["metric_value"] = out["total_kwh"] / out["floor_area_m2"]
    elif metric == "peak_per_area":
        out["metric_value"] = out["peak_demand"] / out["floor_area_m2"]
    elif metric == "peak_power_density":
        out["metric_value"] = 1000.0 * out["peak_demand"] / out["floor_area_m2"]
    elif metric == "eui_per_operating_hour":
        hours = out["operating_hours_per_day"].replace(0, np.nan) * 365.0
        out["metric_value"] = out["total_kwh"] / out["floor_area_m2"] / hours
    elif metric == "baseload_per_area":
        out["metric_value"] = out["baseload"] / out["floor_area_m2"]
    elif metric == "overnight_baseload_per_area":
        out["metric_value"] = (1000.0 * out["overnight_baseload"]
                               / out["floor_area_m2"])
    elif metric == "per_occupant":
        out["metric_value"] = out["total_kwh"] / out["occupants"]
    else:
        out["metric_value"] = out[metric]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["metric_value"])


@tool("performance")
def calculate_normalized_performance(
    ctx: ToolContext,
    # Every other tool here reads an omitted cohort as "all buildings", and the
    # planning rules say so in as many words. This one made it required, so a
    # plan that followed the rule for "across the entire dataset" was rejected
    # for a missing argument and the run fell back to rules.
    building_ids: list[str] | None = None,
    metric: Literal["eui", "peak_per_area", "peak_power_density",
                    "eui_per_operating_hour", "baseload_per_area",
                    "per_occupant", "total_kwh", "mean_demand",
                    "peak_demand", "load_factor",
                    "peak_to_average_ratio", "overnight_baseload_per_area",
                    "seasonal_cv", "daily_load_factor"] = "eui",
    year: int | None = None,
    min_coverage: float | None = None,
    coverage_adjustment: Literal["none", "annualised", "compare"] = "none",
    at_least: float | None = None,
    at_most: float | None = None,
    # "more than 50% above the median EUI of offices at the same site" is not a
    # number: it is 19 numbers, one per site, and which one applies depends on
    # the building being tested. Every threshold here was cohort-wide -- a
    # percentile, a fence, a fixed value computed once over the whole set -- so
    # the question had no expression and came back as the unfiltered cohort.
    # With relative_to, at_least/at_most read as multiples of the building's
    # own group statistic: relative_to="site", at_least=1.5.
    relative_to: Literal["site", "city", "country", "building_type",
                         "building_subtype", "industry",
                         "subindustry"] | None = None,
    relative_statistic: Literal["median", "mean"] = "median",
    adjust_for: list[Literal["year_built", "floor_area_m2", "occupants"]] | None = None,
    across: Literal["building", "site", "city", "country",
                    "building_type", "building_subtype", "industry",
                    "subindustry"] = "building",
    bands: int | None = None,
    eligibility: Literal["fit", "structural"] = "fit",
) -> ToolResult:
    """Area- or occupant-normalised performance, including electricity EUI.

    An annual total is summed from the hours that are present, so a year with
    gaps produces a figure quietly lower than the building's actual year.
    ``coverage_adjustment`` says what to do about that:

    - ``"none"`` reports what was measured.
    - ``"annualised"`` scales the total to the whole calendar year, on the
      assumption that the missing hours resemble the present ones.
    - ``"compare"`` reports both and the gap between them, per building, and
      carries the median understatement in the evidence. This is the answer to
      "what do missing values do to annual EUI".

    Both adjusted modes need ``year``: a span of unknown length cannot be
    scaled to one.

    ``adjust_for`` answers "normalised against their age and size": each
    building is ranked within its group on the named characteristics, the
    metric is regressed on those ranks across the whole cohort, and what is
    reported is the residual -- the part of the metric its age and size do not
    explain. With ``across="site"`` the residuals are summarised as a median
    per site, so a site ranks high when its buildings run flatter, or heavier,
    than their characteristics predict.

    The regression is fitted across the cohort, not within each group. Fitting
    per group forces its residuals to average zero, so every group scores
    about the same and the ranking measures skew rather than performance: on
    this snapshot the per-site fit spreads sites over 0.06 of load factor and
    the pooled fit over 0.36, and they disagree about which sites lead.

    ``bands`` cuts the cohort into that many equal-count groups on the metric,
    which is what "group the portfolio into high, medium and low using simple
    percentile cutoffs" asks for: band 1 is the lowest. Equal counts rather
    than equal ranges, because three named categories from percentiles is
    what the phrase means and any other split needs a reason the question
    does not give. The size of each band is reported, so a reader can see
    that "high" is a third of the portfolio and not a handful of outliers.

    ``at_least`` and ``at_most`` keep only the buildings whose metric falls in
    the range, which is how "which buildings exceed 50 W/m2" is asked. The
    filter belongs here because a large result is handed back as an artifact
    whose preview holds five rows: a plan cannot select from what it receives.

    ``peak_power_density`` is peak demand per square metre in W/m2, the unit
    the field quotes it in; ``peak_per_area`` is the same quantity in kW/m2.
    They differ by 1000, and a threshold compared against the wrong one of
    them matches nothing and reads as an empty answer rather than a unit slip.

    ``min_coverage`` defaults to 0.9, a fitness bar for comparing buildings
    against each other -- and to 0 for a peak, and under either adjustment,
    because a floor there removes exactly the buildings the question is asking
    after. A peak is an observed extreme: a short record can only understate
    it, so a building seen above a threshold is above it whatever is missing.
    Screening education buildings at 0.9 dropped 167 of 589 and 29 of the 144
    that exceed 50 W/m2. The
    median understatement across this snapshot is 0.17% with the floor and
    0.57% without it. Pass a value to override either default.
    """
    if metric not in METRICS:
        return insufficient("unknown_metric", f"valid metrics: {sorted(METRICS)}")
    if coverage_adjustment != "none" and metric in BASELOAD_METRICS:
        return insufficient(
            "coverage_adjustment",
            f"{metric} is a level, not a total: there is nothing to scale to a "
            "whole year, and a short record does not deflate it")
    if coverage_adjustment != "none" and not year:
        return insufficient("year",
                            f"coverage_adjustment={coverage_adjustment!r} scales a "
                            "total to a calendar year, so it needs one named")
    # The 0.9 floor is a fitness bar for comparing buildings against each
    # other. Under a coverage adjustment it removes the very buildings being
    # measured, so the default flips rather than fighting the call: asking
    # what the gaps cost, over a cohort screened for having none, answered
    # 0.17% where the figure is 0.57%. An explicit floor is still honoured.
    if year is None and metric in PER_YEAR_METRICS:
        year = LAST_FULL_YEAR
    if min_coverage is None:
        min_coverage = (0.0 if coverage_adjustment != "none"
                        or metric in EXTREME_METRICS else 0.9)
        # One screen, not two -- the same convention compare_building_groups
        # follows. Where the floor does the screening, the cohort is everyone
        # the metric can be computed for; leaving the capability gate on as
        # well applies a second coverage verdict, scoped to the whole record
        # rather than the year asked about. Median 2017 office EUI came out
        # 111.7 over 229 buildings against 121.8 over 262 on the floor the
        # metric actually needs.
        if min_coverage and eligibility == "fit":
            eligibility = "structural"
    # The eligibility screen encodes "has enough data", and a coverage
    # adjustment is a question *about* the buildings that do not. Screening
    # first answers it over the survivors: the median understatement came back
    # 0.531% where the figure over every building with a floor area is 0.571%.
    # The same reasoning already flips the coverage floor two lines above.
    if coverage_adjustment == "none":
        building_ids, ineligible = _eligible_for(ctx, metric, building_ids,
                                                 eligibility)
    else:
        ineligible = 0
    if ineligible and not building_ids:
        return insufficient(
            ELIGIBILITY[metric],
            f"none of the {ineligible} requested building(s) are eligible for "
            f"{metric}: the snapshot marks {ELIGIBILITY[metric]} false for all")
    df = _frame_for(ctx, metric, building_ids, year, min_coverage)
    if df.empty:
        return insufficient("coverage",
                            f"no requested building has a period with coverage "
                            f">= {min_coverage}" + (f" in {year}" if year else ""))
    scored = _apply_metric(df, metric)
    if scored.empty:
        needed = {"eui": "floor_area_m2", "peak_per_area": "floor_area_m2",
                  "per_occupant": "occupants"}.get(metric, metric)
        if needed in df.columns:
            values = pd.to_numeric(df[needed], errors="coerce")
            absent, nonpositive = int(values.isna().sum()), int((values <= 0).sum())
            reason = (f"absent for {absent}" if absent else "") + \
                     (" and " if absent and nonpositive else "") + \
                     (f"zero or negative for {nonpositive}" if nonpositive else "")
        else:
            reason = "not available"
        return insufficient(needed,
                            f"{metric} needs a usable {needed}: {reason} "
                            f"of {len(df)} requested building(s)")
    # "Which buildings exceed 50 W/m2" is a filter on the measure, and the
    # tool is the only place it can be applied: 589 scored buildings go to an
    # artifact, whose preview carries five rows, so a plan cannot select from
    # what it gets back.
    # Counted before the threshold: a building filtered out by the range
    # asked for is not a building that lacked the inputs, and reporting it as
    # one turned a complete answer into PARTIAL with "445 building(s) lacked
    # the inputs for peak_power_density".
    # Counted before any grouping: once the rows are one per site, the
    # difference from the building frame is the grouping itself, not buildings
    # that lacked an input, and it reported "1,427 building(s) lacked the
    # inputs for load_factor" over an answer of eleven sites.
    dropped = len(df) - len(scored)
    adjusted: dict = {}
    if adjust_for:
        group = {"building": None, "site": "site_id", "city": "city",
                 "country": "country", "building_type": "building_type",
                 "building_subtype": "building_subtype",
                 "industry": "industry",
                 "subindustry": "subindustry"}[across]
        have = [c for c in adjust_for if c in scored.columns]
        usable = scored.dropna(subset=have + ["metric_value"]) if have else scored.iloc[0:0]
        if len(usable) < len(have) + 2:
            return insufficient(
                "adjust_for",
                f"{len(usable)} building(s) carry both {metric} and all of "
                f"{adjust_for}; a regression on {len(have)} characteristic(s) "
                f"needs more than that")
        ranked = usable.copy()
        for column in have:
            ranked[f"rank_{column}"] = (
                ranked.groupby(group)[column].rank(pct=True) if group
                else ranked[column].rank(pct=True))
        design = np.column_stack(
            [np.ones(len(ranked))] + [ranked[f"rank_{c}"].to_numpy(float) for c in have])
        beta, *_ = np.linalg.lstsq(design, ranked["metric_value"].to_numpy(float),
                                   rcond=None)
        ranked["expected"] = design @ beta
        ranked["residual"] = ranked["metric_value"] - ranked["expected"]
        adjusted = {"adjusted_for": have, "across": across,
                    "coefficients": {c: round(float(b), 6)
                                     for c, b in zip(have, beta[1:])},
                    "buildings_fitted": int(len(ranked))}
        if group:
            scored = (ranked.groupby(group)
                            .agg(n_buildings=("building_id", "size"),
                                 median_residual=("residual", "median"),
                                 median_metric=("metric_value", "median"))
                            .reset_index()
                            .sort_values("median_residual", ascending=False))
            scored["metric_value"] = scored["median_residual"]
        else:
            scored = ranked.sort_values("residual", ascending=False)
            scored["metric_value"] = scored["residual"]

    band_sizes: dict = {}
    if bands and len(scored) >= bands:
        # NTILE's own division, not qcut's: with 1,256 buildings in three
        # bands the remainder goes to the earlier ones, so the top band holds
        # 418 and not 419. The same boundary the size-decile work settled.
        ranked = scored.sort_values("metric_value").reset_index(drop=True)
        total, buckets = len(ranked), int(bands)
        larger, size = total % buckets, total // buckets

        def _band(position: int) -> int:
            edge = larger * (size + 1)
            if position < edge:
                return position // (size + 1) + 1
            return larger + (position - edge) // size + 1

        ranked["band"] = [_band(i) for i in range(total)]
        scored = ranked
        band_sizes = {int(b): int(n) for b, n
                      in scored["band"].value_counts().sort_index().items()}
    kept_over = None
    relative = None
    if relative_to is not None:
        key = {"site": "site_id", "city": "city", "country": "country",
               "building_type": "building_type",
               "building_subtype": "building_subtype",
               "industry": "industry", "subindustry": "subindustry"}[relative_to]
        if key not in scored.columns:
            return insufficient(
                "relative_to", f"the catalog carries no {key} to compare within")
        # A group of one is its own median, so every member sits exactly on the
        # line and none clears it. Saying so beats reporting a count that
        # quietly excludes every site with a single building of this kind.
        group = scored.groupby(key)["metric_value"]
        reference = (group.transform("median") if relative_statistic == "median"
                     else group.transform("mean"))
        scored = scored.assign(**{
            f"group_{relative_statistic}": reference.round(6),
            "ratio_to_group": (scored["metric_value"] / reference).round(6)})
        sizes = group.transform("size")
        alone = int((sizes == 1).sum())
        relative = {"relative_to": relative_to,
                    "statistic": relative_statistic,
                    "groups": int(scored[key].nunique()),
                    "buildings_alone_in_their_group": alone}
    if at_least is not None or at_most is not None:
        before = len(scored)
        # Against the group statistic when one was named, against the raw
        # measure otherwise. The same two parameters, read in the units the
        # question used.
        against = (scored["ratio_to_group"] if relative is not None
                   else scored["metric_value"])
        if at_least is not None:
            scored = scored[against > float(at_least)]
            against = against.loc[scored.index]
        if at_most is not None:
            scored = scored[against <= float(at_most)]
        kept_over = f"{len(scored)} of {before}"
        if scored.empty:
            return empty(f"no building's {metric} falls in the requested range")
    # `across` asked for one row per group, and without `adjust_for` it was
    # accepted and ignored: "the EUI of each building type" came back as 1,186
    # per-building rows with the grouping nowhere in it. The adjusted path
    # below has always grouped; this is the same request without the
    # regression attached to it.
    grouped_plain = group_key = None
    if not adjusted and across not in ("building", None):
        key = {"site": "site_id", "city": "city", "country": "country",
               "building_type": "building_type",
               "building_subtype": "building_subtype",
               "industry": "industry", "subindustry": "subindustry"}[across]
        if key not in scored.columns:
            return insufficient(
                "across", f"the catalog carries no {key} to group by")
        # The unrecorded group is kept and named. Dropping it would report a
        # median over the buildings that happen to carry a subtype as though
        # it were over all of them.
        labelled = scored.copy()
        labelled[key] = labelled[key].fillna(f"(no {key} recorded)")
        grouped_plain = (labelled.groupby(key)
                         .agg(n_buildings=("building_id", "nunique"),
                              metric_value=("metric_value", "median"),
                              mean_metric=("metric_value", "mean"),
                              min_metric=("metric_value", "min"),
                              max_metric=("metric_value", "max"))
                         .reset_index()
                         .sort_values("metric_value", ascending=False))
        scored = grouped_plain
        group_key = key
        columns = [key, "n_buildings", "metric_value", "mean_metric",
                   "min_metric", "max_metric"]
    if grouped_plain is not None:
        pass
    elif adjusted and adjusted.get("across") != "building":
        # One row per group, not per building: the per-building columns are
        # gone by construction, and asking for them here failed the whole
        # call with a KeyError.
        group_key = {"site": "site_id", "city": "city", "country": "country",
                     "building_type": "building_type",
                     "building_subtype": "building_subtype",
                     "industry": "industry",
                     "subindustry": "subindustry"}[adjusted["across"]]
        columns = [group_key, "n_buildings", "median_metric", "metric_value"]
    else:
        columns = ["building_id", "building_type", "city", "country",
                   "floor_area_m2", "occupants", "total_kwh", "metric_value"]
        if band_sizes:
            columns.append("band")
        if metric == "eui_per_operating_hour":
            # The inference is the weak link, so it is reported beside the
            # number it produced rather than left in the method note.
            columns.insert(-1, "operating_hours_per_day")
        if metric in BASELOAD_METRICS:
            columns.insert(-1, "baseload")
        if relative is not None:
            # The group's figure and the ratio to it are the working of the
            # answer: "1,006 kWh/m2, which is 2.4x the median for its site"
            # is checkable, "1,006" on its own is not. The grouping column
            # comes too -- a per-site result that does not say which site
            # cannot be read or checked.
            columns += [c for c in (key, f"group_{relative_statistic}",
                                    "ratio_to_group") if c not in columns]
    # Not every metric comes from the same frame, so not every frame carries
    # every column: a baseload is read off the hourly record and has no annual
    # total beside it.
    columns = [c for c in columns if c in scored.columns]
    extra: dict = {}
    if coverage_adjustment != "none":
        # The year is the denominator, not the rows the aggregate happens to
        # hold. A day with no reading has no row at all, so measuring coverage
        # against the rows present makes a meter that died in November look
        # complete: 479 of 1,256 buildings hold fewer than 365 daily rows for
        # 2017, and one of them, Crow_education_Omer, is missing 52 whole days.
        hours = 24 * (366 if calendar.isleap(int(year)) else 365)
        observed = pd.to_numeric(scored["observed_hours"], errors="coerce")
        fraction = (observed / hours).clip(upper=1.0)
        scaled = scored["metric_value"] / fraction.replace(0, np.nan)
        understated = (1 - fraction) * 100.0
        if coverage_adjustment == "annualised":
            scored = scored.assign(metric_value=scaled)
        else:
            scored = scored.assign(**{f"{metric}_annualised": scaled,
                                      "understated_pct": understated,
                                      "hours_observed": observed,
                                      "hours_in_year": hours})
            columns += [f"{metric}_annualised", "understated_pct",
                        "hours_observed", "hours_in_year"]
        extra = {"coverage_adjustment": coverage_adjustment,
                 "hours_in_year": hours,
                 "median_understated_pct": round(float(understated.median()), 3),
                 "mean_understated_pct": round(float(understated.mean()), 3),
                 "buildings_understated_over_5pct": int((understated > 5).sum())}
        if coverage_adjustment == "compare":
            # "Quantify the impact" is one number, and the table is where it
            # came from. A tool that computed a summary as the point of the
            # call should say so rather than leave a reader to find it among
            # 1,547 rows, so the headline names the answer and composition
            # leads with it.
            extra["headline"] = {
                "label": ("median annual "
                          + ("EUI" if metric == "eui" else metric.replace("_", " "))
                          + " understated by the hours missing from the year"),
                "value": extra["median_understated_pct"], "unit": "%"}
    if grouped_plain is not None and len(scored):
        # `across` returned one row per group, sorted by the metric, and the
        # preview carries five of them. Read as a per-building frame the
        # headline called nine cities "9 buildings" and reported the median of
        # the nine city medians -- Tempe's 102.8 -- for a question asking which
        # city was highest, where the leading row says 181.3.
        #
        # The unrecorded group is kept in the table, because dropping it would
        # describe part of the cohort as though it were all of it, but it does
        # not get to lead a ranking: "(no city recorded)" is not a city, and
        # naming it as the highest answers a question nobody asked.
        placeholder = f"(no {group_key} recorded)"
        named = scored[scored[group_key] != placeholder]
        top = (named if len(named) else scored).iloc[0]
        unit = METRICS[metric][1]
        extra.setdefault("headline", {
            # One decimal is right for a kWh/m2 and wrong for a ratio: a
            # seasonal CV of 0.130 printed as "0.1" loses the finding and the
            # ranking with it, since three categories round to the same 0.1.
            "label": (f"highest median {metric.replace('_', ' ')} by "
                      f"{group_key.replace('_', ' ')}: {top[group_key]} "
                      f"{_readable(float(top['metric_value']))} {unit} "
                      f"(n = {int(top['n_buildings'])})"
                      + (f" of {len(scored)} {group_key.replace('_', ' ')} "
                         f"groups" if len(scored) > 1 else "")),
            "value": round(float(top["metric_value"]), 6),
            "unit": unit})
    if kept_over is None and not adjusted and grouped_plain is None \
            and len(scored) > PREVIEW_ROWS:
        # A cohort-wide distribution goes back as an artifact whose preview
        # holds five rows, and five rows out of 1,456 do not describe a
        # portfolio. The median is what the set amounts to, so the tool names
        # the number that answered rather than leaving it to be read off a
        # table that was truncated on the way out.
        extra.setdefault("headline", {
            "label": f"median {metric.replace('_', ' ')} across "
                     f"{len(scored)} buildings",
            "value": round(float(scored["metric_value"].median()), 6),
            "unit": METRICS[metric][1]})
    if metric == "eui_per_operating_hour" and "operating_hours_per_day" in scored:
        extra["median_operating_hours_per_day"] = round(
            float(scored["operating_hours_per_day"].median()), 1)
        extra["inferred_divisor"] = (
            "operating hours per day inferred from each building's own daily "
            "profile (hours above the overnight floor by a quarter of the "
            "day's range), multiplied by 365")
    out = scored[columns]
    out = _name_metric_column(out, metric)
    result = frame_result(out.sort_values(metric, ascending=False), ctx.store,
                          name=f"performance_{metric}",
                          evidence={"metric": metric, "unit": METRICS[metric][1],
                                    "year": year, "min_coverage": min_coverage,
                                    "ineligible_excluded": ineligible or None,
                                    **({"adjustment": adjusted} if adjusted else {}),
                                    "relative": relative,
                                    "threshold": (None if kept_over is None else
                                                  {"at_least": at_least,
                                                   "at_most": at_most,
                                                   "kept": kept_over,
                                                   "measured_against": (
                                                       f"each building's {relative_statistic} "
                                                       f"within its {relative_to}"
                                                       if relative else metric)}),
                                    "buildings_scored": len(scored),
                                    "buildings_dropped": dropped, **extra,
                                    # After the distribution's own headline:
                                    # a banded answer is about the bands.
                                    **({"bands": int(bands),
                                        "band_sizes": band_sizes,
                                        "headline": {
                                            "label": f"buildings in the top "
                                                     f"band of {int(bands)} on "
                                                     f"{metric.replace('_', ' ')}",
                                            "value": band_sizes.get(int(bands), 0),
                                            "unit": "buildings"}}
                                       if band_sizes else {})})
    if dropped:
        result.status = result.status.PARTIAL if hasattr(result.status, "PARTIAL") else result.status
        result.message = f"{dropped} building(s) lacked the inputs for {metric}"
    return result


@tool("performance")
def construct_peer_group(
    ctx: ToolContext,
    building_id: str,
    match_on: list[Literal["building_type", "floor_area", "country", "city",
                           "year_built", "site"]] | None = None,
    area_tolerance: float = 0.5,
    year_tolerance: int = 20,
    min_peers: int = 5,
) -> ToolResult:
    """Build a comparable cohort by stated criteria, returning the predicate used.

    The predicate is returned alongside the members so the cohort can be
    reproduced exactly.
    """
    criteria = match_on or ["building_type", "floor_area", "country"]
    valid = {"building_type", "floor_area", "country", "city", "year_built", "site"}
    unknown = set(criteria) - valid
    if unknown:
        return insufficient("unknown_criteria",
                            f"unknown match_on {sorted(unknown)}; valid: {sorted(valid)}")
    target = ctx.db.df("""
        SELECT b.building_id, b.building_type_canonical AS building_type,
               b.floor_area_m2, b.year_built, b.site_id,
               s.city, s.country_canonical AS country
        FROM buildings b JOIN sites s ON s.site_id = b.site_id
        WHERE b.building_id = ?""", [building_id])
    if target.empty:
        return insufficient("unknown_building", f"no such building: {building_id}")
    t = target.iloc[0]

    where, params, predicate = ["b.building_id <> ?"], [building_id], []
    if "building_type" in criteria:
        if pd.isna(t["building_type"]):
            return insufficient("building_type",
                                f"{building_id} has no canonical building type")
        where.append("b.building_type_canonical = ?"); params.append(t["building_type"])
        predicate.append(f"building_type = {t['building_type']}")
    if "floor_area" in criteria:
        if pd.isna(t["floor_area_m2"]):
            return insufficient("floor_area_m2", f"{building_id} has no floor area")
        lo = t["floor_area_m2"] * (1 - area_tolerance)
        hi = t["floor_area_m2"] * (1 + area_tolerance)
        where.append("b.floor_area_m2 BETWEEN ? AND ?"); params += [lo, hi]
        predicate.append(f"floor_area_m2 in [{lo:.0f}, {hi:.0f}]")
    if "country" in criteria and pd.notna(t["country"]):
        where.append("s.country_canonical = ?"); params.append(t["country"])
        predicate.append(f"country = {t['country']}")
    if "city" in criteria and pd.notna(t["city"]):
        where.append("s.city = ?"); params.append(t["city"])
        predicate.append(f"city = {t['city']}")
    if "site" in criteria:
        where.append("b.site_id = ?"); params.append(t["site_id"])
        predicate.append(f"site_id = {t['site_id']}")
    if "year_built" in criteria and pd.notna(t["year_built"]):
        where.append("b.year_built BETWEEN ? AND ?")
        params += [int(t["year_built"]) - year_tolerance,
                   int(t["year_built"]) + year_tolerance]
        predicate.append(f"year_built +/- {year_tolerance}")

    peers = ctx.db.df(f"""
        SELECT b.building_id, b.building_type_canonical AS building_type,
               b.floor_area_m2, b.year_built, s.city, s.country_canonical AS country
        FROM buildings b
        JOIN sites s ON s.site_id = b.site_id
        JOIN capabilities c ON c.building_id = b.building_id
        WHERE {" AND ".join(where)} AND c.can_consumption_analysis
        ORDER BY b.building_id
    """, params)
    evidence = {"target": building_id, "criteria": criteria,
                "predicate": " AND ".join(predicate), "n_peers": len(peers)}
    if len(peers) < min_peers:
        return insufficient("peer_group_too_small",
                            f"only {len(peers)} peers found (need {min_peers}); "
                            "relax match_on or the tolerances",
                            evidence=evidence)
    return ok({"peer_group": peers["building_id"].tolist(),
               "predicate": evidence["predicate"]},
              rows=len(peers), evidence=evidence)


@tool("performance")
def benchmark_building(
    ctx: ToolContext,
    building_id: str,
    peer_ids: list[str],
    metric: Literal["eui", "peak_per_area", "per_occupant", "total_kwh"] = "eui",
    year: int | None = None,
    min_coverage: float | None = None,
) -> ToolResult:
    """Position one building against its peers: percentile, z-score, ratio.

    No coverage floor by default, as in the other peer comparisons here: the
    peer group is already screened by the capability the metric needs, and a
    second screen removes peers rather than unfit ones. At 0.9 it dropped 32
    of 293 offices and moved this building from the 79th percentile to the
    78th -- a change in the answer produced by a change in who was counted.
    """
    if min_coverage is None:
        min_coverage = 0.0
    # A peer comparison runs over a common period. Left open it sums
    # whatever span each building happens to hold, and the percentile that
    # comes back is partly a ranking of record lengths: Robin_office_Donald
    # sat at the 77th percentile over the whole record and the 79th over the
    # year every one of its peers also has.
    if year is None:
        year = LAST_FULL_YEAR
    everyone = list(dict.fromkeys([building_id, *peer_ids]))
    df = _apply_metric(_frame_for(ctx, metric, everyone, year, min_coverage), metric)
    if df.empty or building_id not in set(df["building_id"]):
        return insufficient("target_not_scored",
                            f"{building_id} could not be scored on {metric} "
                            f"with coverage >= {min_coverage}")
    peers = df[df["building_id"] != building_id]
    if peers.empty:
        return insufficient("no_scored_peers", "no peer could be scored on this metric")
    target = float(df.loc[df["building_id"] == building_id, "metric_value"].iloc[0])
    values = peers["metric_value"].to_numpy(dtype=float)
    # Out of a hundred, which is how a percentile is quoted -- reported as a
    # fraction it reads as 0.79 where the answer is the 79th percentile -- and
    # over the population rather than over the peers. The building is one of
    # the buildings it is being placed among: dividing by its peers alone put
    # it at the 79.52nd percentile of 293 where it is at the 79.25th of 294.
    percentile = float((values < target).sum()) / (len(values) + 1) * 100.0
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    data = {
        "building_id": building_id, "metric": metric,
        "unit": METRICS[metric][1], "value": target,
        "peer_count": int(len(values)),
        "peer_median": float(np.median(values)),
        "peer_mean": float(values.mean()),
        "percentile": percentile,
        "z_score": (target - values.mean()) / std if std else None,
        "ratio_to_peer_median": target / float(np.median(values)) if np.median(values) else None,
        "rank": int((values > target).sum() + 1),
    }
    return ok(data, evidence={"headline": {
                                  "label": f"percentile among {len(values)} "
                                           f"peers on {metric}",
                                  "value": round(percentile, 2),
                                  "unit": "percentile"},
                              "year": year, "min_coverage": min_coverage,
                              "peers_scored": int(len(values)),
                              "peers_requested": len(peer_ids)})


def _shape_frame(ctx: ToolContext, building_ids: list[str] | None,
                 year: int | None, min_coverage: float) -> pd.DataFrame:
    """Per-building mean and peak demand from the daily aggregate."""
    params: list = []
    where = "1=1" + in_clause("a.building_id", building_ids, params)
    if year:
        where += " AND EXTRACT(year FROM a.period_start) = ?"; params.append(year)
    params.append(min_coverage)
    # The same metadata every other frame carries. A shape metric grouped by
    # site asked for site_id and got a KeyError, because this frame was written
    # for one tool that never grouped: "which sites lead on load factor once
    # age and size are accounted for" failed on the column, not on the
    # question.
    return ctx.db.df(f"""
        SELECT a.building_id,
               ANY_VALUE(b.building_type_canonical) AS building_type,
               -- Carried so `across` can group by them. They cost nothing to
               -- select and grouping by a column the frame does not hold is a
               -- KeyError at the end of a long scan.
               ANY_VALUE(b.building_subtype) AS building_subtype,
               ANY_VALUE(b.industry) AS industry,
               ANY_VALUE(b.subindustry) AS subindustry,
               ANY_VALUE(b.floor_area_m2) AS floor_area_m2,
               ANY_VALUE(b.occupants) AS occupants,
               ANY_VALUE(b.year_built) AS year_built,
               ANY_VALUE(b.site_id) AS site_id,
               ANY_VALUE(s.city) AS city,
               ANY_VALUE(s.country_canonical) AS country,
               AVG(a.mean_demand) AS mean_demand,
               MAX(a.peak_demand) AS peak_demand,
               -- Two different load factors, and questions ask for both. The
               -- annual one divides the period's mean by its single highest
               -- hour, so one spike depresses a whole year. The daily one is
               -- each day's own mean over its own peak, averaged: it says how
               -- flat a typical day is, which is what "load factor" usually
               -- means to an operator. 0.457 and 0.762 for the same offices.
               AVG(a.mean_demand / NULLIF(a.peak_demand, 0)) AS daily_load_factor,
               SUM(a.total_kwh) AS total_kwh,
               SUM(a.n_observations) AS observed_hours
        FROM agg_daily a
        JOIN buildings b ON b.building_id = a.building_id
        JOIN sites s ON s.site_id = b.site_id
        WHERE a.grain = 'daily' AND {where}
        GROUP BY a.building_id
        HAVING AVG(a.coverage_fraction) >= ?
    """, params)


# A baseload is a level, not a sum: the demand a building never drops below,
# estimated as a low percentile of its own hourly readings. It comes from the
# raw record because no aggregate carries a percentile, and it carries no
# coverage floor -- a level is not understated by a short record the way a
# total is, which is the same reason a peak carries none.
BASELOAD_METRICS = ("baseload_per_area", "overnight_baseload_per_area")


def _baseload_frame(ctx: ToolContext, building_ids: list[str] | None,
                    year: int | None, percentile: float = 0.05,
                    overnight: bool = False) -> pd.DataFrame:
    """Per-building low-percentile demand joined to metadata.

    ``overnight`` reads the level between midnight and 05:00 instead, which is
    the other reading of "baseload": a floor observed in the hours a building
    is least likely to be occupied, rather than inferred from the shape of the
    whole day.
    """
    params: list = [percentile]
    where = "e.quality_flag = 'OK'" + in_clause("e.building_id", building_ids, params)
    if year:
        # e.timestamp does not exist -- the columns are timestamp_utc and
        # timestamp_local, and the snapshot carries a precomputed `year`
        # beside them. Every baseload metric scoped to a year raised
        # BinderException from here, so "which buildings are above the 90th
        # percentile of nighttime baseload in 2017" had no working route at
        # all, in any tool that asked for one.
        where += " AND e.year = ?"; params.append(year)
    return ctx.db.df(f"""
        SELECT e.building_id,
               ANY_VALUE(b.building_type_canonical) AS building_type,
               -- Carried so `across` can group by them. They cost nothing to
               -- select and grouping by a column the frame does not hold is a
               -- KeyError at the end of a long scan.
               ANY_VALUE(b.building_subtype) AS building_subtype,
               ANY_VALUE(b.industry) AS industry,
               ANY_VALUE(b.subindustry) AS subindustry,
               ANY_VALUE(b.floor_area_m2) AS floor_area_m2,
               ANY_VALUE(b.occupants) AS occupants,
               ANY_VALUE(b.year_built) AS year_built,
               ANY_VALUE(s.city) AS city,
               ANY_VALUE(s.country_canonical) AS country,
               ANY_VALUE(b.site_id) AS site_id,
               QUANTILE_CONT(e.value, ?) AS baseload,
               AVG(e.value) FILTER (
                   WHERE EXTRACT(hour FROM e.timestamp_local) BETWEEN 0 AND 4)
                   AS overnight_baseload,
               COUNT(*) AS n_observations
        FROM electricity e
        JOIN buildings b ON b.building_id = e.building_id
        JOIN sites s ON s.site_id = b.site_id
        WHERE {where}
        GROUP BY e.building_id
    """, params)


SEASONAL_METRICS = ("seasonal_cv",)


def _readable(value: float) -> str:
    """A figure at the precision its magnitude deserves."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:,.1f}"
    if magnitude >= 1:
        return f"{value:,.2f}"
    return f"{value:,.4g}"


def _seasonal_frame(ctx: ToolContext, building_ids: list[str] | None,
                    year: int | None, min_coverage: float) -> pd.DataFrame:
    """A building's twelve-month climatology, summarised as one number.

    Seasonal variation is not the spread of a building's readings -- that is
    dominated by the daily cycle. It is the spread of its *monthly* levels:
    the mean demand in each calendar month, averaged over the years on record
    so a single hot summer does not stand for the season, and the coefficient
    of variation over those twelve values.

    A month observed for a few days is not that month's level, so months below
    half coverage are dropped, and a building without all twelve is left out
    rather than compared on a partial year.
    """
    params: list = [max(0.5, float(min_coverage or 0.0))]
    where = "a.coverage_fraction >= ? AND a.mean_demand IS NOT NULL"
    where += in_clause("a.building_id", building_ids, params)
    if year:
        where += " AND EXTRACT(year FROM a.period_start) = ?"; params.append(year)
    monthly = ctx.db.df(f"""
        SELECT a.building_id,
               ANY_VALUE(b.building_type_canonical) AS building_type,
               ANY_VALUE(b.building_subtype) AS building_subtype,
               ANY_VALUE(b.industry) AS industry,
               ANY_VALUE(b.subindustry) AS subindustry,
               ANY_VALUE(b.floor_area_m2) AS floor_area_m2,
               ANY_VALUE(b.occupants) AS occupants,
               ANY_VALUE(b.site_id) AS site_id,
               ANY_VALUE(s.city) AS city,
               ANY_VALUE(s.country_canonical) AS country,
               EXTRACT(month FROM a.period_start) AS month,
               AVG(a.mean_demand) AS monthly_mean
        FROM agg_monthly a
        JOIN buildings b ON b.building_id = a.building_id
        JOIN sites s ON s.site_id = b.site_id
        WHERE {where}
        GROUP BY a.building_id, EXTRACT(month FROM a.period_start)
    """, params)
    if monthly.empty:
        return monthly
    keep = [c for c in ("building_type", "building_subtype", "industry",
                        "subindustry", "floor_area_m2", "occupants", "site_id",
                        "city", "country") if c in monthly.columns]
    meta = monthly.groupby("building_id")[keep].first().reset_index()
    stats = (monthly.groupby("building_id")["monthly_mean"]
             .agg(months_observed="size",
                  seasonal_cv=lambda v: (v.std(ddof=1) / v.mean()
                                         if len(v) > 1 and v.mean() else None),
                  highest_month="max", lowest_month="min")
             .reset_index())
    stats = stats[stats["months_observed"] >= 12]
    out = meta.merge(stats, on="building_id", how="inner")
    out["month_ratio"] = (out["highest_month"] / out["lowest_month"]).round(4)
    return out.dropna(subset=["seasonal_cv"])


def _frame_for(ctx: ToolContext, metric: str, building_ids: list[str] | None,
               year: int | None, min_coverage: float) -> pd.DataFrame:
    """The frame a metric is computed from.

    Three metrics do not live in the monthly aggregate: load factor and
    peak-to-average describe a shape and are read from the daily grain, and a
    baseload is a percentile of the hourly record. Choosing the frame in one
    place is what lets any tool here rank or compare on any of them, rather
    than each tool knowing about the metrics it happened to be written for.
    """
    if metric in ("load_factor", "peak_to_average_ratio", "daily_load_factor"):
        return _shape_frame(ctx, building_ids, year, min_coverage)
    if metric in SEASONAL_METRICS:
        return _seasonal_frame(ctx, building_ids, year, min_coverage)
    if metric in BASELOAD_METRICS:
        return _baseload_frame(ctx, building_ids, year,
                               overnight=metric == "overnight_baseload_per_area")
    return _performance_frame(ctx, building_ids, year, min_coverage, metric)


@tool("performance")
def compare_building_groups(
    ctx: ToolContext,
    group_a: list[str],
    group_b: list[str],
    metric: Literal["eui", "peak_per_area", "per_occupant", "total_kwh",
                    "mean_demand", "load_factor", "peak_to_average_ratio",
                    "baseload_per_area"] = "eui",
    label_a: str = "group_a",
    label_b: str = "group_b",
    # "Compare load factor and peak-to-average between offices and education"
    # names two measures and one comparison. With a single `metric` the plan
    # had to pick one and the answer reported whichever it picked, silently
    # dropping the other half of the question.
    metrics: list[Literal["eui", "peak_per_area", "peak_power_density",
                          "eui_per_operating_hour", "per_occupant", "total_kwh",
                          "mean_demand", "peak_demand", "load_factor",
                          "peak_to_average_ratio", "baseload_per_area",
                          "overnight_baseload_per_area", "daily_load_factor",
                          "seasonal_cv"]] | None = None,
    statistic: Literal["auto", "median", "total", "mean"] = "auto",
    year: int | None = None,
    min_coverage: float | None = None,
    eligibility: Literal["fit", "structural"] = "fit",
) -> ToolResult:
    """Compare two cohorts on one metric, with distribution summaries.

    ``min_coverage`` defaults by metric: 0.9 for the intensities, whose
    denominator is a year the building has to have had, and 0 for totals,
    levels and shapes, where a floor removes population members rather than
    unfit ones. Pass a value to override either.

    load_factor and peak_to_average_ratio describe the *shape* of a building's
    demand and are read from the daily aggregate: averaging the monthly means
    weights a 28-day month like a 31-day one, which moved the education/office
    load-factor ratio from 0.918 to 0.920. Shape needs no floor area or
    occupancy, so these two compare cohorts that share no metadata at all.

    Each group is screened by the capability the metric needs, the same gate
    calculate_normalized_performance applies, so the two cohorts are compared
    on the same footing however they were resolved. Without it a cohort
    resolved by a plain catalog filter carried buildings whose record is too
    short to normalise -- 519 pre-1980 buildings against the 425 that can be
    scored -- and the comparison moved with them.
    """
    # No coverage floor by default, for any metric. Comparing populations is
    # a sum over each, and both groups are already screened by the capability
    # the metric needs. A second screen removes population members rather than
    # unfit ones, which understates whichever side loses more of them.
    if min_coverage is None:
        # The docstring above has always said 0.9 for the intensities and the
        # code said 0.0, so the comparison screened on the capability gate
        # instead -- a different notion of "enough data" that excludes
        # buildings the metric can be computed for and keeps some it cannot
        # place. Office against education came out 111.6 vs 101.9 over 236 and
        # 494 buildings; on the floor the metric actually needs it is 121.8 vs
        # 115.1 over 262 and 501.
        min_coverage = 0.0 if metric in EXTREME_METRICS else 0.9
        # One screen, not two: where the floor does the screening, the cohort
        # is everyone the metric can be computed for.
        if min_coverage and eligibility == "fit":
            eligibility = "structural"
    # An intensity is per year wherever it is computed. Comparing two cohorts
    # on EUI over the whole eighteen-month record is not a comparison of
    # annual intensities: offices against education came out 1.105 where the
    # 2017 ratio is 1.0948.
    if year is None and metric in PER_YEAR_METRICS:
        year = LAST_FULL_YEAR
    # Several measures, one comparison: each is scored on its own through the
    # ordinary path and the results are stacked, so a reader sees both groups
    # on every measure rather than one measure chosen for them.
    wanted = [m for m in (metrics or []) if m] or [metric]
    if len(wanted) > 1:
        own = globals()["compare_building_groups"]
        run = getattr(own, "fn", own)
        frames, notes = [], []
        for name in wanted:
            part = run(ctx, group_a=group_a, group_b=group_b, metric=name,
                       statistic=statistic, year=year, min_coverage=min_coverage,
                       label_a=label_a, label_b=label_b, eligibility=eligibility)
            if part.status is Status.OK or part.handle:
                frame = ctx.store.get(part.handle) if part.handle else None
                if frame is not None:
                    frames.append(frame.assign(metric=name))
                    head = (part.evidence or {}).get("headline") or {}
                    if head.get("label"):
                        notes.append(head["label"])
        if not frames:
            return insufficient("no_metric_scored",
                                f"neither group could be scored on any of {wanted}")
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[["metric"] + [c for c in combined.columns if c != "metric"]]
        return frame_result(
            combined, ctx.store, name="group_comparison",
            evidence={"metrics": wanted, "groups": [label_a, label_b],
                      "year": year, "statistic": statistic,
                      "per_metric": notes,
                      "headline": {"label": "; ".join(notes) or
                                   f"{label_a} vs {label_b} on {', '.join(wanted)}",
                                   "value": None, "unit": None}})

    rows, ineligible = [], {}
    for label, ids in ((label_a, group_a), (label_b, group_b)):
        eligible, dropped = _eligible_for(ctx, metric, ids, eligibility)
        if dropped:
            ineligible[label] = dropped
        if not eligible:
            gate = ELIGIBILITY.get(metric)
            return insufficient(
                gate or "empty_group",
                f"{label} holds no building that can be scored on {metric}"
                + (f": the snapshot marks {gate} false for all "
                   f"{dropped} of them" if gate and dropped else ""))
        scored = _apply_metric(_frame_for(ctx, metric, eligible, year, min_coverage),
                               metric)
        if scored.empty:
            return insufficient("group_not_scored",
                                f"{label} could not be scored on {metric}")
        v = scored["metric_value"]
        rows.append({"group": label, "n": int(v.size),
                     # "How does the total consumption of one group compare to
                     # the other" is a question about the sums, and this
                     # reported only the middle of each distribution. A cohort
                     # of 163 large buildings against 820 small ones differs in
                     # count as well as in size, which the medians hide.
                     "total": float(v.sum()),
                     "mean": float(v.mean()), "median": float(v.median()),
                     "p25": float(v.quantile(0.25)), "p75": float(v.quantile(0.75)),
                     "min": float(v.min()), "max": float(v.max())})
    out = pd.DataFrame(rows)
    ratio = out.loc[0, "median"] / out.loc[1, "median"] if out.loc[1, "median"] else None
    total_ratio = (out.loc[0, "total"] / out.loc[1, "total"]
                   if out.loc[1, "total"] else None)
    # "Compare A versus B" is answered by the ratio, and a table of two
    # distributions leaves the reader to divide two numbers out of eighteen.
    # The tool names the one it was asked for.
    # Which ratio answers depends on the measure. A total is a sum over the
    # cohort, so "how does the consumption of one group compare to the other"
    # is the ratio of the sums; an intensity or a level is a per-building
    # figure, and the middle of each distribution is what compares.
    # "Median annual electricity consumption" and "total annual electricity
    # consumption" are different questions about the same metric, and the tool
    # answered both with the sum: asked for the median it compared 3.5e8 with
    # 8.0e8 where the medians are 555,220 and 878,775. "auto" keeps the old
    # reading -- a total for a total, the middle building for an intensity --
    # and a named statistic overrides it.
    sums = metric == "total_kwh" if statistic == "auto" else statistic == "total"
    chosen = total_ratio if sums else ratio
    # A ratio on its own is not the answer to "how does X compare with Y": the
    # question asks for both figures, and 1.023 tells a reader neither of them.
    # The two values lead and the ratio follows, because a reader who wants
    # the ratio can see it and one who wants the levels could not recover them.
    column = ("total" if sums
              else statistic if statistic in ("median", "mean") else "median")
    pair = {}
    if column in out.columns and "group" in out.columns:
        for row in out.itertuples():
            pair[str(getattr(row, "group"))] = float(getattr(row, column))
    if len(pair) == 2:
        left, right = pair.get(label_a), pair.get(label_b)
        if left is None or right is None:
            (left, right) = list(pair.values())
        unit = METRICS[metric][1]
        headline = {
            "label": f"{column} {metric.replace('_', ' ')}: {label_a} "
                     f"{left:,.4g} vs {label_b} {right:,.4g} {unit}"
                     + (f" ({chosen:.4g}x)" if chosen is not None else ""),
            "value": round(float(left), 4), "unit": unit}
    else:
        headline = ({"label": f"{column} {metric.replace('_', ' ')}, "
                              f"{label_a} / {label_b}",
                     "value": round(float(chosen), 4), "unit": "ratio"}
                    if chosen is not None else None)
    return frame_result(out, ctx.store, name="group_comparison",
                        evidence={"metric": metric, "unit": METRICS[metric][1],
                                  "median_ratio": ratio,
                                  "total_ratio": total_ratio, "year": year,
                                  "ratio_is": f"{label_a} / {label_b}",
                                  "ineligible_excluded": ineligible or None,
                                  **({"headline": headline} if headline else {})})


@tool("performance", analyst_only=("exclude_outliers_above",))
def rank_buildings(
    ctx: ToolContext,
    building_ids: list[str],
    metric: Literal["eui", "peak_per_area", "peak_power_density",
                    "eui_per_operating_hour", "baseload_per_area",
                    "per_occupant", "total_kwh", "mean_demand",
                    "peak_demand", "load_factor",
                    "peak_to_average_ratio"] = "eui",
    order: Literal["desc", "asc"] = "desc",
    limit: int = 10,
    year: int | None = None,
    min_coverage: float | None = None,
    exclude_outliers_above: float | None = None,
) -> ToolResult:
    """Rank buildings by a metric.

    The ranking is unfiltered by default and reports what it is unsure of rather
    than quietly removing it. Extreme normalised values in this snapshot come
    from implausible *denominators* -- a parking structure recorded as 36 m2 --
    not from implausible consumption, and coverage is near-perfect on every one
    of them, so no quality predicate catches them. Such rows are flagged in
    ``questionable_denominator`` and counted in the evidence; they stay in the
    ranking.

    ``min_coverage`` defaults by direction: 0 for "highest", 0.9 for "lowest".
    A short record understates a total, so it cannot promote a building into a
    top ten -- only keep one out -- while it is exactly what puts a building at
    the bottom of one. Coverage is measured over the period asked for rather
    than over its worst month: eight of the ten highest kWh/person buildings
    are 95% complete across 2017 with one month at 71%, and a floor on the
    worst month threw all eight away.

    ``exclude_outliers_above`` drops rows above a value of the metric. It exists
    for an analyst who has a reason for a specific ceiling. Do not set it to
    make a ranking look tidier: a ceiling on kWh/m2 cannot tell a mis-recorded
    floor area from a genuine heavy consumer, and choosing one silently changes
    which buildings are returned.
    """
    building_ids, ineligible = _eligible_for(ctx, metric, building_ids)
    if ineligible and not building_ids:
        return insufficient(
            ELIGIBILITY[metric],
            f"none of the {ineligible} requested building(s) are eligible for "
            f"{metric}: the snapshot marks {ELIGIBILITY[metric]} false for all")
    # A coverage floor cuts differently at the two ends of a ranking, because
    # a partial record understates a sum. At the top it can only remove true
    # positives: a building that reaches the highest ten on nine months would
    # rank higher still on twelve, so it has earned its place. At the bottom it
    # is essential, because one month of data is the cheapest way to look like
    # the lowest consumer in the estate. So the default follows the direction
    # asked for, and an explicit value is honoured either way.
    if year is None and metric in PER_YEAR_METRICS:
        year = LAST_FULL_YEAR
    if min_coverage is None:
        min_coverage = 0.0 if order == "desc" else 0.9
    scored = _apply_metric(_frame_for(ctx, metric, building_ids, year, min_coverage),
                           metric)
    if scored.empty:
        return insufficient("nothing_scored",
                            f"no requested building could be scored on {metric}")
    excluded = 0
    if exclude_outliers_above is not None:
        before = len(scored)
        scored = scored[scored["metric_value"] <= exclude_outliers_above]
        excluded = before - len(scored)
        if scored.empty:
            return empty(f"every building exceeded the outlier ceiling "
                         f"{exclude_outliers_above}")
    scored = scored.sort_values("metric_value", ascending=(order == "asc"))
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["questionable_denominator"] = _questionable_denominator(scored, metric)
    columns = ["rank", "building_id", "building_type", "city", "country",
               "floor_area_m2", "metric_value", "questionable_denominator"]
    if metric == "eui_per_operating_hour":
        # The inferred divisor travels with the number it produced: a building
        # that never turns off reads as operating 24 hours, and its intensity
        # is low for that reason and not for an operational one.
        columns.insert(-2, "operating_hours_per_day")
    out = scored[[c for c in columns if c in scored.columns]].head(limit)
    out = _name_metric_column(out, metric)
    return frame_result(out, ctx.store, name=f"ranking_{metric}",
                        evidence={"metric": metric, "unit": METRICS[metric][1],
                                  "order": order, "candidates_scored": len(scored),
                                  "questionable_denominator": int(
                                      out["questionable_denominator"].sum()),
                                  "outliers_excluded": excluded,
                                  "outlier_ceiling": exclude_outliers_above,
                                  "ineligible_excluded": ineligible or None,
                                  "year": year, "min_coverage": min_coverage})


def _band_outliers(ctx: ToolContext, building_ids: list[str], metric: str,
                   threshold: float, band_factor: float, min_band_size: int,
                   side: str, year: int | None, min_coverage: float) -> ToolResult:
    """Each building's standing among the others of roughly its own size."""
    scored = _apply_metric(
        _frame_for(ctx, metric, building_ids, year, min_coverage), metric)
    usable = scored.dropna(subset=["floor_area_m2", "metric_value"])
    usable = usable[usable["floor_area_m2"] > 0].reset_index(drop=True)
    if len(usable) < 5:
        return insufficient("cohort_too_small",
                            f"{len(usable)} building(s) carry both {metric} and "
                            f"a floor area; a band needs more than that")
    areas = usable["floor_area_m2"].to_numpy(float)
    values = usable["metric_value"].to_numpy(float)
    rows = []
    for position, (area, value) in enumerate(zip(areas, values)):
        band = (areas >= area / band_factor) & (areas <= area * band_factor)
        # The building belongs to its own band, so the band has to be big
        # enough to say anything: a band of three is not a comparison, it is
        # a coincidence. Ten, including the building itself, is the line.
        if band.sum() < min_band_size:
            continue
        band[position] = False
        neighbours = values[band]
        # ...and it counts in its own denominator, which is what makes the
        # figure a percentile of the band rather than of its neighbours.
        rows.append({"building_id": usable["building_id"].iloc[position],
                     "building_type": usable["building_type"].iloc[position],
                     "floor_area_m2": float(area), metric: float(value),
                     "band_size": int(band.sum()) + 1,
                     "band_median": float(np.median(neighbours)),
                     "percentile_in_band": round(
                         float((neighbours < value).sum())
                         / (len(neighbours) + 1) * 100.0, 2)})
    if not rows:
        return insufficient("no_bands",
                            f"no building has two peers within a factor of "
                            f"{band_factor:g} of its floor area")
    frame = pd.DataFrame(rows)
    high, low = 100.0 - float(threshold), float(threshold)
    if side == "high":
        kept = frame[frame["percentile_in_band"] >= high]
    elif side == "low":
        kept = frame[frame["percentile_in_band"] <= low]
    else:
        kept = frame[(frame["percentile_in_band"] >= high)
                     | (frame["percentile_in_band"] <= low)]
    kept = kept.sort_values("percentile_in_band", ascending=(side == "low"))
    evidence = {"metric": metric, "peers": "size_band",
                "band_factor": band_factor, "side": side,
                "threshold_pct": threshold, "year": year,
                "min_coverage": min_coverage,
                "cohort_scored": int(len(frame)), "n_outliers": int(len(kept)),
                "definition": f"each building against those within a factor of "
                              f"{band_factor:g} of its floor area"}
    if kept.empty:
        return ok({"outliers": []}, evidence=evidence)
    return frame_result(kept, ctx.store, name="peer_band_outliers",
                        evidence=evidence)


#: Peer groupings that are a column of the performance frame, as opposed to
#: "cohort" (everyone together) and "size_band" (a floor-area neighbourhood).
PEER_COLUMNS = {"building_type": "building_type", "site": "site_id",
                "city": "city", "country": "country",
                "building_subtype": "building_subtype"}


def _flag_outliers(v: pd.Series, method: str, threshold: float,
                   side: str = "both") -> tuple:
    """score, flag and the bounds they came from, for one comparison set.

    ``side`` is honoured here because "above the 90th percentile" is one tail
    and returning both answers a different question: on this snapshot the two
    tails are 306 buildings where the high one is 136, so the count is more
    than double and every building in the bottom tenth is reported as if it
    were among the heaviest.
    """
    if method == "zscore":
        std = v.std(ddof=1)
        score = (v - v.mean()) / std if std else v * 0
        high, low = score > threshold, score < -threshold
        bounds = {"mean": float(v.mean()), "std": float(std)}
    elif method == "iqr":
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        score = (v - v.median()) / (iqr if iqr else 1)
        high, low = v > hi, v < lo
        bounds = {"q1": float(q1), "q3": float(q3),
                  "lower": float(lo), "upper": float(hi)}
    else:
        # percentile: threshold is the tail width in percent, so 10 keeps the
        # highest and lowest tenth.
        lo, hi = v.quantile(threshold / 100), v.quantile(1 - threshold / 100)
        score = v.rank(pct=True)
        high, low = v > hi, v < lo
        bounds = {"lower": float(lo), "upper": float(hi)}
    flag = high if side == "high" else low if side == "low" else (high | low)
    return score, flag, bounds


def _persistent_outliers(ctx, building_ids, metric, method, threshold, peers,
                         band_factor, min_band_size, side, years, min_years,
                         min_coverage) -> ToolResult:
    """Outliers in at least ``min_years`` of the named years.

    Scores each year independently through the ordinary path, then keeps the
    buildings flagged often enough and reports which years each was flagged
    in -- so a reader can see whether "persistent" meant two years out of two
    or two out of five.
    """
    seen: dict[str, list[int]] = {}
    per_year: dict[str, int] = {}
    for one in years:
        result = identify_peer_outliers.__wrapped__(
            ctx, building_ids=building_ids, metric=metric, method=method,
            threshold=threshold, peers=peers, band_factor=band_factor,
            min_band_size=min_band_size, side=side, year=one,
            min_coverage=min_coverage)
        frame = ctx.store.get_or_none(getattr(result, "handle", None))
        flagged = [] if frame is None else frame["building_id"].tolist()
        per_year[str(one)] = len(flagged)
        for b in flagged:
            seen.setdefault(b, []).append(one)
    kept = {b: ys for b, ys in seen.items() if len(ys) >= min_years}
    evidence = {"metric": metric, "method": method, "threshold": threshold,
                "peers": peers, "years": years, "min_years": min_years,
                "outliers_per_year": per_year,
                "n_persistent": len(kept),
                "n_ever_flagged": len(seen)}
    if not kept:
        return ok({"outliers": []}, evidence=evidence)
    out = pd.DataFrame(
        [{"building_id": b, "years_flagged": len(ys),
          "years": ", ".join(str(y) for y in ys)}
         for b, ys in sorted(kept.items(), key=lambda kv: (-len(kv[1]), kv[0]))])
    evidence["headline"] = {
        "label": f"buildings flagged in at least {min_years} of "
                 f"{len(years)} years",
        "value": len(kept), "unit": "buildings"}
    return frame_result(out, ctx.store, name="persistent_outliers",
                        evidence=evidence)


@tool("performance")
def identify_peer_outliers(
    ctx: ToolContext,
    building_ids: list[str],
    # The three this was written for, not the three it can do. _frame_for
    # resolves the frame any of these needs, so restricting the enum only
    # meant "which offices are above the 90th percentile of nighttime
    # baseload" had no tool, while the identical question about EUI did.
    metric: Literal["eui", "peak_per_area", "peak_power_density",
                    "eui_per_operating_hour", "per_occupant", "total_kwh",
                    "mean_demand", "peak_demand", "load_factor",
                    "peak_to_average_ratio", "baseload_per_area",
                    "overnight_baseload_per_area", "seasonal_cv",
                    "daily_load_factor"] = "eui",
    method: Literal["zscore", "iqr", "percentile"] = "iqr",
    threshold: float = 1.5,
    peers: Literal["cohort", "size_band", "building_type", "building_subtype",
                   "site", "city", "country"] = "cohort",
    band_factor: float = 1.25,
    min_band_size: int = 10,
    side: Literal["both", "high", "low"] = "both",
    year: int | None = None,
    years: list[int] | None = None,
    min_years: int = 1,
    min_coverage: float | None = None,
    eligibility: Literal["fit", "structural"] = "fit",
) -> ToolResult:
    """Find cohort members that differ materially from their peers.

    ``peers="size_band"`` compares each building against the others within
    ``band_factor`` of its floor area either way, rather than against the whole
    cohort. "A top-10% EUI compared to others of similar floor area" is that
    question: a 500 m2 school and a 50,000 m2 one are not peers, and ranking
    them together answers about size as much as about intensity. A band rather
    than a quantile bucket, because a bucket boundary splits buildings that are
    genuinely comparable.

    ``side`` keeps the high tail, the low tail or both. "Top 10%" is one tail,
    and reporting both answers a question about the heaviest with a list that
    also holds the lightest.
    """
    # "A persistent outlier across at least two complete years" is not the
    # same building as "an outlier in the latest year": a one-off bad year
    # reads identically in a single-year answer, and the question is asking
    # which buildings are reliably heavy. Each year is scored on its own and
    # the buildings flagged in enough of them are kept.
    if years:
        return _persistent_outliers(
            ctx, building_ids, metric, method, threshold, peers, band_factor,
            min_band_size, side, sorted({int(y) for y in years}),
            max(1, int(min_years)), min_coverage)
    if year is None and metric in PER_YEAR_METRICS:
        year = LAST_FULL_YEAR
    # No coverage floor by default. The cohort handed in was resolved by the
    # capability the metric needs, which is what encodes "has enough data"; a
    # second screen on top thins the bands rather than the unfit, and it cost
    # four of the forty-five buildings that stand out against their size peers.
    if min_coverage is None:
        # The same convention calculate_normalized_performance follows, and for
        # the same reason: None means "the floor this metric needs", not "no
        # floor". A total or an intensity is deflated by a short record, so
        # comparing buildings on it uses the 0.9 fitness bar; an extreme can
        # only be understated by one, so it uses none. This said 0.0 for every
        # metric, which let part-year buildings into a peer comparison and
        # reported 156 buildings above their category's 90th percentile where
        # 136 clear it on records long enough to mean anything.
        min_coverage = 0.0 if metric in EXTREME_METRICS else 0.9
        # One screen, not two -- the convention compare_building_groups and
        # calculate_normalized_performance both follow. Where the floor does
        # the screening, the cohort is everyone the metric can be computed
        # for; leaving the capability gate on as well applies a second
        # coverage verdict scoped to the whole record rather than the year in
        # question. The 90th-percentile office EUI came out 272.0 over 229
        # buildings against 279.3 over 262 on the floor the metric needs, and
        # the count above both that percentile and the nighttime baseload one
        # went from 25 to 20.
        if min_coverage and eligibility == "fit":
            eligibility = "structural"
    # The same fit/structural choice the other performance tools take: ranking
    # buildings against their peers wants the fitness bar, counting how many
    # clear a percentile wants everyone the metric can be computed for.
    building_ids, ineligible = _eligible_for(ctx, metric, building_ids,
                                             eligibility)
    if peers == "size_band":
        return _band_outliers(ctx, building_ids, metric, threshold, band_factor,
                              min_band_size, side, year, min_coverage)
    scored = _apply_metric(_frame_for(ctx, metric, building_ids, year, min_coverage),
                           metric)
    if len(scored) < 5:
        return insufficient("cohort_too_small",
                            f"only {len(scored)} buildings scored; need at least 5")
    v = scored["metric_value"].astype(float)
    group_column = PEER_COLUMNS.get(peers)
    per_group: dict = {}
    if group_column:
        # "Above the 90th percentile within its own use category" compares each
        # building only against its own kind. Run against the whole cohort it
        # answers a different question, and on a mixed portfolio the answer is
        # mostly which categories are intensive rather than which buildings
        # are.
        if group_column not in scored.columns:
            return insufficient(
                "unknown_peer_grouping",
                f"the performance frame has no {group_column!r} column")
        score = pd.Series(index=scored.index, dtype=float)
        flag = pd.Series(False, index=scored.index)
        for name, part in scored.groupby(group_column):
            values = part["metric_value"].astype(float)
            if len(values) < min_band_size:
                # Too few peers to place a building among them. Skipped and
                # counted, rather than folded into the cohort comparison the
                # caller asked not to use.
                per_group[str(name)] = {"n": int(len(values)), "skipped": True}
                continue
            sc, fl, bounds = _flag_outliers(values, method, threshold, side)
            score.loc[part.index] = sc
            flag.loc[part.index] = fl
            per_group[str(name)] = {"n": int(len(values)),
                                    "n_outliers": int(fl.sum()), **bounds}
        bounds = {"per_group": per_group}
    else:
        score, flag, bounds = _flag_outliers(v, method, threshold, side)

    out = scored.assign(score=score, is_outlier=flag)
    keep = ["building_id", "building_type", "city", "metric_value", "score"]
    if group_column and group_column not in keep:
        keep.insert(1, group_column)
    out = out.loc[out["is_outlier"].fillna(False), keep]
    out = _name_metric_column(out, metric).sort_values("score",
                                                       ascending=False)
    evidence = {"metric": metric, "method": method, "threshold": threshold,
                "peers": peers,
                "cohort_size": int(len(scored)), "n_outliers": int(len(out)),
                "bounds": bounds, "cohort_median": float(v.median())}
    if group_column:
        evidence["peer_groups"] = len(per_group)
        evidence["min_peers_per_group"] = min_band_size
    if out.empty:
        return ok({"outliers": []}, evidence=evidence)
    return frame_result(out, ctx.store, name="peer_outliers", evidence=evidence)


# How much of a building's consumption moves with the weather, as the absolute
# correlation between its daily total and the site's daily mean outdoor
# temperature. The sign is dropped deliberately: a heating-led building
# correlates negatively and a cooling-led one positively, and both are equally
# weather-driven, which is what the index is crediting.
WEATHER_SENSITIVITY_SQL = """
    WITH elec AS (
        SELECT e.building_id, e.site_id, CAST(e.timestamp_local AS DATE) AS d,
               SUM(e.value) AS v
        FROM electricity e
        WHERE e.quality_flag = 'OK' {year_filter}
        GROUP BY 1, 2, 3),
         wx AS (
        SELECT w.site_id, CAST(w.timestamp_local AS DATE) AS d,
               AVG(w.air_temperature) AS t
        FROM weather w GROUP BY 1, 2)
    SELECT elec.building_id,
           ABS(CORR(elec.v, wx.t)) AS weather_sensitivity,
           REGR_SLOPE(elec.v, wx.t) AS weather_slope_kwh_per_degc,
           COUNT(*) AS weather_days
    FROM elec JOIN wx ON wx.site_id = elec.site_id AND wx.d = elec.d
    WHERE elec.v IS NOT NULL AND wx.t IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) >= 30
"""

# The components an index can be built from, and the sign each carries.
#
# A sign is not a weight. Equal weights say the three components matter
# equally; the sign says which way each one points, and without it the index
# adds hours to kWh/m2 and means nothing. Consumption per square metre counts
# against a building. Weather sensitivity and operating hours count *for* it:
# a building whose load tracks the weather, or that runs eighteen hours a day,
# is expected to use more, and an index meant to strip out weather and
# schedule bias has to credit them rather than charge for them.
INDEX_COMPONENTS: dict[str, tuple[str, float, str]] = {
    "eui": ("metric", +1.0, "annual kWh per m2"),
    "baseload_per_area": ("metric", +1.0, "overnight floor per m2"),
    "peak_power_density": ("metric", +1.0, "peak W per m2"),
    "load_factor": ("metric", -1.0, "mean over peak demand; flatter is better"),
    "weather_sensitivity": ("weather", -1.0,
                            "|correlation| of daily use with outdoor temperature"),
    "operating_hours": ("hours", -1.0, "hours a day the building is in use"),
    # How a building stands against the ones it shares a site with, rather
    # than against the portfolio. A 300 kWh/m2 office is unremarkable on a
    # heavy campus and the worst thing on a light one, and a prioritisation
    # that only knows the absolute figure cannot tell those apart.
    "peer_relative_eui": ("relative", +1.0,
                          "EUI over the median EUI of its own site"),
    # How often the building ran far above its own normal. Counted per
    # building against its own mean, so a large building is not flagged for
    # being large.
    "anomaly_count": ("anomaly", +1.0,
                      "hours above the building's own mean + 3 SD"),
}

DEFAULT_INDEX = ("eui", "weather_sensitivity", "operating_hours")


@tool("performance")
def construct_performance_index(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    components: list[Literal["eui", "baseload_per_area", "peak_power_density",
                             "load_factor", "weather_sensitivity",
                             "operating_hours", "peer_relative_eui",
                             "anomaly_count"]] | None = None,
    weights: list[float] | None = None,
    year: int | None = None,
    min_coverage: float | None = None,
    order: Literal["desc", "asc"] = "desc",
    limit: int = 20,
) -> ToolResult:
    """Rank buildings on an index combining several normalised measures.

    Each component is standardised as a z-score across the cohort being
    ranked, multiplied by its sign, and averaged. The default is EUI against
    weather sensitivity and operating hours: what is left of a building's
    intensity once the weather it sits in and the hours it runs are taken into
    account, which is what "normalised performance index" names.

    Equal weights are the default and are reported rather than tuned. A tuned
    weight needs a target to tune against, and this snapshot carries no
    measure of true efficiency to fit to, so a fitted weighting would be an
    arbitrary choice presented as an analytical one. Pass ``weights`` to state
    a different one; it is reported beside the answer either way.

    A z-score is a position within this cohort, so the index says which
    buildings stand out among the ones ranked and carries no meaning outside
    them. Ranking 1,400 buildings and ranking 40 offices produce different
    numbers for the same building, as they should.

    Buildings missing any component are dropped rather than scored on the
    rest, and the count is reported: an index averaged over a different set of
    components per building is not one index.
    """
    chosen = list(components or DEFAULT_INDEX)
    unknown = [c for c in chosen if c not in INDEX_COMPONENTS]
    if unknown:
        return insufficient("unknown_component",
                            f"unknown component(s) {unknown}; valid: "
                            f"{sorted(INDEX_COMPONENTS)}")
    if len(chosen) < 2:
        return insufficient("components",
                            "an index combines at least two measures; one "
                            "measure is a ranking, which rank_buildings does")
    if weights is not None and len(weights) != len(chosen):
        return insufficient("weights",
                            f"{len(weights)} weight(s) for {len(chosen)} "
                            f"component(s); one each")
    if year is None:
        year = LAST_FULL_YEAR
    if min_coverage is None:
        min_coverage = 0.9
    building_ids, ineligible = _eligible_for(ctx, "eui", building_ids)
    if ineligible and not building_ids:
        return insufficient("can_eui",
                            f"none of the {ineligible} requested building(s) "
                            f"can be normalised by floor area")

    frame = None
    for component in chosen:
        kind, _sign, _note = INDEX_COMPONENTS[component]
        if kind == "metric":
            part = _apply_metric(
                _frame_for(ctx, component, building_ids, year, min_coverage),
                component)
            part = part[["building_id", "metric_value"]].rename(
                columns={"metric_value": component})
        elif kind == "relative":
            scored = _apply_metric(
                _frame_for(ctx, "eui", building_ids, year, min_coverage), "eui")
            if "site_id" not in scored.columns:
                return insufficient("peer_relative_eui",
                                    "the frame carries no site to compare within")
            median = scored.groupby("site_id")["metric_value"].transform("median")
            part = scored.assign(peer_relative_eui=scored["metric_value"] / median)
            part = part[["building_id", "peer_relative_eui"]].dropna()
        elif kind == "anomaly":
            part = ctx.db.df(f"""
                WITH stats AS (
                    SELECT building_id, AVG(value) AS mu, STDDEV_SAMP(value) AS sd
                    FROM electricity
                    WHERE quality_flag = 'OK' AND value IS NOT NULL
                      {f"AND year = {int(year)}" if year else ""}
                    GROUP BY building_id)
                SELECT e.building_id,
                       COUNT(*) FILTER (
                           WHERE e.value > s.mu + 3 * s.sd) AS anomaly_count
                FROM electricity e JOIN stats s ON s.building_id = e.building_id
                WHERE e.quality_flag = 'OK' AND e.value IS NOT NULL
                  AND s.sd IS NOT NULL AND s.sd > 0
                  {f"AND e.year = {int(year)}" if year else ""}
                GROUP BY e.building_id""")
        elif kind == "weather":
            part = ctx.db.df(WEATHER_SENSITIVITY_SQL.format(
                year_filter=(f"AND EXTRACT(year FROM e.timestamp_local) = {int(year)}"
                             if year else "")))
            part = part[["building_id", "weather_sensitivity"]]
        else:
            part = ctx.db.df(OPERATING_HOURS_SQL).rename(
                columns={"operating_hours_per_day": "operating_hours"})
        if part.empty:
            return insufficient(component,
                                f"no building could be scored on {component}")
        frame = part if frame is None else frame.merge(part, on="building_id",
                                                       how="inner")
    if building_ids is not None:
        frame = frame[frame["building_id"].isin(set(building_ids))]
    complete = frame.dropna(subset=chosen)
    dropped = len(frame) - len(complete)
    if len(complete) < 3:
        return insufficient(
            "cohort_too_small",
            f"{len(complete)} building(s) carry all of {chosen}; a z-score "
            f"across a cohort needs more than that")

    # Weights are relative: an index is a weighted average of its components,
    # so they are normalised to sum to one. A planner writing 1.0 for each of
    # three components means "equally", not "three times the scale", and the
    # number it produced was three times the index it described.
    given = [float(w) for w in (weights if weights is not None
                                else [1.0] * len(chosen))]
    total = sum(abs(w) for w in given)
    if not total:
        return insufficient("weights", "every weight is zero; nothing is ranked")
    share = [w / total for w in given]
    equal = len({round(abs(w), 9) for w in given}) == 1
    index = np.zeros(len(complete))
    for component, weight in zip(chosen, share):
        values = complete[component].astype(float)
        spread = values.std(ddof=0)
        # A component every building shares tells the ranking nothing, and
        # dividing by its zero spread would put NaN through the whole index.
        z = (values - values.mean()) / spread if spread else values * 0.0
        complete[f"z_{component}"] = z
        index = index + weight * INDEX_COMPONENTS[component][1] * z.to_numpy()
    out = complete.assign(performance_index=index)
    out = out.sort_values("performance_index", ascending=(order == "asc"))
    out["rank"] = np.arange(1, len(out) + 1)
    columns = (["rank", "building_id", "performance_index"] + chosen
               + [f"z_{c}" for c in chosen])
    flagged = 0
    if any(c in AREA_NORMALISED for c in chosen):
        # The extreme end of any area-normalised ranking in this snapshot is
        # made of wrong denominators, not unusual buildings: a parking
        # structure recorded as 36 m2 reaches second place on this index. The
        # row stays and says so.
        out = out.merge(ctx.db.df("SELECT building_id, floor_area_m2 FROM buildings"),
                        on="building_id", how="left")
        out["questionable_denominator"] = _questionable_denominator(out, "eui")
        flagged = int(out["questionable_denominator"].head(limit).sum())
        columns = columns + ["questionable_denominator"]
    return frame_result(out[columns].head(limit), ctx.store,
                        name="performance_index",
                        evidence={
                            "components": chosen,
                            "weights": {c: round(w * INDEX_COMPONENTS[c][1], 4)
                                        for c, w in zip(chosen, share)},
                            "weighting": ("equal, stated rather than tuned"
                                          if equal else
                                          "as passed, normalised to sum to 1"),
                            "standardised": "z-score across the ranked cohort",
                            "definitions": {c: INDEX_COMPONENTS[c][2]
                                            for c in chosen},
                            "order": order,
                            "buildings_ranked": int(len(complete)),
                            "questionable_denominator": flagged,
                            "buildings_dropped": int(dropped),
                            "ineligible_excluded": ineligible or None,
                            "year": year, "min_coverage": min_coverage,
                            "schedule_span": ("the whole record: the daily "
                                              "profile the hours are read from "
                                              "is precomputed over it"),
                            "headline": {
                                # The components are named here because the
                                # text view keeps the columns its headline is
                                # about, and an index is unreadable without
                                # the measures that made it.
                                "label": (("least" if order == "desc" else "most")
                                          + " efficient on an index of "
                                          + ", ".join(chosen)),
                                "value": str(out.iloc[0]["building_id"]),
                                "unit": None}})
