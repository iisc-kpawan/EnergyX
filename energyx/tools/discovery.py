"""Data Discovery & Access tools (8).

Answers "what exists, is it usable, and give me the slice I need". Catalog
lookups are cheap; measurement fetches are not, so the intended order is always
resolve-then-fetch: narrow the entity set against the catalog first, then pull
only those buildings.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ._common import STRUCTURAL
from .base import ToolResult, empty, frame_result, insufficient, ok, partial
from .registry import ToolContext, tool

# task name -> capability column (docs/DATA_MODEL.md capabilities table)
TASK_CAPABILITY = {
    "consumption_analysis": "can_consumption_analysis",
    "eui_analysis": "can_eui",
    "weather_analysis": "can_weather_analysis",
    "occupancy_normalisation": "can_occupancy_normalisation",
    "peer_benchmark": "can_peer_benchmark",
    "forecasting": "can_forecasting",
    "spatial_analysis": "can_spatial_analysis",
}

AGG_VIEW = {"daily": "agg_daily", "monthly": "agg_monthly", "yearly": "agg_yearly"}


def _in_clause(column: str, values: list[str] | None, params: list) -> str:
    """Safe IN clause; DuckDB unnests a list parameter."""
    if not values:
        return ""
    params.append(list(values))
    return f" AND {column} IN (SELECT unnest(?))"


def _eq(column: str, value, params: list) -> str:
    if value is None:
        return ""
    params.append(value)
    return f" AND {column} = ?"


#: Buildings whose electricity *and* whose site's weather both clear a coverage
#: floor inside one calendar year. Cached per (floor, years) because the join
#: is the expensive part.
_YEAR_COVERAGE: dict[tuple, set] = {}


def _complete_year_buildings(ctx: ToolContext, floor: float,
                             years: tuple[int, ...],
                             weather_floor: float | None) -> set:
    """Buildings with a complete year, optionally weather-complete too.

    "A complete calendar year" is a property of a *year*, not of the record as
    a whole: a building 60% covered across two years may hold one clean year
    inside them, and one 80% covered across both holds neither. And "with
    overlapping weather" means the site's thermometer was running in that same
    year -- checked against the year the building qualified on, not against
    the span.
    """
    key = (round(float(floor), 4), years,
           None if weather_floor is None else round(float(weather_floor), 4))
    if key in _YEAR_COVERAGE:
        return _YEAR_COVERAGE[key]
    listed = ", ".join(str(int(y)) for y in years)
    weather_clause = ""
    if weather_floor is not None:
        weather_clause = f"""
              AND EXISTS (
                  SELECT 1 FROM weather w
                  WHERE w.site_id = b.site_id
                    AND EXTRACT(year FROM w.timestamp_local) = elec.y
                  GROUP BY w.site_id
                  HAVING COUNT(*) FILTER (WHERE w.air_temperature IS NOT NULL)
                         >= {float(weather_floor)} * hours.expected)"""
    frame = ctx.db.df(f"""
        WITH hours AS (
            SELECT y, CASE WHEN y % 4 = 0 THEN 8784 ELSE 8760 END AS expected
            FROM (SELECT UNNEST([{listed}]) AS y)
        ), elec AS (
            SELECT e.building_id,
                   EXTRACT(year FROM e.timestamp_local) AS y,
                   COUNT(*) AS n
            FROM electricity e WHERE e.value IS NOT NULL
              AND EXTRACT(year FROM e.timestamp_local) IN ({listed})
            GROUP BY 1, 2
        )
        SELECT DISTINCT b.building_id
        FROM buildings b
        JOIN elec ON elec.building_id = b.building_id
        JOIN hours ON hours.y = elec.y
        WHERE elec.n >= {float(floor)} * hours.expected{weather_clause}
    """)
    _YEAR_COVERAGE[key] = set(frame["building_id"])
    return _YEAR_COVERAGE[key]


#: Twelve *consecutive months* is not a calendar year, and the difference is
#: not academic: a building complete from July 2016 through June 2017 has
#: twelve consecutive months and no complete calendar year, and the calendar
#: rule drops it. A question only means the calendar when it says so -- "one
#: complete calendar year" does, "at least 12 consecutive months" does not --
#: so the two rules are separate parameters rather than one with a default.
#: The most hourly readings one fetch will hold in memory. A year of one
#: building is 8,760; the whole portfolio is 26 million, which is not a
#: fetch, it is an out-of-memory kill. Two hundred buildings for a year sit
#: comfortably under this, and anything larger has an aggregate that answers
#: it.
MAX_FETCH_ROWS = 4_000_000

_ROLLING_YEAR: dict[tuple, set[str]] = {}


def _rolling_year_buildings(ctx: ToolContext, floor: float,
                            window_months: int = 12,
                            weather_floor: float | None = None) -> set[str]:
    """Buildings with a run of ``window_months`` months at or above ``floor``.

    The months have to be adjacent in time: a building missing April is not
    twelve consecutive months either side of it. Coverage is measured against
    the hours those months actually hold, so February and a 31-day month are
    not weighted alike.
    """
    key = (round(float(floor), 4), int(window_months),
           None if weather_floor is None else round(float(weather_floor), 4))
    if key in _ROLLING_YEAR:
        return _ROLLING_YEAR[key]
    monthly = ctx.db.df("""
        SELECT building_id, DATE_TRUNC('month', timestamp_local) AS month,
               COUNT(*) AS n
        FROM electricity WHERE quality_flag = 'OK'
        GROUP BY 1, 2 ORDER BY 1, 2
    """)
    weather = None
    if weather_floor is not None:
        # Weather is a property of the site, so it is counted per site and
        # then mapped onto that site's buildings. Joining first duplicates
        # every reading once per building -- 28.6 million rows on this
        # snapshot, because Rat alone has 305 buildings sharing one weather
        # station -- and the query was killed by the OOM reaper rather than
        # being slow.
        per_site = ctx.db.df("""
            SELECT site_id, DATE_TRUNC('month', timestamp_local) AS month,
                   COUNT(*) FILTER (WHERE air_temperature IS NOT NULL) AS n
            FROM weather GROUP BY 1, 2 ORDER BY 1, 2
        """)
        sites = ctx.db.df("SELECT building_id, site_id FROM buildings")
        weather = sites.merge(per_site, on="site_id")[
            ["building_id", "month", "n"]].sort_values(["building_id", "month"])
    keep: set[str] = set()
    for frame, bar in ((monthly, float(floor)),
                       *(((weather, float(weather_floor)),) if weather is not None else ())):
        passed: set[str] = set()
        for building, group in frame.groupby("building_id"):
            group = group.sort_values("month").reset_index(drop=True)
            stamps = pd.to_datetime(group["month"])
            hours = stamps.dt.days_in_month * 24
            # A month absent from the record breaks the run; so does a month
            # that is present but sits a month or more away from the last one.
            step = (stamps.dt.year * 12 + stamps.dt.month).diff()
            block = (step != 1).cumsum()
            for _, run in group.groupby(block):
                if len(run) < window_months:
                    continue
                present = run["n"].rolling(window_months).sum()
                expected = hours.loc[run.index].rolling(window_months).sum()
                if ((present / expected) >= bar).any():
                    passed.add(building)
                    break
        keep = passed if not keep else (keep & passed)
    _ROLLING_YEAR[key] = keep
    return keep


#: Longest unbroken run of hourly readings per building, computed once.
#: "At least 12 consecutive months with no gap longer than the sampling
#: interval" is a question about an unbroken run, and coverage cannot answer
#: it: a building can be 99% complete over two years and never have a single
#: unbroken year, while another sits at 60% with a clean twelve months inside
#: it. Coverage said 628 buildings qualified; 285 actually do.
_LONGEST_RUNS: dict[str, int] | None = None


def _longest_run_hours(ctx: ToolContext) -> dict[str, int]:
    """building_id -> longest run of consecutive present hourly readings."""
    global _LONGEST_RUNS
    if _LONGEST_RUNS is None:
        frame = ctx.db.df("""
            WITH stamped AS (
                SELECT building_id, timestamp_local,
                       LAG(timestamp_local) OVER (PARTITION BY building_id
                                                  ORDER BY timestamp_local) AS prev
                FROM electricity WHERE value IS NOT NULL
            ), marked AS (
                SELECT building_id, timestamp_local,
                       CASE WHEN prev IS NULL
                                 OR DATE_DIFF('hour', prev, timestamp_local) > 1
                            THEN 1 ELSE 0 END AS breaks
                FROM stamped
            ), grouped AS (
                SELECT building_id, timestamp_local,
                       SUM(breaks) OVER (PARTITION BY building_id
                                         ORDER BY timestamp_local) AS run_id
                FROM marked
            ), runs AS (
                SELECT building_id, run_id, COUNT(*) AS hours
                FROM grouped GROUP BY 1, 2
            )
            SELECT building_id, MAX(hours) AS longest FROM runs GROUP BY 1
        """)
        _LONGEST_RUNS = {row.building_id: int(row.longest or 0)
                         for row in frame.itertuples()}
    return _LONGEST_RUNS


#: Longest run of missing hours *between* a building's first and last valid
#: reading, computed once per process. The snapshot's stored
#: ``longest_gap_hours`` measures something wider -- it counts a record that
#: starts late or stops early as a gap -- and the two disagree by 122
#: buildings at a seven-day threshold: 610 against 488. A building whose
#: meter was installed in March does not have a two-month gap in January; it
#: has a shorter record, which is a different fact and has its own filters.
_INTERNAL_GAPS: dict[str, float] | None = None


def _internal_gap_hours(ctx: ToolContext) -> dict[str, float]:
    """building_id -> longest internal gap in hours."""
    global _INTERNAL_GAPS
    if _INTERNAL_GAPS is None:
        frame = ctx.db.df("""
            WITH stamped AS (
                SELECT building_id, timestamp_local,
                       LAG(timestamp_local) OVER (PARTITION BY building_id
                                                  ORDER BY timestamp_local) AS prev
                FROM electricity WHERE value IS NOT NULL
            )
            SELECT building_id,
                   MAX(DATE_DIFF('hour', prev, timestamp_local) - 1) AS gap
            FROM stamped WHERE prev IS NOT NULL GROUP BY 1
        """)
        _INTERNAL_GAPS = {row.building_id: float(row.gap or 0.0)
                          for row in frame.itertuples()}
    return _INTERNAL_GAPS


@tool("data_discovery")
def search_catalog(
    ctx: ToolContext,
    entity: Literal["datasets", "countries", "cities", "sites",
                    "building_types", "building_subtypes", "industries",
                    "subindustries", "years"],
    country: str | None = None,
    city: str | None = None,
    site_id: str | None = None,
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
    has_weather: bool | None = None,
    limit: int = 200,
) -> ToolResult:
    """Counts grouped by site, city, country, type, year or dataset.

    Aggregates, not members: it answers "how many buildings per country" and
    "which years does this cover" with one row per group. To get the buildings
    themselves, use find_buildings -- this returns a tally and never a list of
    building ids.

    This is the whole answer to "how many X buildings per site" in one call:
    pass ``entity="sites"`` with ``building_type="Education"`` and each row is a
    site with its education-building count. Listing the buildings and tallying
    them separately answers a different question.


    Returns one row per entity with building counts, so "how many buildings per
    country" is a single call here rather than listing buildings and tallying
    them. Replaces separate list_datasets / list_countries / list_cities /
    list_sites tools with one parameterised entry point.

    Whether a site has weather is a catalog fact: every "sites" row carries
    has_weather, and ``has_weather=True`` counts only those. Fetching readings
    to see which sites return any answers a question about download size.
    """
    params: list = []
    where = "1=1"
    where += _eq("s.country_canonical", country, params)
    where += _eq("s.city", city, params)
    # A question naming a site could not be narrowed to it: "how many buildings
    # are at site Rat" listed all 19 sites, leaving the reader to find the row
    #.
    where += _eq("s.site_id", site_id, params)
    where += _eq("b.building_type_canonical", building_type, params)
    if has_weather is not None:
        where += (" AND (s.site_id IN (SELECT site_id FROM weather_sources))"
                  f" = {'TRUE' if has_weather else 'FALSE'}")

    queries = {
        # What is *in* the dataset, not just where it came from. "SELECT *"
        # returned source_url, license and citation -- provenance -- while
        # saying nothing about buildings, sites, span or resolution, which is
        # what "what does this dataset contain" actually asks.
        "datasets": """
            -- Ordered so the text view, which keeps the first and last three
            -- columns, keeps what the question asked: name, size, and span.
            SELECT d.dataset_name,
                   COUNT(DISTINCT b.building_id)  AS buildings,
                   COUNT(DISTINCT b.site_id)      AS sites,
                   COUNT(DISTINCT st.stream_id)   AS electricity_streams,
                   ANY_VALUE(st.unit)             AS unit,
                   d.building_sector, d.dataset_id,
                   d.source_url, d.license, d.citation,
                   MIN(st.start_timestamp)        AS first_reading,
                   MAX(st.end_timestamp)          AS last_reading,
                   ANY_VALUE(st.resolution_minutes) AS resolution_minutes
            FROM datasets d
            LEFT JOIN buildings b ON b.dataset_id = d.dataset_id
            LEFT JOIN electricity_streams st ON st.building_id = b.building_id
            GROUP BY d.dataset_id, d.dataset_name, d.building_sector,
                     d.source_url, d.license, d.citation
            ORDER BY buildings DESC""",
        # "How many buildings are available in each country" is a question
        # about the whole portfolio, and 400 of this snapshot's 1,636
        # buildings sit at sites BDG2 anonymised -- a quarter of it. Dropping
        # them made the answer four countries totalling 1,236 and said
        # nothing about the 400 that went missing. They are a group with an
        # unknown country, not an absence of buildings, so they are reported
        # as one and named for what they are.
        "countries": f"""
            SELECT COALESCE(s.country_canonical, 'Unknown (anonymised site)')
                       AS country,
                   COUNT(DISTINCT s.site_id) AS sites,
                   COUNT(b.building_id) AS buildings
            FROM sites s LEFT JOIN buildings b ON b.site_id = s.site_id
            WHERE {where}
            GROUP BY 1 ORDER BY buildings DESC""",
        "cities": f"""
            SELECT s.city, s.country_canonical AS country,
                   COUNT(DISTINCT s.site_id) AS sites,
                   COUNT(b.building_id) AS buildings
            FROM sites s LEFT JOIN buildings b ON b.site_id = s.site_id
            WHERE {where} AND s.city IS NOT NULL
            GROUP BY 1, 2 ORDER BY buildings DESC""",
        # has_weather is here because "how many sites have weather data" had
        # no catalog answer, and the plan written instead was to fetch every
        # reading for every site and count what came back: 330,717 hourly rows
        # reported as the number of sites. A fact the catalog holds should
        # never have to be inferred from the size of a download.
        "sites": f"""
            SELECT s.site_id, s.site_name, s.city, s.country_canonical AS country,
                   s.timezone, s.utc_offset_hours, COUNT(b.building_id) AS buildings,
                   ANY_VALUE(w.site_id) IS NOT NULL AS has_weather
            FROM sites s LEFT JOIN buildings b ON b.site_id = s.site_id
            LEFT JOIN weather_sources w ON w.site_id = s.site_id
            WHERE {where}
            GROUP BY 1,2,3,4,5,6 ORDER BY buildings DESC""",
        # Which calendar years hold data, one row each. Deriving this from the
        # stream span instead reports a year that does not exist: the last
        # reading is 2018-01-01 07:00 UTC, which is 2017-12-31 local time at the
        # westernmost site. The yearly aggregate is cut on local calendar
        # boundaries, so it answers "what years does this cover" correctly.
        "years": f"""
            SELECT CAST(YEAR(y.period_start) AS INTEGER) AS year,
                   COUNT(DISTINCT y.building_id) AS buildings,
                   COUNT(DISTINCT y.site_id)     AS sites,
                   ROUND(SUM(y.total_kwh), 1)    AS total_kwh
            FROM agg_yearly y
            JOIN buildings b ON b.building_id = y.building_id
            JOIN sites s ON s.site_id = b.site_id
            WHERE {where}
            GROUP BY 1 ORDER BY 1""",
        "building_subtypes": f"""
            SELECT b.building_subtype AS building_subtype,
                   COUNT(*) AS buildings,
                   COUNT(DISTINCT b.site_id) AS sites
            FROM buildings b JOIN sites s USING (site_id)
            WHERE {where} AND b.building_subtype IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC""",
        "industries": f"""
            SELECT b.industry AS industry, COUNT(*) AS buildings,
                   COUNT(DISTINCT b.subindustry) AS subindustries
            FROM buildings b JOIN sites s USING (site_id)
            WHERE {where} AND b.industry IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC""",
        "subindustries": f"""
            SELECT b.subindustry AS subindustry, ANY_VALUE(b.industry) AS industry,
                   COUNT(*) AS buildings
            FROM buildings b JOIN sites s USING (site_id)
            WHERE {where} AND b.subindustry IS NOT NULL
            GROUP BY 1 ORDER BY 3 DESC""",
        "building_types": f"""
            SELECT b.building_type_canonical AS building_type,
                   COUNT(*) AS buildings,
                   ROUND(MEDIAN(b.floor_area_m2), 1) AS median_floor_area_m2
            FROM buildings b JOIN sites s ON s.site_id = b.site_id
            WHERE {where} AND b.building_type_canonical IS NOT NULL
            GROUP BY 1 ORDER BY buildings DESC""",
    }
    df = ctx.db.df(queries[entity] + f" LIMIT {int(limit)}", params or None)
    # Say what was counted. With building_type="Education" the column still read
    # "buildings", so 111 looked like every building at the site rather than its
    # education buildings.
    if building_type and "buildings" in df.columns:
        df = df.rename(columns={"buildings":
                                f"{str(building_type).lower()}_buildings"})
    # How many matched before the limit. A planner asking "how many sites" set
    # limit=1, got one row, and the answer read "1 site" -- a truncation
    # presented as a count.
    matched = int(ctx.db.df(
        f"SELECT COUNT(*) AS n FROM ({queries[entity]})", params or None
    )["n"].iloc[0])
    # "What building-use categories are represented, and how many are there"
    # asks for the groups and a count OF THE GROUPS. The buildings column sits
    # right beside it, so without saying which number is the answer the reply
    # can come back 965 -- the buildings in the United States -- for a question
    # whose answer is 16.
    noun = entity.replace("_", " ")
    return frame_result(df, ctx.store, name=f"catalog_{entity}",
                        evidence={"entity": entity, "matched": matched,
                                  "returned": len(df), "limit": limit,
                                  "truncated_by_limit": matched > len(df),
                                  "headline": {
                                      "label": f"distinct {noun}",
                                      "value": matched, "unit": noun}})


@tool("data_discovery")
def find_buildings(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    country: str | None = None,
    countries: list[str] | None = None,
    city: str | None = None,
    site_id: str | None = None,
    building_type: Literal["Assembly", "Education", "FoodService", "Healthcare", "Industrial",
                       "Lodging", "Office", "Other", "Parking",
                       "PublicServices", "ReligiousWorship", "Retail",
                       "Services", "TechnologyScience", "Utility",
                       "Warehouse"] | None = None,
    # The catalog carries three finer dimensions than the canonical type,
    # and none of them was reachable: building_subtype on 1,615 of 1,636
    # buildings (104 values: Dormitory, Fire Station, College Laboratory),
    # industry and subindustry on the 579 that have them. "Which
    # dormitories", "college laboratories versus classrooms" and
    # "primary/secondary schools" are all questions about the subtype, and
    # answering them with the canonical type conflates a dormitory with a
    # lecture hall because both are Education.
    # "Which buildings have meter data available" is not "which have a meter
    # registered": six streams exist in this snapshot and hold no readings at
    # all, so the two answers differ by six buildings and only one of them is
    # what the question asked.
    has_readings: bool | None = None,
    building_subtype: str | None = None,
    industry: str | None = None,
    subindustry: str | None = None,
    min_floor_area_m2: float | None = None,
    max_floor_area_m2: float | None = None,
    floor_area_decile_at_least: int | None = None,
    floor_area_decile_at_most: int | None = None,
    min_year_built: int | None = None,
    max_year_built: int | None = None,
    requires: list[Literal["consumption_analysis", "eui_analysis",
                           "weather_analysis", "occupancy_normalisation",
                           "peer_benchmark", "forecasting",
                           "spatial_analysis"]] | None = None,
    eligibility: Literal["structural", "fit"] = "structural",
    min_coverage: float | None = None,
    coverage_year: int | None = None,
    has_metadata: list[Literal["floor_area", "year_built", "occupants",
                               "number_of_floors", "building_type"]] | None = None,
    min_months_of_data: int | None = None,
    min_complete_days_in_year: int | None = None,
    min_varying_days_in_year: int | None = None,
    gap_at_most_hours: float | None = None,
    gap_longer_than_hours: float | None = None,
    min_unbroken_hours: int | None = None,
    complete_year_coverage: float | None = None,
    complete_years: list[int] | None = None,
    # "At least 12 consecutive months" rather than "one complete calendar
    # year". Both are twelve months of data and they select different
    # buildings, so the question's own words pick between them.
    consecutive_months_coverage: float | None = None,
    consecutive_months: int = 12,
    weather_coverage_at_least: float | None = None,
    # No cap by default. 500 was a listing convenience and it decided answers:
    # a cohort of 1,572 buildings arrived as the first 500 in catalogue order,
    # and every statistic computed over it described a third of the portfolio
    # while reading like the whole. The result is a handle with a short
    # preview, so a large frame costs a reader nothing -- the rows live in the
    # artifact store, not in the reply. Pass a limit when the question asks
    # for a top-N; otherwise the answer is the dataset.
    limit: int | None = None,
) -> ToolResult:
    """The set of buildings matching attributes, coverage and record length.

    This is the tool for "which buildings ...": location, usage, size, age,
    required capabilities, how complete their data is and how long it runs.
    A question about twelve months of continuous data is answered here, not by
    a data-quality tool -- those describe one stream, this selects the set.

    ``requires`` names capabilities from check_task_eligibility, so an eligible
    set can be resolved in a single call before any measurement is fetched. It
    selects on what a metric needs to be *computable* -- for eui_analysis, a
    positive floor area and a meter with data. Whether a given year is complete
    enough to trust is a separate judgement, made by the analysis tool over the
    year actually asked about; pass ``eligibility="fit"`` to select on the
    snapshot's bundled verdict instead, which also requires enough record
    overall.

    ``min_year_built`` and ``max_year_built`` are both inclusive, so "built
    before 1980" is max_year_built=1979 and "after 2010" is
    min_year_built=2011. Written as the round number each takes in the year
    the question excludes.

    ``min_coverage`` is a floor on the whole record unless ``coverage_year``
    names a year, in which case it is a floor on that year alone.

    ``min_months_of_data`` is how long the stream runs and
    ``gap_at_most_hours`` how broken it may be inside that span. "At least
    12 months of continuous data" needs both, and neither existed: the
    only length-shaped parameter was a fraction, so a plan asking for
    twelve months wrote min_coverage=12 and was discarded for it.

    "The top 10% largest buildings" and "the bottom half" are ranks within the
    portfolio, not floor areas, and a caller cannot know the cut-off without
    asking. The deciles are rank buckets over all 1,636 buildings, every one of
    which carries a floor area: ``floor_area_decile_at_least=10`` is the top
    tenth and ``floor_area_decile_at_most=5`` the bottom half.

    Buckets rather than a value cut-off, because the two are not the same
    answer. Cutting at the interpolated ninetieth percentile takes 164
    buildings where the top decile holds 163, and that one building moves the
    consumption ratio against the bottom half from 2.76 to 2.80. The cut-offs
    in metres are reported so the reader can see where the line fell.

    ``min_complete_days_in_year`` counts days inside a single calendar year
    that carry every expected hour, and ``min_varying_days_in_year`` counts
    how many of those days actually move rather than repeating one value.
    A building qualifies if any one of its years clears the floors, so
    neither needs a year named.

    Whenever a complete year is wanted *for* something -- benchmarking,
    rating, a baseline, year-on-year comparison -- pass both, the second a
    little below the first: min_complete_days_in_year=365 with
    min_varying_days_in_year=350. 328 buildings hold a complete 365-day
    year; 59 of those are stuck meters repeating one value all year, and
    they benchmark to a number that means nothing. Passing only the first
    floor answers "which buildings have the data", not "which buildings
    can be benchmarked", and the two differ by a fifth of the set.
    """
    params: list = []
    where = "1=1"
    # A value that is not in the catalog at all is a different answer from a
    # combination that happens to match nothing, and only one of them is the
    # caller's mistake. A specialist turned the building id Lamb_office_Bertha
    # into city="Bertha", which matched nothing, and that empty result was read
    # downstream as a real "none". Say which value does not exist, and what
    # does.
    for label, value in (("city", city), ("site_id", site_id)):
        if value is None:
            continue
        known = set(ctx.db.df(
            f"SELECT DISTINCT {label} AS v FROM sites WHERE {label} IS NOT NULL"
        )["v"])
        if str(value) not in known:
            plural = {"city": "cities", "site_id": "site ids"}[label]
            return insufficient(
                label, f"no {label} {value!r} in this dataset; "
                       f"the {plural} are {sorted(known)}")
    # The same for a type that is not one. Every building here is commercial,
    # so "all commercial buildings" is the whole portfolio and there is no
    # Commercial type to filter on -- asked for one, this returned EMPTY, and
    # "no buildings match" reads as a fact about the estate rather than as the
    # filter being wrong.
    if building_type is not None:
        types = set(ctx.db.df(
            "SELECT DISTINCT building_type_canonical AS v FROM buildings"
            " WHERE building_type_canonical IS NOT NULL")["v"])
        if str(building_type) not in types:
            return insufficient(
                "building_type",
                f"no building type {building_type!r} in this dataset; the "
                f"types are {sorted(types)}. Every building here is a "
                f"commercial, non-residential one, so a whole-portfolio "
                f"question needs no type filter at all")
    # A region covers several countries and means their union, so it is one
    # query with a list rather than a branch each: running one branch per
    # country and reporting whichever ran last answered "education
    # buildings in Europe" with Ireland's twenty and dropped the UK's.
    # Naming every country is how a model writes "anywhere", but as a filter
    # it is not the same thing: 400 buildings sit at sites with no country
    # recorded, and listing all four dropped every one of them from a question
    # that had asked for no country at all. A filter that admits every known
    # value constrains nothing the caller meant, so it is dropped rather than
    # applied.
    if countries and set(countries) >= set(ctx.db.df(
            "SELECT DISTINCT country_canonical FROM sites"
            " WHERE country_canonical IS NOT NULL")["country_canonical"]):
        countries = None
    if countries:
        where += " AND s.country_canonical IN (SELECT unnest(?))"
        params.append(list(countries))
    else:
        where += _eq("s.country_canonical", country, params)
    where += _eq("s.city", city, params)
    where += _eq("b.site_id", site_id, params)
    where += _eq("b.building_type_canonical", building_type, params)
    if has_readings is not None:
        clause = ("EXISTS" if has_readings else "NOT EXISTS")
        where += (f" AND {clause} (SELECT 1 FROM electricity_streams es"
                  f" WHERE es.building_id = b.building_id"
                  f" AND es.n_observations > 0)")
    # Matched case-insensitively: the catalog writes "College Laboratory" and
    # a question writes "college laboratory". A value that names nothing is a
    # refusal listing what exists, for the same reason an unknown canonical
    # type is -- an empty result reads as a fact about the estate rather than
    # as a filter that missed.
    for column, value, label in (("b.building_subtype", building_subtype,
                                  "building_subtype"),
                                 ("b.industry", industry, "industry"),
                                 ("b.subindustry", subindustry, "subindustry")):
        if value is None:
            continue
        known = ctx.db.df(
            f"SELECT DISTINCT {column.split('.')[1]} AS v FROM buildings "
            f"WHERE {column.split('.')[1]} IS NOT NULL")["v"].tolist()
        match = next((k for k in known
                      if str(k).lower() == str(value).strip().lower()), None)
        if match is None:
            shown = sorted(known)[:25]
            return insufficient(
                label,
                f"no {label.replace('_', ' ')} {value!r} in this dataset; "
                f"{len(known)} exist"
                + (f", including {shown}" if shown else ""))
        where += f" AND {column} = ?"
        params.append(match)
    percentile_cuts: dict = {}
    # A decile is one of ten. "The bottom 50%" written as
    # floor_area_decile_at_most=50 matches every decile there is, so the
    # cohort it produced was the whole portfolio presented as its smaller
    # half -- and the comparison built on it was off by the size of the
    # dataset rather than visibly wrong.
    for name, value in (("floor_area_decile_at_least", floor_area_decile_at_least),
                        ("floor_area_decile_at_most", floor_area_decile_at_most)):
        if value is not None and not 1 <= int(value) <= 10:
            return insufficient(
                name, f"{name}={value} is not a decile: deciles run 1 to 10, "
                      f"so the bottom half is 5 and the top tenth is 10")
    if (floor_area_decile_at_least is not None
            or floor_area_decile_at_most is not None):
        deciles = ctx.db.df("""
            SELECT building_id, floor_area_m2,
                   NTILE(10) OVER (ORDER BY floor_area_m2) AS decile
            FROM buildings""")
        keep = deciles
        if floor_area_decile_at_least is not None:
            keep = keep[keep["decile"] >= int(floor_area_decile_at_least)]
        if floor_area_decile_at_most is not None:
            keep = keep[keep["decile"] <= int(floor_area_decile_at_most)]
        percentile_cuts = {
            "floor_area_decile_at_least": floor_area_decile_at_least,
            "floor_area_decile_at_most": floor_area_decile_at_most,
            "buildings_in_deciles": int(len(keep)),
            "floor_area_range_m2": [round(float(keep["floor_area_m2"].min()), 1),
                                    round(float(keep["floor_area_m2"].max()), 1)]
            if len(keep) else None}
        where += " AND b.building_id IN (SELECT unnest(?))"
        params.append(list(keep["building_id"]))

    if min_floor_area_m2 is not None:
        where += " AND b.floor_area_m2 >= ?"; params.append(min_floor_area_m2)
    if max_floor_area_m2 is not None:
        where += " AND b.floor_area_m2 <= ?"; params.append(max_floor_area_m2)
    if min_year_built is not None:
        where += " AND b.year_built >= ?"; params.append(min_year_built)
    if max_year_built is not None:
        where += " AND b.year_built <= ?"; params.append(max_year_built)
    # No implicit quality floor. find_buildings answers "which buildings match
    # these attributes"; whether a building has enough data to *analyse* is the
    # analysis tool's question, and every one of them carries its own
    # min_coverage. Inheriting ctx.min_coverage=0.9 here meant "how many
    # buildings are at site Hog" answered 137 instead of 163 -- the eligible
    # subset presented as a census. Ask for a floor explicitly, or use
    # `requires` to resolve an analysable set.
    coverage = min_coverage
    if coverage is not None and coverage_year is None:
        where += " AND st.coverage_fraction >= ?"; params.append(coverage)
    elif coverage is not None:
        # "less than 2% missing for year 2017" is a floor on that year, not on
        # the record. The stream's coverage_fraction spans everything it holds,
        # so a meter that ran all of 2016 and died in March 2017 passes a
        # whole-record floor and fails the question. The yearly aggregate
        # measures the year the question named.
        where += (" AND EXISTS (SELECT 1 FROM agg_yearly y"
                  " WHERE y.building_id = b.building_id"
                  " AND YEAR(y.period_start) = ? AND y.coverage_fraction >= ?)")
        params.append(int(coverage_year))
        params.append(coverage)

    # "Complete building metadata" names no fields, and the fields differ
    # enormously: floor area is on all 1,636 buildings, year built on 817 and
    # occupants on 230. Answering it with one definition of "complete" hides
    # that choice, so the caller names the fields and the answer reports them.
    for field in (has_metadata or []):
        column = {"floor_area": "b.floor_area_m2",
                  "year_built": "b.year_built",
                  "occupants": "b.occupants",
                  "number_of_floors": "b.number_of_floors",
                  "building_type": "b.building_type_canonical"}.get(field)
        if column:
            where += f" AND {column} IS NOT NULL"
            if field == "occupants":
                # Present and zero is not usable as a divisor, and 41 buildings
                # record exactly that.
                where += " AND b.occupants > 0"

    if min_months_of_data is not None:
        where += (" AND DATE_DIFF('day', st.start_timestamp,"
                  " st.end_timestamp) >= ?")
        params.append(int(round(float(min_months_of_data) * 30.44)))
    # Whole-day completeness, counted inside a calendar year. The yearly
    # coverage fraction cannot answer this: it is observed over expected hours
    # for the year, so a meter missing one hour of leap-year 2016 scores
    # 0.99989 and is refused by a 1.0 floor while still holding 365 whole days
    # -- which is what "a complete 365-day year" asks for. A day is flat when
    # its readings never move off one value, and a flat year benchmarks to a
    # number that means nothing, so the two floors are separate.
    day_having = []
    if min_complete_days_in_year is not None:
        day_having.append(("SUM(CASE WHEN d.coverage_fraction >= 1.0"
                           " THEN 1 ELSE 0 END) >= ?",
                           int(min_complete_days_in_year)))
    if min_varying_days_in_year is not None:
        day_having.append(("SUM(CASE WHEN d.coverage_fraction >= 1.0"
                           " AND d.peak_demand > d.min_demand"
                           " THEN 1 ELSE 0 END) >= ?",
                           int(min_varying_days_in_year)))
    if day_having:
        where += (" AND EXISTS (SELECT 1 FROM agg_daily d"
                  " WHERE d.building_id = b.building_id AND d.grain = 'daily'"
                  " GROUP BY YEAR(d.period_start) HAVING "
                  + " AND ".join(clause for clause, _ in day_having) + ")")
        params.extend(value for _, value in day_having)

    # The complement, and the question usually asked. "Gaps exceeding one week"
    # wants the broken streams. With only an upper bound available a plan wrote
    # max_gap_hours=168 and returned the 968 streams whose gaps are *shorter*
    # than a week -- the opposite set, and a plausible-looking one. The names
    # say their direction now: a parameter whose meaning depends on knowing a
    # convention will be read by the convention the reader already has, and
    # "max gap" reads as "the longest gap" to almost anyone.
    # Narrowing the catalogue to named buildings is the obvious way to ask
    # "this building, with its metadata" and the tool refused it: the step
    # errored, the replan guessed the name was a *subtype*, and the question
    # died on "no building subtype 'Fox_office_Alice'".
    if building_ids:
        where += _in_clause("b.building_id", list(building_ids), params)

    if complete_year_coverage is not None or weather_coverage_at_least is not None:
        floor = complete_year_coverage if complete_year_coverage is not None else 0.9
        years = tuple(int(y) for y in (complete_years or (2016, 2017)))
        keep = _complete_year_buildings(ctx, floor, years,
                                        weather_coverage_at_least)
        where += (" AND 1=0" if not keep
                  else _in_clause("b.building_id", sorted(keep), params))

    if consecutive_months_coverage is not None:
        keep = _rolling_year_buildings(
            ctx, float(consecutive_months_coverage), int(consecutive_months),
            weather_coverage_at_least)
        where += (" AND 1=0" if not keep
                  else _in_clause("b.building_id", sorted(keep), params))

    if min_unbroken_hours is not None:
        runs = _longest_run_hours(ctx)
        keep = {b for b, hours in runs.items() if hours >= int(min_unbroken_hours)}
        where += (" AND 1=0" if not keep
                  else _in_clause("b.building_id", sorted(keep), params))

    if gap_longer_than_hours is not None or gap_at_most_hours is not None:
        gaps = _internal_gap_hours(ctx)
        keep = {b for b, g in gaps.items()
                if (gap_longer_than_hours is None or g > float(gap_longer_than_hours))
                and (gap_at_most_hours is None or g <= float(gap_at_most_hours))}
        if not keep:
            where += " AND 1=0"
        else:
            where += _in_clause("b.building_id", sorted(keep), params)

    unknown = [r for r in (requires or []) if r not in TASK_CAPABILITY]
    if unknown:
        return insufficient("unknown_capability",
                            f"unknown capability {unknown}; "
                            f"valid: {sorted(TASK_CAPABILITY)}")
    for req in requires or []:
        gate = TASK_CAPABILITY[req]
        # Selection asks whether a metric is *computable*; whether it is
        # trustworthy over the window in question is the analysis tool's
        # judgement, made against that window with min_coverage. Where the
        # snapshot bundles the two -- can_eui carries a coverage verdict over
        # the whole record -- selecting on the bundle applies a second,
        # differently-scoped coverage rule that the caller never asked for and
        # cannot see. ``eligibility="fit"`` asks for the bundled verdict.
        if eligibility == "structural" and gate in STRUCTURAL:
            where += f" AND b.{STRUCTURAL[gate]}"
            if req != "consumption_analysis":
                where += " AND c.can_consumption_analysis"
        else:
            where += f" AND c.{gate}"

    df = ctx.db.df(f"""
        SELECT b.building_id, b.site_id, s.city, s.country_canonical AS country,
               b.building_type_canonical AS building_type, b.floor_area_m2,
               b.year_built, b.occupants, st.coverage_fraction
        FROM buildings b
        JOIN sites s ON s.site_id = b.site_id
        JOIN capabilities c ON c.building_id = b.building_id
        LEFT JOIN electricity_streams st ON st.building_id = b.building_id
        WHERE {where}
        ORDER BY b.building_id{f" LIMIT {int(limit)}" if limit else ""}
    """, params or None)
    if df.empty:
        return empty("no buildings match the filter",
                     evidence={"filters": {"country": country, "city": city,
                                           "building_type": building_type,
                                           "requires": requires,
                                           "min_coverage": coverage}})
    # A count means nothing without the population it was drawn from. Asked
    # "how many buildings are in this dataset", this tool answers 1,256 -- the
    # buildings clearing a 90% coverage floor -- while the catalog holds 1,636
    # and 1,578 are metered. Reporting only the 1,256 reads as a census.
    matched_before_limit = int(ctx.db.df(f"""
        SELECT COUNT(*) AS n
        FROM buildings b
        JOIN sites s ON s.site_id = b.site_id
        JOIN capabilities c ON c.building_id = b.building_id
        LEFT JOIN electricity_streams st ON st.building_id = b.building_id
        WHERE {where}
    """, params or None)["n"].iloc[0])
    catalog_total = int(ctx.db.df(
        "SELECT COUNT(*) AS n FROM buildings")["n"].iloc[0])
    metered_total = int(ctx.db.df(
        "SELECT COUNT(DISTINCT building_id) AS n "
        "FROM electricity_streams")["n"].iloc[0])
    return frame_result(df, ctx.store, name="find_buildings",
                        evidence={"entity": "buildings",
                                  "n_buildings": len(df),
                                  "matched_filter": matched_before_limit,
                                  "buildings_in_catalog": catalog_total,
                                  "buildings_metered": metered_total,
                                  "min_coverage": coverage,
                                  **percentile_cuts,
                                  # "Continuous" has no single threshold, so
                                  # the one used has to travel with the answer:
                                  # no gap at all gives 113 buildings and a gap
                                  # of up to a day gives 628, and a reader
                                  # cannot tell those apart from the count.
                                  "coverage_year": coverage_year,
                                  "min_months_of_data": min_months_of_data,
                                  "min_complete_days_in_year":
                                      min_complete_days_in_year,
                                  "min_varying_days_in_year":
                                      min_varying_days_in_year,
                                  "gap_at_most_hours": gap_at_most_hours,
                                  "has_metadata": has_metadata,
                                  "limit": limit,
                                  "truncated_by_limit":
                                      matched_before_limit > len(df),
                                  "requires": requires or []})


@tool("data_discovery")
def get_building_metadata(ctx: ToolContext, building_ids: list[str],
                          include_attributes: bool = False) -> ToolResult:
    """Retrieve full metadata for one or more buildings."""
    params: list = []
    clause = _in_clause("b.building_id", building_ids, params)
    df = ctx.db.df(f"""
        SELECT b.*, s.site_name, s.timezone, s.utc_offset_hours
        FROM buildings b JOIN sites s ON s.site_id = b.site_id
        WHERE 1=1 {clause} ORDER BY b.building_id
    """, params or None)
    missing = sorted(set(building_ids) - set(df["building_id"]))
    extras = {}
    if include_attributes and not df.empty:
        aparams: list = []
        aclause = _in_clause("building_id", building_ids, aparams)
        attrs = ctx.db.df(
            f"SELECT * FROM building_attributes WHERE 1=1 {aclause}", aparams or None)
        extras["attributes"] = attrs.to_dict(orient="records")
    result = frame_result(df, ctx.store, name="building_metadata",
                          evidence={"requested": len(building_ids), **extras})
    if missing and result.ok:
        return partial(result.data, message=f"{len(missing)} unknown building_id(s)",
                       handle=result.handle, rows=result.rows,
                       columns=result.columns, preview=result.preview,
                       evidence={**result.evidence, "unknown": missing[:20]})
    return result


@tool("data_discovery")
def get_data_availability(ctx: ToolContext, building_ids: list[str]) -> ToolResult:
    """Coverage, period and resolution of the streams already chosen.

    Describes buildings you have; it does not choose them. Selecting on
    coverage or record length is find_buildings."""
    params: list = []
    clause = _in_clause("st.building_id", building_ids, params)
    df = ctx.db.df(f"""
        SELECT st.building_id, st.stream_id, st.site_id, st.unit,
               st.resolution_minutes, st.start_timestamp, st.end_timestamp,
               st.n_observations, st.expected_observations, st.coverage_fraction,
               q.missing_fraction, q.zero_masked_fraction, q.longest_gap_hours,
               q.quality_status,
               ws.weather_source_id IS NOT NULL AS weather_available,
               ws.air_temperature_completeness
        FROM electricity_streams st
        LEFT JOIN data_quality q ON q.stream_id = st.stream_id
        LEFT JOIN sites s ON s.site_id = st.site_id
        LEFT JOIN weather_sources ws ON ws.weather_source_id = s.weather_source_id
        WHERE 1=1 {clause} ORDER BY st.building_id
    """, params or None)
    return frame_result(df, ctx.store, name="data_availability",
                        evidence={"requested": len(building_ids)})


@tool("data_discovery")
def check_task_eligibility(
    ctx: ToolContext,
    building_ids: list[str],
    task: Literal["consumption_analysis", "eui_analysis", "weather_analysis",
                  "occupancy_normalisation", "peer_benchmark", "forecasting",
                  "spatial_analysis"],
) -> ToolResult:
    """Can these buildings support a given analysis? Names the failing predicate.

    An empty eligible set is a routine outcome, not an error: for some questions
    the correct answer is a refusal that says which predicate failed.
    """
    if task not in TASK_CAPABILITY:
        return insufficient("unknown_task",
                            f"unknown task {task!r}; valid: {sorted(TASK_CAPABILITY)}")
    column = TASK_CAPABILITY[task]
    params: list = []
    clause = _in_clause("c.building_id", building_ids, params)
    df = ctx.db.df(f"""
        SELECT c.building_id, c.{column} AS eligible,
               b.floor_area_m2, b.occupants, b.country_canonical,
               st.coverage_fraction
        FROM capabilities c
        LEFT JOIN buildings b ON b.building_id = c.building_id
        LEFT JOIN electricity_streams st ON st.building_id = c.building_id
        WHERE 1=1 {clause} ORDER BY c.building_id
    """, params or None)
    if df.empty:
        return empty(f"no known buildings among {len(building_ids)} requested")

    eligible = df.loc[df["eligible"].fillna(False), "building_id"].tolist()
    ineligible = df.loc[~df["eligible"].fillna(False), "building_id"].tolist()
    evidence = {"task": task, "predicate": column,
                "n_eligible": len(eligible), "n_ineligible": len(ineligible),
                "ineligible_sample": ineligible[:10]}
    if not eligible:
        return insufficient(column,
                            f"no building satisfies {column} for task {task!r}",
                            evidence=evidence)
    if ineligible:
        return partial({"eligible": eligible, "ineligible": ineligible},
                       message=f"{len(ineligible)} of {len(df)} buildings fail {column}",
                       evidence=evidence)
    return ok({"eligible": eligible, "ineligible": []}, evidence=evidence)


@tool("data_discovery")
def fetch_electricity(
    ctx: ToolContext,
    building_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    resolution: Literal["hourly", "daily", "monthly", "yearly"] = "hourly",
    timezone: Literal["local", "utc"] = "local",
) -> ToolResult:
    """Retrieve electricity measurements. Returns a handle, not the rows.

    ``hourly`` reads the raw series; coarser resolutions read the precomputed
    aggregates, so a yearly question never scans 26M rows.
    """
    params: list = []
    if resolution == "hourly":
        ts = "timestamp_local" if timezone == "local" else "timestamp_utc"
        clause = _in_clause("building_id", building_ids, params)
        where = f"1=1 {clause}"
        if start:
            where += f" AND {ts} >= ?"; params.append(start)
        if end:
            where += f" AND {ts} < ?"; params.append(end)
        # How many rows this would materialise, asked before materialising
        # them. A plan that fetches the portfolio's hourly readings is asking
        # for 26 million rows in a DataFrame, and the process does not
        # survive it: the OOM reaper takes the whole run, so one bad plan
        # costs every question after it too. Counting first is a scan and
        # cheap; being killed is neither recoverable nor reported.
        rows = int(ctx.db.df(
            f"SELECT COUNT(*) AS n FROM electricity WHERE {where}",
            params or None)["n"].iloc[0])
        if rows > MAX_FETCH_ROWS:
            return insufficient(
                "too_many_rows",
                f"{rows:,} hourly readings for this selection, above the "
                f"{MAX_FETCH_ROWS:,} a single fetch will hold. Narrow the "
                f"buildings or the window, or ask for resolution='daily' -- "
                f"the daily aggregate answers anything about totals, "
                f"coverage or day-level shape without the readings.")
        df = ctx.db.df(f"""
            SELECT building_id, stream_id, site_id,
                   {ts} AS timestamp, value, quality_flag
            FROM electricity WHERE {where}
            ORDER BY building_id, timestamp
        """, params or None)
    else:
        view = AGG_VIEW[resolution]
        clause = _in_clause("building_id", building_ids, params)
        where = f"1=1 {clause}"
        if start:
            where += " AND period_start >= ?"; params.append(start)
        if end:
            where += " AND period_start < ?"; params.append(end)
        df = ctx.db.df(f"""
            SELECT building_id, stream_id, site_id, period_start AS timestamp,
                   total_kwh, mean_demand, peak_demand, min_demand,
                   median_demand, peak_hour_local, n_observations,
                   expected_observations, coverage_fraction
            FROM {view} WHERE {where}
            ORDER BY building_id, timestamp
        """, params or None)
    return frame_result(df, ctx.store, name=f"electricity_{resolution}",
                        force_handle=True,
                        evidence={"resolution": resolution, "timezone": timezone,
                                  "buildings": len(building_ids),
                                  "start": start, "end": end})


@tool("data_discovery")
def fetch_weather(
    ctx: ToolContext,
    site_ids: list[str] | None = None,
    building_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    variables: list[Literal["air_temperature", "dew_temperature", "wind_speed",
                            "wind_direction", "cloud_coverage",
                            "precip_depth_1hr", "precip_depth_6hr",
                            "sea_level_pressure"]] | None = None,
    timezone: Literal["local", "utc"] = "local",
) -> ToolResult:
    """Retrieve site-level weather, addressed by site or by building."""
    if not site_ids and not building_ids:
        return insufficient("no_target", "give site_ids or building_ids")
    if not site_ids:
        params: list = []
        clause = _in_clause("building_id", building_ids, params)
        sites = ctx.db.df(
            f"SELECT DISTINCT site_id FROM buildings WHERE 1=1 {clause}",
            params or None)
        site_ids = sites["site_id"].tolist()
        if not site_ids:
            return empty("no sites resolved from the given buildings")

    cols = ["air_temperature", "dew_temperature", "sea_level_pressure",
            "wind_direction", "wind_speed", "cloud_coverage",
            "precip_depth_1hr", "precip_depth_6hr"]
    chosen = [c for c in (variables or cols) if c in cols]
    if not chosen:
        return insufficient("unknown_variables",
                            f"none of {variables} are weather variables; valid: {cols}")

    ts = "timestamp_local" if timezone == "local" else "timestamp_utc"
    params = []
    clause = _in_clause("site_id", site_ids, params)
    where = f"1=1 {clause}"
    if start:
        where += f" AND {ts} >= ?"; params.append(start)
    if end:
        where += f" AND {ts} < ?"; params.append(end)
    df = ctx.db.df(f"""
        SELECT site_id, {ts} AS timestamp, {", ".join(chosen)}
        FROM weather WHERE {where} ORDER BY site_id, timestamp
    """, params or None)
    return frame_result(df, ctx.store, name="weather", force_handle=True,
                        evidence={"sites": site_ids, "variables": chosen})


@tool("data_discovery")
def fetch_contextual_data(
    ctx: ToolContext,
    building_ids: list[str],
    start: str | None = None,
    end: str | None = None,
    resolution: Literal["hourly", "daily"] = "hourly",
    include_weather: bool = True,
    include_metadata: bool = True,
) -> ToolResult:
    """Electricity joined to weather and building metadata in one aligned frame.

    This is the workhorse behind multi-source questions: one call instead of
    three fetches plus two joins, and one handle instead of three.
    """
    params: list = []
    clause = _in_clause("e.building_id", building_ids, params)
    where = f"1=1 {clause}"
    if resolution == "hourly":
        base = """
            SELECT e.building_id, e.site_id, e.timestamp_local AS timestamp,
                   e.value, e.quality_flag
            FROM electricity e"""
        ts_filter = "e.timestamp_local"
    else:
        base = """
            SELECT a.building_id, a.site_id, a.period_start AS timestamp,
                   a.total_kwh AS value, a.coverage_fraction
            FROM agg_daily a"""
        clause = _in_clause("a.building_id", building_ids, params := [])
        where = f"1=1 {clause}"
        ts_filter = "a.period_start"
    if start:
        where += f" AND {ts_filter} >= ?"; params.append(start)
    if end:
        where += f" AND {ts_filter} < ?"; params.append(end)

    weather_join = ""
    weather_cols = ""
    if include_weather:
        grain = "timestamp_local" if resolution == "hourly" else \
                "CAST(timestamp_local AS DATE)"
        weather_cols = (", w.air_temperature, w.dew_temperature, w.wind_speed, "
                        "w.cloud_coverage")
        weather_join = f"""
            LEFT JOIN (
                SELECT site_id, {grain} AS wts,
                       AVG(air_temperature) AS air_temperature,
                       AVG(dew_temperature) AS dew_temperature,
                       AVG(wind_speed) AS wind_speed,
                       AVG(cloud_coverage) AS cloud_coverage
                FROM weather GROUP BY 1, 2
            ) w ON w.site_id = base.site_id AND w.wts = base.timestamp"""
    meta_join = ""
    meta_cols = ""
    if include_metadata:
        meta_cols = (", b.building_type_canonical AS building_type, "
                     "b.floor_area_m2, b.year_built, b.occupants, "
                     "s.city, s.country_canonical AS country")
        meta_join = """
            LEFT JOIN buildings b ON b.building_id = base.building_id
            LEFT JOIN sites s ON s.site_id = base.site_id"""

    df = ctx.db.df(f"""
        WITH base AS ({base} WHERE {where})
        SELECT base.* {weather_cols} {meta_cols}
        FROM base {weather_join} {meta_join}
        ORDER BY base.building_id, base.timestamp
    """, params or None)
    return frame_result(df, ctx.store, name="contextual", force_handle=True,
                        evidence={"buildings": len(building_ids),
                                  "resolution": resolution,
                                  "weather": include_weather,
                                  "metadata": include_metadata})
