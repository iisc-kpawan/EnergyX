"""Open the snapshot in DuckDB and define the views tasks query.

Views:
- the catalog tables: datasets, sites, buildings, building_attributes, electricity_streams,
  data_quality, capabilities, weather_sources;
- building_sites: buildings with their site's name, city, country and UTC offset;
- electricity and weather: the hourly readings, with `timestamp_local` (site-local standard time)
  added to the stored UTC timestamp;
- agg_daily, agg_monthly, agg_yearly, profile_daily, profile_weekly: EnergyX's own aggregates, for
  cross-checking the spreadsheet's draft SQL only.

In `electricity`, an hour with no row is a gap in the source, and a null value is a reading
BDG2's cleaning removed. Both count as missing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from ..core.config import Config

CATALOG_TABLES = (
    "datasets", "sites", "buildings", "building_attributes", "electricity_streams",
    "data_quality", "capabilities", "weather_sources",
)
# EnergyX's precomputed aggregates, under the names its own SQL uses. The spreadsheet's draft SQL
# reads them, so they are here for cross-checks; gold tasks compute from the readings instead.
AGGREGATE_VIEWS = {
    "agg_daily": "aggregates/daily/*.parquet",
    "agg_monthly": "aggregates/monthly/*.parquet",
    "agg_yearly": "aggregates/yearly/*.parquet",
    "profile_daily": "aggregates/profiles/daily_profile.parquet",
    "profile_weekly": "aggregates/profiles/weekly_profile.parquet",
}
VIEWS = (*CATALOG_TABLES, "building_sites", "electricity", "weather", *AGGREGATE_VIEWS)

# The views each required_data label in the manifest refers to.
REQUIRED_DATA_VIEWS = {
    "Building Metadata": ("buildings", "building_attributes", "sites", "building_sites"),
    "Electricity Consumption Data": ("electricity", "electricity_streams"),
    "Weather Data": ("weather", "weather_sources"),
    "Dataset/Catalog Metadata": ("datasets", "data_quality", "capabilities"),
}

# Stored UTC plus the site's constant offset gives site-local standard time.
_LOCAL_TIME = "timezone('UTC', r.timestamp_utc) + to_minutes(CAST(round(s.utc_offset_hours * 60) AS BIGINT))"


def quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def open_snapshot(config: Config) -> duckdb.DuckDBPyConnection:
    manifest = config.snapshot_dir / "MANIFEST.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no snapshot at {config.snapshot_dir} (MANIFEST.json missing)")
    version = json.loads(manifest.read_text()).get("snapshot_version")
    if version != config.data_version:
        raise ValueError(f"snapshot is {version!r}, config expects {config.data_version!r}")

    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    # Floating-point sums and averages can change with the order rows arrive in, and with
    # several threads that order varies between runs.
    con.execute("SET threads = 1")
    con.execute("SET enable_progress_bar = false")
    _create_views(con, config.snapshot_dir)
    return con


def _create_views(con: duckdb.DuckDBPyConnection, snap: Path) -> None:
    for table in CATALOG_TABLES:
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({quote(snap / 'catalog' / f'{table}.parquet')})")
    con.execute("""
        CREATE VIEW building_sites AS
        SELECT b.*, s.site_name, s.country_code, s.utc_offset_hours, s.climate_zone, s.weather_source_id
        FROM buildings b JOIN sites s USING (site_id)
    """)
    con.execute(f"""
        CREATE VIEW electricity AS
        SELECT {_LOCAL_TIME} AS timestamp_local, r.timestamp_utc, r.site_id, r.building_id,
               r.stream_id, r.value, r.quality_flag
        FROM read_parquet({quote(snap / 'electricity' / '**' / '*.parquet')}, hive_partitioning = true) r
        JOIN sites s USING (site_id)
    """)
    con.execute(f"""
        CREATE VIEW weather AS
        SELECT {_LOCAL_TIME} AS timestamp_local, r.* EXCLUDE (dataset, year)
        FROM read_parquet({quote(snap / 'weather' / '**' / '*.parquet')}, hive_partitioning = true) r
        JOIN sites s USING (site_id)
    """)
    for view, pattern in AGGREGATE_VIEWS.items():
        if any(snap.glob(pattern)):
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet({quote(snap / pattern)})")


def run(con: duckdb.DuckDBPyConnection, query: str, params: dict | list | None = None) -> pd.DataFrame:
    return (con.execute(query, params) if params else con.execute(query)).df()


def has_rows(con: duckdb.DuckDBPyConnection, view: str) -> bool:
    exists = con.execute("SELECT 1 FROM duckdb_views() WHERE view_name = ?", [view]).fetchone()
    if view not in VIEWS or not exists:
        return False
    return con.execute(f"SELECT 1 FROM {view} LIMIT 1").fetchone() is not None


def in_list(column: str, values: list) -> tuple[str, list]:
    values = list(values)
    return f"{column} IN ({', '.join('?' for _ in values)})", values


def select_readings(view: str, key: str, ids: list[str], start: datetime | None, end: datetime | None,
                    columns: str) -> tuple[str, list]:
    """A query for one view's readings for some IDs, from `start` (inclusive) to `end` (exclusive), local time."""
    clause, params = in_list(key, ids)
    clauses = [clause]
    if start is not None:
        clauses.append("timestamp_local >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp_local < ?")
        params.append(end)
    return f"SELECT {columns} FROM {view} WHERE {' AND '.join(clauses)} ORDER BY {key}, timestamp_local", params
