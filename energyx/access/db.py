"""DuckDB access to an EnergyX snapshot.

Every connection into the snapshot comes from here, because two invariants must
hold on all of them and neither is the default.

**1. Session timezone is UTC.**
Measurements are stored as UTC instants, which DuckDB types as
``TIMESTAMP WITH TIME ZONE``. DuckDB renders and extracts from that type in the
*session* timezone, which defaults to the host's. On a machine set to
``Asia/Kolkata`` an innocuous ``EXTRACT(hour FROM timestamp_utc)`` silently
returns 5 instead of 0 -- corrupting every load profile, peak-hour and daily
boundary without raising anything. Pinning the session to UTC makes the stored
instant the literal instant.

**2. Local time is derived in exactly one place.**
The ``electricity`` view exposes ``timestamp_local`` already computed from the
site's constant offset. Nothing downstream re-implements that shift, so
a question about "3pm" cannot mean different things in different tools.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from ..config import PATHS

# Local wall-clock from the stored UTC instant, using the site's constant offset.
# Fractional offsets (e.g. +5.5) are supported via minutes.
LOCAL_TS_SQL = "e.timestamp_utc + to_minutes(CAST(s.utc_offset_hours * 60 AS BIGINT))"

CATALOG_TABLES = (
    "datasets", "sites", "buildings", "building_attributes",
    "electricity_streams", "weather_sources", "data_quality", "capabilities",
)

AGGREGATE_VIEWS = {
    "agg_daily": "aggregates/daily/*.parquet",
    "agg_monthly": "aggregates/monthly/*.parquet",
    "agg_yearly": "aggregates/yearly/*.parquet",
    "profile_daily": "aggregates/profiles/daily_profile.parquet",
    "profile_weekly": "aggregates/profiles/weekly_profile.parquet",
}


def connect(snapshot: Path | None = None, threads: int = 4,
            read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB session with the snapshot registered as views."""
    snapshot = Path(snapshot or PATHS.snapshot)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")          # invariant 1 -- never remove
    con.execute(f"PRAGMA threads={threads}")

    for table in CATALOG_TABLES:
        path = snapshot / "catalog" / f"{table}.parquet"
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW {table} AS "
                        f"SELECT * FROM read_parquet('{path}')")

    elec_glob = snapshot / "electricity" / "**" / "*.parquet"
    if (snapshot / "electricity").exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW electricity_raw AS
            SELECT * FROM read_parquet('{elec_glob}', hive_partitioning = true)
        """)
        # invariant 2 -- the single definition of local time
        con.execute(f"""
            CREATE OR REPLACE VIEW electricity AS
            SELECT e.timestamp_utc,
                   {LOCAL_TS_SQL} AS timestamp_local,
                   e.stream_id, e.building_id, e.site_id,
                   e.dataset AS dataset_id,
                   e.value, e.quality_flag, e.year
            FROM read_parquet('{elec_glob}', hive_partitioning = true) e
            JOIN sites s USING (site_id)
        """)

    weather_glob = snapshot / "weather" / "**" / "*.parquet"
    if (snapshot / "weather").exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW weather AS
            SELECT w.*,
                   w.timestamp_utc + to_minutes(CAST(s.utc_offset_hours * 60 AS BIGINT))
                       AS timestamp_local,
                   s.site_id
            FROM read_parquet('{weather_glob}', hive_partitioning = true) w
            JOIN sites s ON s.weather_source_id = w.weather_source_id
        """)

    for name, pattern in AGGREGATE_VIEWS.items():
        matches = list(snapshot.glob(pattern))
        if matches:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                        f"SELECT * FROM read_parquet('{snapshot / pattern}')")
    return con


class SnapshotDB:
    """Convenience wrapper holding one configured connection."""

    def __init__(self, snapshot: Path | None = None, threads: int = 4):
        self.snapshot = Path(snapshot or PATHS.snapshot)
        self.con = connect(self.snapshot, threads=threads)

    def sql(self, query: str, params: dict | list | None = None):
        return self.con.execute(query, params) if params else self.con.execute(query)

    def df(self, query: str, params: dict | list | None = None):
        return self.sql(query, params).df()

    def value(self, query: str, params: dict | list | None = None):
        row = self.sql(query, params).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "SnapshotDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
