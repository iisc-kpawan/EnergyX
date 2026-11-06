"""GoldContext: the only way a task module reaches the data.

It opens the snapshot in DuckDB (see data_access/sql.py for the views) and, when a task needs it,
EnergyX's knowledge graph. Every query a task runs is recorded in `queries_run`.

Some catalog columns were derived from the readings by EnergyX's ingest: coverage in
electricity_streams, and everything in data_quality and capabilities. Compute such measures from
the readings and use those columns only to cross-check, so a bug in EnergyX can't also make the
gold answer wrong.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..data_access import kg, measurements, metadata, sql, weather
from .config import Config, load_config


class GoldContext:
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.data_version = self.config.data_version
        self.period = self.config.period
        self.seed = self.config.seed
        self.queries_run: list[str] = []
        self._db = sql.open_snapshot(self.config)
        self._graph: kg.KnowledgeGraph | None = None

    def sql(self, query: str, params: dict | list | None = None) -> pd.DataFrame:
        """Run a read query and return a DataFrame."""
        self.queries_run.append(" ".join(query.split()))
        return sql.run(self._db, query, params)

    def get_metadata(self, **filters) -> pd.DataFrame:
        """Buildings with their site's location. A filter matches a column exactly; a list matches any of its values."""
        return self.sql(*metadata.select_buildings(**filters))

    def get_measurements(
        self, building_ids: list[str], start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Hourly electricity in site-local time, from `start` (inclusive) to `end` (exclusive)."""
        return self.sql(*measurements.select_readings(building_ids, start, end))

    def get_weather(
        self,
        site_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        variables: list[str] | None = None,
    ) -> pd.DataFrame:
        """Hourly weather in site-local time, from `start` (inclusive) to `end` (exclusive)."""
        return self.sql(*weather.select_weather(site_ids, start, end, variables))

    def query_kg(self, sparql: str) -> pd.DataFrame:
        """Run a SPARQL SELECT on EnergyX's graph. The ekg:, ekgd:, rdf:, rdfs: and xsd: prefixes are predefined."""
        if self._graph is None:
            self._graph = kg.KnowledgeGraph(self.config.graph_dir)
        self.queries_run.append("SPARQL: " + " ".join(sparql.split()))
        return self._graph.query(sparql)

    def view_has_rows(self, view: str) -> bool:
        """Whether a view exists and isn't empty. Not recorded in queries_run."""
        return sql.has_rows(self._db, view)
