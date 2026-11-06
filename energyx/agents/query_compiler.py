"""Allow-listed AnalysisSpec to read-only DuckDB query compilation."""
from __future__ import annotations

from dataclasses import dataclass

from .analysis_spec import AnalysisSpec


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    params: list
    value_field: str = "value"


MEASURES = {
    "electricity_consumption": "total_kwh",
    "mean_demand": "mean_demand",
    "peak_demand": "peak_demand",
    "eui": "total_kwh / NULLIF(floor_area_m2, 0)",
}
AGGREGATES = {"sum": "SUM", "mean": "AVG", "median": "MEDIAN",
              "min": "MIN", "max": "MAX", "count": "COUNT"}
GROUPS = {"building_id", "building_type", "site_id", "city", "country"}


def compile_analysis(spec: AnalysisSpec) -> CompiledQuery:
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))
    if spec.task not in {"aggregate", "compare", "prioritize"}:
        raise ValueError(f"unsupported compiled task: {spec.task}")
    if spec.measure not in MEASURES:
        raise ValueError(f"unsupported measure: {spec.measure}")
    aggregate = spec.aggregation or "sum"
    if aggregate == "quantile":
        quantile = spec.parameters.get("quantile")
        if quantile is None or not 0 <= quantile <= 1:
            raise ValueError("quantile must be between 0 and 1")
        if spec.group_by or set(spec.entities) - {"building_ids", "site_id"}:
            raise ValueError("hourly quantile supports building or site filters only")
        params = [quantile]
        where = []
        for key, column in (("building_ids", "building_id"), ("site_id", "site_id")):
            value = spec.entities.get(key)
            if value:
                if key == "building_ids":
                    where.append(f"{column} IN (SELECT unnest(?))")
                    params.append(list(value))
                else:
                    where.append(f"{column} = ?"); params.append(value)
        if spec.period.get("start"):
            where.append("timestamp_local >= ?"); params.append(spec.period["start"])
        if spec.period.get("end"):
            where.append("timestamp_local < ?"); params.append(spec.period["end"])
        where.append("quality_flag = 'OK'")
        sql = ("SELECT QUANTILE_CONT(value, ?) AS value FROM electricity" +
               ((" WHERE " + " AND ".join(where)) if where else ""))
        return CompiledQuery(sql, params)
    if aggregate not in AGGREGATES:
        raise ValueError(f"unsupported aggregation: {aggregate}")
    if set(spec.group_by) - GROUPS:
        raise ValueError("unsupported group field")

    params = []
    where = []
    entity_columns = {"building_ids": "a.building_id", "building_type": "b.building_type_canonical",
                      "site_id": "b.site_id", "city": "b.city",
                      "country": "b.country_canonical"}
    for key, column in entity_columns.items():
        value = spec.entities.get(key)
        if value:
            if key == "building_ids":
                where.append(f"{column} IN (SELECT unnest(?))")
                params.append(list(value))
            else:
                where.append(f"{column} = ?")
                params.append(value)
    if spec.period.get("start"):
        where.append("a.period_start >= ?"); params.append(spec.period["start"])
    if spec.period.get("end"):
        where.append("a.period_start < ?"); params.append(spec.period["end"])

    groups = spec.group_by or (["building_id"] if spec.task == "prioritize" else [])
    select_groups = ", ".join(groups)
    metric = MEASURES[spec.measure]
    fn = AGGREGATES[aggregate]
    value = "COUNT(*)" if aggregate == "count" else f"{fn}({metric})"
    select = f"{select_groups}, {value} AS value" if groups else f"{value} AS value"
    group_sql = f" GROUP BY {select_groups}" if groups else ""
    order = ""
    if spec.ranking:
        direction = "ASC" if spec.ranking.get("direction") == "ascending" else "DESC"
        limit = int(spec.ranking.get("limit", 10))
        if limit <= 0:
            raise ValueError("ranking limit must be positive")
        order = f" ORDER BY value {direction} LIMIT ?"; params.append(limit)
    sql = f"""WITH per_building AS (
        SELECT a.building_id, ANY_VALUE(b.building_type_canonical) AS building_type,
               ANY_VALUE(b.site_id) AS site_id, ANY_VALUE(b.city) AS city,
               ANY_VALUE(b.country_canonical) AS country,
               SUM(a.total_kwh) AS total_kwh, AVG(a.mean_demand) AS mean_demand,
               MAX(a.peak_demand) AS peak_demand,
               ANY_VALUE(b.floor_area_m2) AS floor_area_m2
        FROM agg_monthly a JOIN buildings b USING (building_id)
        {('WHERE ' + ' AND '.join(where)) if where else ''}
        GROUP BY a.building_id
    ) SELECT {select} FROM per_building{group_sql}{order}"""
    return CompiledQuery(sql, params)
