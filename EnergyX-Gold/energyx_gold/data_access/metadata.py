"""Building and site metadata."""

from __future__ import annotations

from .sql import in_list


def select_buildings(**filters) -> tuple[str, list]:
    """A query for building_sites rows. A filter matches a column exactly; a list matches any of its values."""
    clauses, params = [], []
    for column, value in filters.items():
        if not column.isidentifier():
            raise ValueError(f"bad column name {column!r}")
        if isinstance(value, (list, tuple, set)):
            clause, values = in_list(column, sorted(value))
            clauses.append(clause)
            params.extend(values)
        else:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return f"SELECT * FROM building_sites{where} ORDER BY building_id", params
