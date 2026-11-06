"""Bring Your Own Dataset: onboard a second dataset into the snapshot.

This is the claim the design makes concrete -- *guided interoperability*, not
automatic support for anything. A user declares a manifest; EnergyX validates
it, converts to the canonical model, appends to the catalog and re-projects the
graph. The agent layer does not change.

Federation happens in the catalog and the graph, not in the tools: once REFIT's
rows sit beside BDG2's, every existing tool queries both without knowing a
second dataset arrived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import ecdm
from ..config import PATHS, SNAPSHOT_VERSION
from . import registry as manifest_registry

CATALOG_TABLES = ("datasets", "sites", "buildings", "electricity_streams",
                  "data_quality", "capabilities", "weather_sources",
                  "building_attributes")


@dataclass
class RegistrationReport:
    dataset_id: str
    buildings: int = 0
    sites: int = 0
    streams: int = 0
    rows: int = 0
    partitions: int = 0
    capabilities: dict = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    appliance_streams: int = 0
    end_uses: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _read_source(source_dir: Path, manifest) -> pd.DataFrame:
    """Load the electricity source into a wide frame with a timestamp column."""
    path = source_dir / manifest.get("electricity", "path")
    frame = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    if manifest.get("electricity", "index_is_timestamp"):
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    else:
        frame = frame.rename(
            columns={manifest.get("electricity", "timestamp"): "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _convert(values: np.ndarray, rule: str | None) -> np.ndarray:
    """Harmonise units into kWh. The manifest declares the rule; nothing is guessed."""
    if rule == "watts_to_kwh_hourly":
        # An hourly mean power in W is Wh over that hour; /1000 gives kWh.
        return values / 1000.0
    if rule == "watt_minutes_to_kwh":
        # Sum of per-minute watt samples over an hour: /60 gives Wh, /1000 kWh.
        return values / 60_000.0
    return values


def register(source_dir: str | Path, manifest_path: str | Path | None = None,
             snapshot: Path | None = None) -> RegistrationReport:
    source_dir = Path(source_dir)
    manifest_file = Path(manifest_path) if manifest_path else source_dir / "energyx.yaml"
    manifest = manifest_registry.load(manifest_file)
    dataset_id = manifest.dataset_id
    report = RegistrationReport(dataset_id=dataset_id)

    problems = manifest_registry.validate(manifest, manifest_file.parent)
    problems = [p for p in problems if "electricity.path not found" not in p]
    if problems:
        report.problems = problems
        return report

    snapshot = Path(snapshot or PATHS.snapshot)
    base = manifest_file.parent

    sites = pd.read_csv(base / manifest.get("sites", "path"))
    buildings = pd.read_csv(base / manifest.get("metadata", "path"))
    # Path("") is PosixPath(".") and therefore truthy -- an explicit None check
    # is the only safe way to fall back to the given source directory.
    declared = manifest.get("electricity", "source_dir")
    wide = _read_source(Path(declared) if declared else source_dir, manifest)

    offset = pd.Timedelta(hours=float(
        manifest.get("timestamps", "utc_offset_hours", default=0.0)))
    strip = manifest.get("electricity", "building_id_strip_prefix") or ""
    unit_rule = manifest.get("electricity", "unit_conversion")
    resolution = int(manifest.get("electricity", "resolution_minutes", default=60))

    appliance_layout = manifest.get("electricity", "layout") == "wide_appliance"
    columns = [c for c in wide.columns if c != "timestamp"]
    local = wide["timestamp"]
    utc = (local - offset).dt.tz_localize("UTC")
    values = _convert(wide[columns].to_numpy(dtype="float64"), unit_rule)
    present = ~np.isnan(values)
    row_idx, col_idx = np.nonzero(present)

    site_id = str(sites["site_id"].iloc[0])
    end_use_map = manifest.get("end_use_map", default={}) or {}

    if appliance_layout:
        # Every column is an end-use channel of ONE building, so the building
        # repeats and the *stream* varies -- the inverse of the other layouts.
        single = str(manifest.get("electricity", "building_id"))
        building_ids = [single]
        stream_names = [f"{single}.{_slug(c)}" for c in columns]
        row_building = [single] * len(col_idx)
        row_stream = [stream_names[i] for i in col_idx]
    else:
        raw_ids = [(c.replace(strip, "") if strip else c) for c in columns]
        prefix = manifest.get("electricity", "building_id_prefix") or ""
        building_ids = [f"{prefix}{b}" if prefix and not b.startswith(prefix) else b
                        for b in raw_ids]
        stream_names = [ecdm.stream_id(b) for b in building_ids]
        row_building = [building_ids[i] for i in col_idx]
        row_stream = [stream_names[i] for i in col_idx]

    long = pd.DataFrame({
        "timestamp_utc": utc.to_numpy()[row_idx],
        "building_id": pd.Categorical(row_building),
        "value": values[row_idx, col_idx],
        "quality_flag": pd.Categorical(["OK"] * len(row_idx)),
        "year": local.dt.year.to_numpy()[row_idx],
    })
    long["stream_id"] = pd.Categorical(row_stream)

    target_root = snapshot / "electricity" / f"dataset={dataset_id}" / f"site_id={site_id}"
    for year, frame in long.groupby("year", sort=True):
        target = target_root / f"year={year}"
        target.mkdir(parents=True, exist_ok=True)
        ordered = frame.sort_values(["building_id", "timestamp_utc"])
        pq.write_table(
            pa.Table.from_pandas(
                ordered[["timestamp_utc", "stream_id", "building_id",
                         "value", "quality_flag"]], preserve_index=False),
            target / "part-000.parquet", compression="zstd")
        report.partitions += 1
        report.rows += len(ordered)

    expected = len(wide)
    level = manifest.get("electricity", "stream_level", default="whole_building")
    stream_rows, quality_rows = [], []
    for j, column in enumerate(columns):
        building_id = building_ids[0] if appliance_layout else building_ids[j]
        stream_name = stream_names[j]
        valid = int(present[:, j].sum())
        coverage = valid / expected if expected else 0.0
        stream_rows.append({
            "stream_id": stream_name, "building_id": building_id,
            "site_id": site_id, "dataset_id": dataset_id,
            "stream_level": level,
            "end_use": end_use_map.get(column) if appliance_layout else None,
            "unit": "kWh", "resolution_minutes": resolution,
            "start_timestamp": utc.iloc[0], "end_timestamp": utc.iloc[-1],
            "n_observations": valid, "expected_observations": expected,
            "coverage_fraction": coverage, "source_stream_name": columns[j],
            "parquet_path": f"electricity/dataset={dataset_id}/site_id={site_id}",
            "iri": ecdm.stream_iri(dataset_id, stream_name)})
        quality_rows.append({
            "stream_id": stream_name,
            "coverage_fraction": coverage, "missing_fraction": 1 - coverage,
            "zero_masked_fraction": 0.0, "duplicate_count": 0,
            "longest_gap_hours": float(_longest_gap(present[:, j])),
            "first_valid_timestamp": utc.iloc[0], "last_valid_timestamp": utc.iloc[-1],
            "quality_status": _status(coverage),
            "dst_duplicate_hours": 0, "dst_missing_hours": 0})

    country = str(sites["country"].iloc[0])
    canonical, region = ecdm.canonicalize_country(country)
    site_frame = ecdm.conform(pd.DataFrame([{
        "site_id": site_id, "dataset_id": dataset_id,
        "site_name": sites["site_name"].iloc[0], "city": sites["city"].iloc[0],
        "state_region": region, "country_canonical": canonical,
        "country_raw": country, "country_code": ecdm.COUNTRY_CODES.get(canonical),
        "latitude": sites["latitude"].iloc[0], "longitude": sites["longitude"].iloc[0],
        "timezone": sites["timezone"].iloc[0],
        "utc_offset_hours": float(sites["utc_offset_hours"].iloc[0]),
        "climate_zone": None, "weather_source_id": None,
        "n_buildings": len(building_ids),
        "iri": ecdm.site_iri(dataset_id, site_id)}]), "sites")

    building_frame = ecdm.conform(pd.DataFrame([{
        "building_id": b, "site_id": site_id, "dataset_id": dataset_id,
        "building_type_canonical": ecdm.canonicalize_building_type(
            buildings.loc[buildings.building_id.str.endswith(b.split("_")[-1]),
                          "building_type"].iloc[0]
            if (buildings.building_id.str.endswith(b.split("_")[-1])).any()
            else "Residential"),
        "building_type_raw": "Residential", "building_subtype": None,
        "floor_area_m2": None, "year_built": None, "number_of_floors": None,
        "occupants": None, "heating_type_canonical": None,
        "heating_type_raw": None, "industry": None, "subindustry": None,
        "latitude": sites["latitude"].iloc[0], "longitude": sites["longitude"].iloc[0],
        "timezone": sites["timezone"].iloc[0], "city": sites["city"].iloc[0],
        "state_region": region, "country_canonical": canonical,
        "iri": ecdm.building_iri(dataset_id, b)} for b in building_ids]), "buildings")

    coverage_by_building: dict[str, float] = {}
    for row in stream_rows:
        coverage_by_building[row["building_id"]] = max(
            coverage_by_building.get(row["building_id"], 0.0),
            row["coverage_fraction"])
    floor = float(manifest.get("capabilities", "min_coverage", default=0.9))
    capability_frame = ecdm.conform(pd.DataFrame([{
        "building_id": b,
        "can_consumption_analysis": coverage_by_building[b] > 0,
        "can_eui": False,                      # no floor areas in REFIT
        "can_weather_analysis": False,         # no weather source declared
        "can_occupancy_normalisation": False,  # no occupancy
        "can_peer_benchmark": False,           # needs floor area
        "can_forecasting": coverage_by_building[b] >= floor,
        "can_spatial_analysis": bool(canonical),
        "has_coordinates": True} for b in building_ids]), "capabilities")

    dataset_frame = ecdm.conform(pd.DataFrame([{
        "dataset_id": dataset_id,
        "dataset_name": manifest.get("dataset", "name"),
        "dataset_version": SNAPSHOT_VERSION,
        "description": manifest.get("dataset", "description"),
        "source_url": manifest.get("dataset", "source_url"),
        "license": manifest.get("dataset", "license"),
        "citation": manifest.get("dataset", "citation"),
        "building_sector": manifest.get("dataset", "sector")}]), "datasets")

    _append(snapshot, "datasets", dataset_frame, "dataset_id")
    _append(snapshot, "sites", site_frame, "site_id")
    _append(snapshot, "buildings", building_frame, "building_id")
    _append(snapshot, "electricity_streams",
            ecdm.conform(pd.DataFrame(stream_rows), "electricity_streams"), "stream_id")
    _append(snapshot, "data_quality",
            ecdm.conform(pd.DataFrame(quality_rows), "data_quality"), "stream_id")
    _append(snapshot, "capabilities", capability_frame, "building_id")

    if appliance_layout:
        report.appliance_streams = sum(1 for r in stream_rows
                                       if r["stream_level"] == "appliance")
        report.end_uses = sorted({r["end_use"] for r in stream_rows if r["end_use"]})
        if manifest.get("electricity", "unit_assumption_verified") is False:
            report.problems.append(
                "unit conversion is a declared but UNVERIFIED assumption "
                f"({manifest.get('electricity', 'unit_conversion')}); "
                "check it against the source documentation")
    report.buildings = len(building_ids)
    report.sites = 1
    report.streams = len(stream_rows)
    report.capabilities = {c: int(capability_frame[c].fillna(False).sum())
                           for c in capability_frame.columns if c.startswith("can_")}
    report.unsupported = [c for c, n in report.capabilities.items() if n == 0]
    return report


def _append(snapshot: Path, table: str, frame: pd.DataFrame, key: str) -> None:
    """Append rows to a catalog table, replacing any row with the same key."""
    path = snapshot / "catalog" / f"{table}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[~existing[key].isin(frame[key])]
        for column in existing.columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = pd.concat([existing, frame[existing.columns]], ignore_index=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in name.lower())
    return "_".join(filter(None, keep.split("_")))


def _longest_gap(mask: np.ndarray) -> int:
    missing = ~mask
    if not missing.any():
        return 0
    padded = np.concatenate(([0], missing.view(np.int8), [0]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max())


def _status(coverage: float) -> str:
    if coverage >= 0.99:
        return "excellent"
    if coverage >= 0.95:
        return "good"
    if coverage >= 0.80:
        return "usable"
    return "poor" if coverage > 0 else "unusable"
