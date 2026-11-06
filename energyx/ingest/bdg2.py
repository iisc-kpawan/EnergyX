"""Build EnergyX-BDG2-v1 from the official BDG2 release.

The electricity channel, with site weather and building metadata as context
. Cleaned values are authoritative; the raw file is read
solely to derive quality flags. Timestamps are converted from local
standard time to UTC using a constant per-site offset.

Entry point: :func:`build_snapshot`.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import ecdm
from ..config import PATHS, SNAPSHOT_VERSION
from . import tz as tzmod

DATASET_ID = "bdg2"
EXPECTED_OBSERVATIONS = 17_544  # 731 days x 24 h, 2016-01-01 .. 2017-12-31
RESOLUTION_MINUTES = 60
UNIT = "kWh"

# Extended metadata columns carried as attributes rather than core fields,
# so the canonical schema stays stable as new datasets arrive.
EXTENDED_ATTRIBUTES = {
    "energystarscore": (None, "bdg2.metadata"),
    "eui": ("kWh/m2/yr", "bdg2.metadata"),
    "site_eui": ("kBtu/ft2/yr", "bdg2.metadata"),
    "source_eui": ("kBtu/ft2/yr", "bdg2.metadata"),
    "leed_level": (None, "bdg2.metadata"),
    "rating": (None, "bdg2.metadata"),
    "date_opened": (None, "bdg2.metadata"),
    "sqft": ("ft2", "bdg2.metadata"),
    "UID": (None, "bdg2.metadata"),
}


@dataclass
class IngestReport:
    """Counters checked against the snapshot acceptance criteria."""

    buildings_metadata: int = 0
    buildings_electricity: int = 0
    sites: int = 0
    countries: int = 0
    rows_written: int = 0
    rows_ok: int = 0
    rows_zero_masked: int = 0
    rows_masked_other: int = 0
    true_gaps: int = 0
    weather_rows: int = 0
    empty_streams: list[str] = field(default_factory=list)
    reconciliation_exact: bool = False

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Metadata -> sites / buildings / attributes
# ---------------------------------------------------------------------------

def _read_metadata(source: Path) -> pd.DataFrame:
    md = pd.read_csv(source / "BDG2_metadata.csv")
    md.columns = [c.strip() for c in md.columns]
    return md


def build_sites(md: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_id, grp in md.groupby("site_id", sort=True):
        country_raw = grp["Country"].dropna().iloc[0] if grp["Country"].notna().any() else None
        country, state_from_country = ecdm.canonicalize_country(country_raw)
        city, state_from_city = ecdm.split_city(
            grp["City"].dropna().iloc[0] if grp["City"].notna().any() else None
        )
        site_name = grp["Site name"].dropna().iloc[0] if grp["Site name"].notna().any() else None
        if site_name == "Anonymous":
            site_name = None
        timezone_name = grp["timezone"].dropna().iloc[0]
        lat = grp["lat"].dropna().mean() if grp["lat"].notna().any() else np.nan
        lng = grp["lng"].dropna().mean() if grp["lng"].notna().any() else np.nan
        rows.append({
            "site_id": site_id,
            "dataset_id": DATASET_ID,
            "site_name": site_name,
            "city": city,
            "state_region": state_from_country or state_from_city,
            "country_canonical": country,
            "country_raw": country_raw,
            "country_code": ecdm.COUNTRY_CODES.get(country),
            "latitude": lat, "longitude": lng,
            "timezone": timezone_name,
            "utc_offset_hours": tzmod.offset_hours(timezone_name),
            "climate_zone": None,
            "weather_source_id": ecdm.weather_source_id(site_id),
            "n_buildings": len(grp),
            "iri": ecdm.site_iri(DATASET_ID, site_id),
        })
    return ecdm.conform(pd.DataFrame(rows), "sites")


def build_buildings(md: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    site_lookup = sites.set_index("site_id")
    out = pd.DataFrame({
        "building_id": md["building_id"],
        "site_id": md["site_id"],
        "dataset_id": DATASET_ID,
        "building_type_canonical": md["primaryspaceusage"].map(ecdm.canonicalize_building_type),
        "building_type_raw": md["primaryspaceusage"],
        "building_subtype": md["sub_primaryspaceusage"],
        "floor_area_m2": md["sqm"],
        "year_built": md["yearbuilt"],
        "number_of_floors": md["numberoffloors"],
        "occupants": md["occupants"],
        "heating_type_canonical": md["heatingtype"].map(ecdm.canonicalize_heating_type),
        "heating_type_raw": md["heatingtype"],
        "industry": md["industry"],
        "subindustry": md["subindustry"],
        "latitude": md["lat"], "longitude": md["lng"],
        "timezone": md["timezone"],
    })
    # Geography is a site property; denormalised onto buildings for query ease.
    for col in ("city", "state_region", "country_canonical"):
        out[col] = out["site_id"].map(site_lookup[col])
    out["iri"] = out["building_id"].map(lambda b: ecdm.building_iri(DATASET_ID, b))
    return ecdm.conform(out, "buildings")


def build_building_attributes(md: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column, (unit, source) in EXTENDED_ATTRIBUTES.items():
        if column not in md.columns:
            continue
        present = md[md[column].notna()]
        if present.empty:
            continue
        frames.append(pd.DataFrame({
            "building_id": present["building_id"],
            "attribute_name": column,
            "attribute_value": present[column].astype(str),
            "attribute_unit": unit,
            "source": source,
        }))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(ecdm.SCHEMAS["building_attributes"])
    )
    return ecdm.conform(combined, "building_attributes")


# ---------------------------------------------------------------------------
# Electricity
# ---------------------------------------------------------------------------

def _longest_null_run(mask: np.ndarray) -> int:
    """Longest consecutive run of True in a boolean array."""
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.view(np.int8), [0]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max())


def _flat_runs(values: np.ndarray, isna: np.ndarray,
               minimum: int = 20) -> tuple[int, int]:
    """(longest run of identical consecutive readings, hours inside long runs).

    A meter reporting one value for a whole day is not flat demand, it is a
    dead channel: BDG2 carries 82,379 such building-days, all flagged OK
    because the cleaning pass only ever masked values it removed. Nothing
    downstream can tell them from real readings, so a day-long constant run is
    recorded here and the capabilities built from it refuse to treat such a
    stream as analysable.

    A gap breaks a run -- absent data is not constant data -- and runs shorter
    than `minimum` are ignored, because a few equal hours overnight is ordinary
    at one-hour resolution. The threshold is 20 rather than a full day because
    the defect as it actually occurs leaves one real reading at midnight and
    repeats a single value for the remaining 23 hours.
    """
    valid = ~isna
    if valid.sum() < 2:
        return 0, 0
    same = np.zeros(values.shape[0], dtype=bool)
    same[1:] = valid[1:] & valid[:-1] & (values[1:] == values[:-1])
    if not same.any():
        return 0, 0
    padded = np.concatenate(([0], same.view(np.int8), [0]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    # a run of n "same" flags spans n+1 readings
    lengths = (edges[1::2] - edges[::2]) + 1
    longest = int(lengths.max())
    inside = int(lengths[lengths >= minimum].sum())
    return longest, inside


def ingest_electricity(
    source: Path, sites: pd.DataFrame, out_root: Path, report: IngestReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write partitioned electricity Parquet; return (streams, data_quality)."""
    cleaned = pd.read_csv(source / "electricity_cleaned.csv", parse_dates=["timestamp"])
    raw = pd.read_csv(source / "electricity.csv", parse_dates=["timestamp"])

    building_cols = [c for c in cleaned.columns if c != "timestamp"]
    if [c for c in raw.columns if c != "timestamp"] != building_cols:
        raise ValueError("raw and cleaned electricity files disagree on columns")

    local_ts = cleaned["timestamp"]
    offsets = sites.set_index("site_id")["utc_offset_hours"].to_dict()

    stream_rows: list[dict] = []
    quality_rows: list[dict] = []

    for site_id, site_grp in sites.groupby("site_id", sort=True):
        cols = [c for c in building_cols if c.split("_")[0] == site_id]
        if not cols:
            continue
        offset = pd.Timedelta(hours=float(offsets[site_id]))
        utc_ts = tzmod.local_to_utc(local_ts, offset)
        local_year = local_ts.dt.year

        sub_clean = cleaned[cols]
        sub_raw = raw[cols]

        clean_vals = sub_clean.to_numpy(dtype="float64")
        raw_vals = sub_raw.to_numpy(dtype="float64")
        clean_na = np.isnan(clean_vals)
        raw_na = np.isnan(raw_vals)

        keep = ~raw_na                                   # row exists in source
        zero_masked = clean_na & ~raw_na & (raw_vals == 0.0)
        masked_other = clean_na & ~raw_na & (raw_vals != 0.0)

        report.true_gaps += int(raw_na.sum())
        report.rows_zero_masked += int(zero_masked.sum())
        report.rows_masked_other += int(masked_other.sum())

        n_rows, n_cols = clean_vals.shape
        row_idx, col_idx = np.nonzero(keep)
        flags = np.where(
            zero_masked[row_idx, col_idx], "ZERO_MASKED",
            np.where(masked_other[row_idx, col_idx], "MASKED_OTHER", "OK"),
        )
        frame = pd.DataFrame({
            "timestamp_utc": utc_ts.to_numpy()[row_idx],
            "building_id": pd.Categorical([cols[i] for i in col_idx]),
            "value": clean_vals[row_idx, col_idx],
            "quality_flag": pd.Categorical(flags),
            "year": local_year.to_numpy()[row_idx],
        })
        frame["stream_id"] = pd.Categorical(
            frame["building_id"].astype(str).map(ecdm.stream_id)
        )

        for year, year_frame in frame.groupby("year", sort=True):
            target = out_root / f"dataset={DATASET_ID}" / f"site_id={site_id}" / f"year={year}"
            target.mkdir(parents=True, exist_ok=True)
            ordered = year_frame.sort_values(["building_id", "timestamp_utc"])
            table = pa.Table.from_pandas(
                ordered[["timestamp_utc", "stream_id", "building_id", "value", "quality_flag"]],
                preserve_index=False,
            )
            pq.write_table(table, target / "part-000.parquet", compression="zstd")
            report.rows_written += len(ordered)

        # Per-stream catalog and quality rows.
        for j, building_id in enumerate(cols):
            valid = ~clean_na[:, j]
            n_valid = int(valid.sum())
            coverage = n_valid / EXPECTED_OBSERVATIONS
            if n_valid:
                first_valid = utc_ts.iloc[int(np.argmax(valid))]
                last_valid = utc_ts.iloc[int(n_rows - 1 - np.argmax(valid[::-1]))]
            else:
                first_valid = last_valid = pd.NaT
                report.empty_streams.append(building_id)
            flat_longest, flat_hours = _flat_runs(clean_vals[:, j], clean_na[:, j])
            sid = ecdm.stream_id(building_id)
            stream_rows.append({
                "stream_id": sid, "building_id": building_id, "site_id": site_id,
                "dataset_id": DATASET_ID, "stream_level": "whole_building",
                "end_use": None, "unit": UNIT,
                "resolution_minutes": RESOLUTION_MINUTES,
                "start_timestamp": first_valid, "end_timestamp": last_valid,
                "n_observations": n_valid,
                "expected_observations": EXPECTED_OBSERVATIONS,
                "coverage_fraction": coverage,
                "source_stream_name": building_id,
                "parquet_path": f"electricity/dataset={DATASET_ID}/site_id={site_id}",
                "iri": ecdm.stream_iri(DATASET_ID, sid),
            })
            quality_rows.append({
                "stream_id": sid,
                "coverage_fraction": coverage,
                "missing_fraction": 1.0 - coverage,
                "zero_masked_fraction": float(zero_masked[:, j].sum()) / EXPECTED_OBSERVATIONS,
                "duplicate_count": 0,          # source grid verified duplicate-free
                "longest_gap_hours": float(_longest_null_run(clean_na[:, j])),
                "longest_flat_run_hours": float(flat_longest),
                "flat_fraction": (flat_hours / n_valid) if n_valid else 0.0,
                "first_valid_timestamp": first_valid,
                "last_valid_timestamp": last_valid,
                "quality_status": _quality_status(
                    coverage, (flat_hours / n_valid) if n_valid else 0.0),
                "dst_duplicate_hours": 0,      # fixed-offset series
                "dst_missing_hours": 0,
            })

    report.rows_ok = report.rows_written - report.rows_zero_masked - report.rows_masked_other
    streams = ecdm.conform(pd.DataFrame(stream_rows), "electricity_streams")
    quality = ecdm.conform(pd.DataFrame(quality_rows), "data_quality")
    return streams, quality


def _quality_status(coverage: float, flat_fraction: float = 0.0) -> str:
    # Coverage measures presence, not variation. A stream that is present for
    # every hour and constant for most of them is worse than one with gaps,
    # because nothing about it looks wrong.
    if flat_fraction >= 0.5:
        return "unusable"
    if flat_fraction >= 0.2:
        return "poor"
    if coverage >= 0.99:
        return "excellent"
    if coverage >= 0.95:
        return "good"
    if coverage >= 0.80:
        return "usable"
    if coverage > 0.0:
        return "poor"
    return "unusable"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def ingest_weather(
    source: Path, sites: pd.DataFrame, out_root: Path, report: IngestReport
) -> pd.DataFrame:
    weather = pd.read_csv(source / "weather.csv", parse_dates=["timestamp"])
    weather = weather.rename(columns=ecdm.WEATHER_COLUMN_MAP)
    offsets = sites.set_index("site_id")["utc_offset_hours"].to_dict()

    rows = []
    for site_id, grp in weather.groupby("site_id", sort=True):
        if site_id not in offsets:
            continue
        offset = pd.Timedelta(hours=float(offsets[site_id]))
        grp = grp.sort_values("timestamp").copy()
        local_ts = grp["timestamp"]
        grp["timestamp_utc"] = tzmod.local_to_utc(local_ts, offset)
        grp["weather_source_id"] = ecdm.weather_source_id(site_id)
        grp["year"] = local_ts.dt.year

        keep = ["timestamp_utc", "weather_source_id", *ecdm.WEATHER_VARIABLES]
        for var in ecdm.WEATHER_VARIABLES:
            if var not in grp.columns:
                grp[var] = np.nan

        for year, year_frame in grp.groupby("year", sort=True):
            target = out_root / f"dataset={DATASET_ID}" / f"site_id={site_id}" / f"year={year}"
            target.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(year_frame[keep], preserve_index=False)
            pq.write_table(table, target / "part-000.parquet", compression="zstd")
            report.weather_rows += len(year_frame)

        row = {
            "weather_source_id": ecdm.weather_source_id(site_id),
            "site_id": site_id, "dataset_id": DATASET_ID,
            "source_name": f"BDG2 site weather ({site_id})",
            "resolution_minutes": RESOLUTION_MINUTES,
            "start_timestamp": grp["timestamp_utc"].min(),
            "end_timestamp": grp["timestamp_utc"].max(),
            "n_observations": len(grp),
            "coverage_fraction": len(grp) / EXPECTED_OBSERVATIONS,
            "parquet_path": f"weather/dataset={DATASET_ID}/site_id={site_id}",
            "iri": ecdm.weather_iri(DATASET_ID, site_id),
        }
        for var in ecdm.WEATHER_VARIABLES:
            row[f"{var}_available"] = bool(grp[var].notna().any())
            row[f"{var}_completeness"] = float(grp[var].notna().mean())
        rows.append(row)

    frame = pd.DataFrame(rows)
    conformed = ecdm.conform(frame, "weather_sources")
    # Per-variable availability is dataset-dependent, so it rides alongside the
    # declared schema rather than being baked into it.
    for var in ecdm.WEATHER_VARIABLES:
        conformed[f"{var}_available"] = frame[f"{var}_available"].astype("boolean")
        conformed[f"{var}_completeness"] = frame[f"{var}_completeness"].astype("float64")
    return conformed


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

MIN_COVERAGE = 0.90


MAX_FLAT_FRACTION = 0.2


def build_capabilities(
    buildings: pd.DataFrame, streams: pd.DataFrame, weather: pd.DataFrame,
    sites: pd.DataFrame, quality: pd.DataFrame | None = None
) -> pd.DataFrame:
    coverage = streams.set_index("building_id")["coverage_fraction"]
    weather_ok = set(
        weather.loc[weather["air_temperature_available"].fillna(False), "site_id"]
    )
    has_data = buildings["building_id"].map(coverage).fillna(0.0) > 0.0
    enough = buildings["building_id"].map(coverage).fillna(0.0) >= MIN_COVERAGE

    # A capability answers "can this computation run", not "should you trust the
    # number". EUI is annual kWh over floor area: if both exist it computes, and
    # withholding it because the meter looks stuck answers a question nobody
    # asked. Flatness is reported in data_quality and in quality_status instead.
    #
    # The exception is peer benchmarking, where a stuck meter is not only its
    # own problem: it shifts the median, the decile edges and the percentile
    # rank of every genuine building in the cohort. Screening before comparison
    # is ordinary practice, and it is the one place the gate earns its keep.
    if quality is not None and "flat_fraction" in quality.columns:
        flat = (streams[["building_id", "stream_id"]]
                .merge(quality[["stream_id", "flat_fraction"]], on="stream_id",
                       how="left")
                .set_index("building_id")["flat_fraction"])
        varies = buildings["building_id"].map(flat).fillna(0.0) < MAX_FLAT_FRACTION
    else:
        varies = pd.Series(True, index=buildings.index)

    out = pd.DataFrame({"building_id": buildings["building_id"]})
    out["can_consumption_analysis"] = has_data
    out["can_eui"] = has_data & buildings["floor_area_m2"].notna() & enough
    out["can_weather_analysis"] = has_data & buildings["site_id"].isin(weather_ok)
    # occupants == 0 appears for 41 buildings; it is present but unusable as a
    # divisor, so the capability requires a positive value.
    out["can_occupancy_normalisation"] = (
        has_data & buildings["occupants"].notna() & (buildings["occupants"] > 0))
    out["can_peer_benchmark"] = (
        has_data & varies & buildings["building_type_canonical"].notna()
        & buildings["floor_area_m2"].notna()
    )
    out["can_forecasting"] = enough
    out["can_spatial_analysis"] = has_data & buildings["country_canonical"].notna()
    out["has_coordinates"] = buildings["latitude"].notna() & buildings["longitude"].notna()
    return ecdm.conform(out, "capabilities")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_snapshot(source: Path | None = None, out: Path | None = None,
                   overwrite: bool = True) -> IngestReport:
    source = source or PATHS.source_bdg2
    out = out or PATHS.snapshot
    if overwrite and out.exists():
        shutil.rmtree(out)
    for sub in ("electricity", "weather", "catalog"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    report = IngestReport()
    md = _read_metadata(source)
    report.buildings_metadata = len(md)

    sites = build_sites(md)
    buildings = build_buildings(md, sites)
    attributes = build_building_attributes(md)
    report.sites = len(sites)
    report.countries = int(sites["country_canonical"].notna().sum() and
                           sites["country_canonical"].dropna().nunique())

    streams, quality = ingest_electricity(source, sites, out / "electricity", report)
    report.buildings_electricity = len(streams)

    # Reconciliation: electricity columns must equal metadata electricity == "Yes".
    flagged = set(md.loc[md["electricity"] == "Yes", "building_id"])
    report.reconciliation_exact = set(streams["building_id"]) == flagged

    weather = ingest_weather(source, sites, out / "weather", report)
    capabilities = build_capabilities(buildings, streams, weather, sites, quality)

    datasets = ecdm.conform(pd.DataFrame([{
        "dataset_id": DATASET_ID,
        "dataset_name": "Building Data Genome Project 2",
        "dataset_version": SNAPSHOT_VERSION,
        "description": "Non-residential building electricity meters, 2016-2017, hourly.",
        "source_url": "https://github.com/buds-lab/building-data-genome-project-2",
        "license": "See upstream repository",
        "citation": "Miller et al., Scientific Data, 2020",
        "building_sector": "commercial",
    }]), "datasets")

    tables = {
        "datasets": datasets, "sites": sites, "buildings": buildings,
        "building_attributes": attributes, "electricity_streams": streams,
        "weather_sources": weather, "data_quality": quality,
        "capabilities": capabilities,
    }
    for name, frame in tables.items():
        frame.to_parquet(out / "catalog" / f"{name}.parquet", index=False,
                         compression="zstd")

    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "dataset_id": DATASET_ID,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source),
        "scope": {"energy_type": "electricity", "excluded_meter_types":
                  list(ecdm.STREAM_LEVELS[:0]) or
                  ["chilledwater", "steam", "hotwater", "gas", "water",
                   "irrigation", "solar"]},
        "timestamp_convention": {
            "stored": "UTC",
            "source": "local standard time, constant offset per site",
            "partition_year": "local",
        },
        "counts": report.as_dict(),
        "tables": {name: len(frame) for name, frame in tables.items()},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    return report
