"""Dataset onboarding tools (4) -- the BYOD path.

Deliberately outside the normal runtime path: onboarding runs once per dataset,
not per question, and it writes to the snapshot rather than reading from it.

The claim these support is **guided interoperability**, not magic: a user maps
their schema in a manifest, EnergyX validates it and reports honestly which
analyses the dataset can and cannot support.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..ingest import registry as manifest_registry
from .base import ToolResult, empty, insufficient, ok
from .registry import ToolContext, tool

READABLE = {".csv": "csv", ".parquet": "parquet", ".yaml": "yaml", ".yml": "yaml"}
CAPABILITY_REQUIREMENTS = {
    "consumption_analysis": [],
    "load_profile_analysis": [],
    "forecasting": [],
    "anomaly_detection": [],
    "eui_analysis": ["floor_area_m2"],
    "occupancy_normalisation": ["occupants"],
    "peer_benchmark": ["building_type", "floor_area_m2"],
    "spatial_analysis": ["city", "country"],
    "weather_analysis": ["__weather__"],
}


def _sniff(path: Path, rows: int = 200) -> dict:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            head = pd.read_csv(path, nrows=rows)
        elif suffix == ".parquet":
            head = pd.read_parquet(path).head(rows)
        else:
            return {"file": path.name, "kind": READABLE.get(suffix, "unknown")}
    except Exception as exc:  # noqa: BLE001
        return {"file": path.name, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "file": path.name,
        "kind": READABLE.get(suffix, "unknown"),
        "columns": [str(c) for c in head.columns][:60],
        "n_columns": int(head.shape[1]),
        "dtypes": {str(c): str(t) for c, t in list(head.dtypes.items())[:20]},
        "sample": head.head(2).astype(str).to_dict(orient="records"),
        "likely_timestamp": [str(c) for c in head.columns
                             if str(c).lower() in
                             {"timestamp", "datetime", "time", "date", "ts"}],
    }


@tool("onboarding")
def inspect_dataset(ctx: ToolContext, path: str, max_files: int = 25) -> ToolResult:
    """Survey a candidate dataset directory: files, schemas, samples, timestamps."""
    root = Path(path)
    if not root.exists():
        return insufficient("missing_path", f"no such path: {path}")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in READABLE)
    if not files:
        return empty(f"no CSV, Parquet or YAML files under {path}")
    reports = [_sniff(p) for p in files[:max_files]]
    manifest = next((p for p in files if p.name in ("energyx.yaml", "energyx.yml")), None)
    wide = [r for r in reports if r.get("n_columns", 0) > 50]
    return ok({
        "root": str(root),
        "n_files": len(files),
        "files": reports,
        "manifest_found": str(manifest) if manifest else None,
        "likely_layout": "wide" if wide else "long",
    }, evidence={"inspected": len(reports), "total_files": len(files),
                 "wide_candidates": [r["file"] for r in wide]})


@tool("onboarding")
def map_to_energyx_schema(ctx: ToolContext, path: str,
                          manifest_path: str | None = None) -> ToolResult:
    """Resolve a source schema onto the canonical model via its manifest.

    Where the manifest is silent, column names are matched against canonical
    fields and reported as *suggestions* with confidence -- never applied
    silently. Unmapped columns are listed so nothing disappears unnoticed.
    """
    root = Path(path)
    manifest_file = Path(manifest_path) if manifest_path else root / "energyx.yaml"
    if not manifest_file.exists():
        return insufficient("no_manifest",
                            f"no manifest at {manifest_file}; BYOD needs energyx.yaml")
    manifest = manifest_registry.load(manifest_file)

    canonical = {
        "building_id": ["building_id", "building", "bldg_id", "bldg", "house_id", "uid"],
        "site_id": ["site_id", "site", "campus", "location_id"],
        "timestamp": ["timestamp", "datetime", "time", "ts", "date"],
        "value": ["value", "consumption", "energy", "kwh", "reading", "demand"],
        "floor_area_m2": ["sqm", "area", "floor_area", "floor_area_m2"],
        "building_type": ["primaryspaceusage", "usage", "type", "building_type"],
        "occupants": ["occupants", "occupancy", "headcount"],
        "year_built": ["yearbuilt", "year_built", "construction_year"],
        "city": ["city", "town"], "country": ["country", "nation"],
        "timezone": ["timezone", "tz"],
    }
    declared: dict[str, str] = {}
    for section in ("electricity", "metadata", "weather"):
        node = manifest.data.get(section) or {}
        for key, value in node.items():
            if isinstance(value, str) and key in canonical:
                declared[key] = value

    suggestions, unmapped = [], []
    meta_path = root / (manifest.get("metadata", "path") or "")
    if meta_path.is_file():
        try:
            columns = [str(c) for c in pd.read_csv(meta_path, nrows=5).columns]
        except Exception:  # noqa: BLE001
            columns = []
        claimed = set(declared.values())
        for column in columns:
            if column in claimed:
                continue
            lowered = column.lower().strip()
            match = next((field for field, aliases in canonical.items()
                          if lowered in aliases), None)
            if match and match not in declared:
                suggestions.append({"source_column": column, "canonical_field": match,
                                    "confidence": 0.9, "basis": "exact alias match"})
            else:
                unmapped.append(column)

    return ok({"dataset_id": manifest.dataset_id, "declared_mapping": declared,
               "suggestions": suggestions,
               "unmapped_columns": unmapped[:40],
               "extensible_layer": "unmapped columns can be carried in "
                                   "building_attributes without schema change"},
              evidence={"manifest": str(manifest_file),
                        "declared": len(declared), "suggested": len(suggestions),
                        "unmapped": len(unmapped)})


@tool("onboarding")
def validate_dataset(ctx: ToolContext, path: str,
                     manifest_path: str | None = None) -> ToolResult:
    """Check a candidate dataset and report which analyses it can support.

    Returns a capability report rather than a pass/fail: a dataset without floor
    area is perfectly usable, just not for EUI.
    """
    root = Path(path)
    manifest_file = Path(manifest_path) if manifest_path else root / "energyx.yaml"
    if not manifest_file.exists():
        return insufficient("no_manifest", f"no manifest at {manifest_file}")
    manifest = manifest_registry.load(manifest_file)
    problems = manifest_registry.validate(manifest, root)

    checks, present = [], set()
    meta_rel = manifest.get("metadata", "path")
    n_buildings = n_sites = 0
    if meta_rel and (root / meta_rel).exists():
        meta = (pd.read_csv(root / meta_rel) if meta_rel.endswith(".csv")
                else pd.read_parquet(root / meta_rel))
        mapping = manifest.data.get("metadata", {})
        for field in ("building_id", "site_id", "building_type", "floor_area_m2",
                      "occupants", "city", "country", "timezone"):
            column = mapping.get(field)
            if column and column in meta.columns and meta[column].notna().any():
                present.add(field)
        bcol, scol = mapping.get("building_id"), mapping.get("site_id")
        if bcol in meta.columns:
            n_buildings = int(meta[bcol].nunique())
            checks.append({"check": "building_id unique",
                           "passed": bool(meta[bcol].is_unique)})
        if scol in meta.columns:
            n_sites = int(meta[scol].nunique())
    else:
        problems.append("metadata file missing; only measurement-only analyses possible")

    weather_rel = manifest.get("weather", "path")
    has_weather = bool(weather_rel and list(root.glob(weather_rel)))
    if has_weather:
        present.add("__weather__")
    checks.append({"check": "weather declared and present", "passed": has_weather})
    checks.append({"check": "timestamp convention declared",
                   "passed": bool(manifest.get("timestamps", "convention"))})

    supported, unsupported = [], []
    for capability, needs in CAPABILITY_REQUIREMENTS.items():
        missing = [n for n in needs if n not in present]
        (supported if not missing else unsupported).append(
            capability if not missing else {"capability": capability,
                                            "missing": missing})
    return ok({
        "dataset_id": manifest.dataset_id,
        "valid": not problems,
        "problems": problems,
        "checks": checks,
        "buildings": n_buildings, "sites": n_sites,
        "supported_capabilities": supported,
        "unsupported_capabilities": unsupported,
    }, evidence={"manifest": str(manifest_file),
                 "fields_present": sorted(f for f in present if not f.startswith("__"))})


@tool("onboarding")
def register_dataset(ctx: ToolContext, path: str, manifest_path: str | None = None,
                     dry_run: bool = True) -> ToolResult:
    """Register a validated dataset into the snapshot catalog.

    Defaults to ``dry_run`` because registration mutates a frozen snapshot, and a
    snapshot that changes underneath a result invalidates it.
    """
    validation = validate_dataset(ctx, path, manifest_path)
    if not validation.ok:
        return validation
    report = validation.data
    if report["problems"]:
        return insufficient("validation_failed",
                            f"{len(report['problems'])} problem(s) must be fixed "
                            "before registration",
                            evidence={"problems": report["problems"]})
    plan = {
        "dataset_id": report["dataset_id"],
        "would_write": [
            f"data/<snapshot>/electricity/dataset={report['dataset_id']}/...",
            f"data/<snapshot>/weather/dataset={report['dataset_id']}/...",
            "catalog/datasets.parquet (append)",
            "catalog/sites.parquet (append)",
            "catalog/buildings.parquet (append)",
            "catalog/electricity_streams.parquet (append)",
            "catalog/data_quality.parquet (append)",
            "catalog/capabilities.parquet (append)",
        ],
        "buildings": report["buildings"], "sites": report["sites"],
        "supported_capabilities": report["supported_capabilities"],
    }
    if dry_run:
        return ok({**plan, "registered": False,
                   "note": "dry run -- pass dry_run=false to write, which mints a "
                           "new snapshot version rather than mutating this one"},
                  evidence={"mode": "dry_run"})
    return insufficient("not_implemented",
                        "live registration mints a new snapshot version; that "
                        "path lands with multi-dataset federation",
                        evidence={"plan": plan})
