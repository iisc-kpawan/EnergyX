"""Schema registry: load and validate dataset manifests.

A manifest states how a source dataset maps onto the canonical model. It is the
contract that lets a new dataset join EnergyX without touching agent or tool
code, and the thing a BYOD user writes.

Validation is deliberately strict about the few fields the tools depend on and
permissive about everything else -- a dataset should not be rejected for lacking
metadata it never claimed to have.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import PATHS

REQUIRED_TOP = ("dataset", "electricity")
REQUIRED_DATASET = ("id", "name")
REQUIRED_ELECTRICITY = ("path", "timestamp", "unit")
REQUIRED_METADATA = ("path", "building_id", "site_id")
# "wide_appliance": columns are end-use channels of ONE building,
# rather than one column per building.
KNOWN_LAYOUTS = ("wide", "long", "wide_appliance")
KNOWN_CONVENTIONS = ("local_standard_time", "local_wall_clock", "utc")


@dataclass
class Manifest:
    path: Path
    data: dict

    @property
    def dataset_id(self) -> str:
        return self.data["dataset"]["id"]

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load(path: Path) -> Manifest:
    return Manifest(path=path, data=yaml.safe_load(path.read_text()))


def load_dataset(dataset_id: str, root: Path | None = None) -> Manifest:
    root = root or PATHS.registry
    return load(root / dataset_id / "energyx.yaml")


def validate(manifest: Manifest, source_dir: Path | None = None) -> list[str]:
    """Return a list of problems. Empty means the manifest is usable."""
    problems: list[str] = []
    data = manifest.data

    for key in REQUIRED_TOP:
        if key not in data:
            problems.append(f"missing top-level section: {key}")
    if problems:
        return problems

    for key in REQUIRED_DATASET:
        if not data["dataset"].get(key):
            problems.append(f"dataset.{key} is required")

    elec = data["electricity"]
    for key in REQUIRED_ELECTRICITY:
        if not elec.get(key):
            problems.append(f"electricity.{key} is required")
    layout = elec.get("layout", "long")
    if layout not in KNOWN_LAYOUTS:
        problems.append(f"electricity.layout must be one of {KNOWN_LAYOUTS}, got {layout!r}")

    meta = data.get("metadata")
    if meta:
        for key in REQUIRED_METADATA:
            if not meta.get(key):
                problems.append(f"metadata.{key} is required when metadata is declared")

    convention = manifest.get("timestamps", "convention", default="local_standard_time")
    if convention not in KNOWN_CONVENTIONS:
        problems.append(f"timestamps.convention must be one of {KNOWN_CONVENTIONS}")
    # A declared offset of 0 (GMT) is a declaration, not an absence. Testing
    # truthiness here rejected every UK and Irish dataset.
    has_offset = (manifest.get("timestamps", "offset_source") is not None
                  or manifest.get("timestamps", "utc_offset_hours") is not None)
    if convention != "utc" and not has_offset:
        problems.append("timestamps: non-UTC data needs offset_source or utc_offset_hours")

    if source_dir is not None:
        for label, rel in (("electricity.path", elec.get("path")),
                           ("metadata.path", (meta or {}).get("path")),
                           ("weather.path", manifest.get("weather", "path"))):
            if rel and not (source_dir / rel).exists():
                problems.append(f"{label} not found: {source_dir / rel}")
        qref = manifest.get("electricity", "quality_reference", "path")
        if qref and not (source_dir / qref).exists():
            problems.append(f"electricity.quality_reference.path not found: {qref}")

    return problems
