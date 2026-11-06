"""Central paths and constants for EnergyX.

Everything that touches the filesystem resolves through here so that a
snapshot can be relocated or versioned without editing call sites.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen snapshot identifier. Every answer is computed against this and cites it.
SNAPSHOT_VERSION = "EnergyX-BDG2-v1"

# --- Scope constants (locked by design discussion) -------------------------
# The only meter channel read is electricity; BDG2's others are ignored. Site
# weather and building metadata are ingested as context.
ENERGY_TYPE = "electricity"
EXCLUDED_METER_TYPES = (
    "chilledwater", "steam", "hotwater", "gas", "water", "irrigation", "solar",
)

# Controlled vocabulary for stream granularity (ECDM).
STREAM_LEVELS = ("whole_building", "submeter", "circuit", "appliance")

# Quality flags carried alongside every reading.
QUALITY_OK = "OK"                    # value present in both raw and cleaned
QUALITY_ZERO_MASKED = "ZERO_MASKED"  # raw reading was exactly 0, masked by cleaning
QUALITY_FLAGS = (QUALITY_OK, QUALITY_ZERO_MASKED)

PLANNING_ENGINES = ("legacy", "structured", "shadow")


def planning_engine(value: str | None = None) -> str:
    """Configured planning path; legacy remains the rollback-safe default."""
    selected = value or os.environ.get("ENERGYX_PLANNING_ENGINE", "legacy")
    if selected not in PLANNING_ENGINES:
        raise ValueError(
            f"planning engine must be one of {', '.join(PLANNING_ENGINES)}"
        )
    return selected


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout for one EnergyX installation."""

    root: Path

    @property
    def source_bdg2(self) -> Path:
        return self.root / "BDG2"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def snapshot(self) -> Path:
        return self.data / SNAPSHOT_VERSION

    @property
    def electricity(self) -> Path:
        return self.snapshot / "electricity"

    @property
    def weather(self) -> Path:
        return self.snapshot / "weather"

    @property
    def aggregates(self) -> Path:
        return self.snapshot / "aggregates"

    @property
    def catalog(self) -> Path:
        return self.snapshot / "catalog"

    @property
    def registry(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        """Run-scoped artifact store: handles, traces, spilled frames."""
        return self.root / "runs"


PATHS = Paths(Path(os.environ.get("ENERGYX_ROOT", REPO_ROOT)))
