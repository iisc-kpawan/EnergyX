"""Load-shape vector index.

Similarity is the one relation that is genuinely expensive to recompute and
cheap to store, so representations live beside the graph rather than being
rebuilt per query: a 24-dimension normalised daily profile per building, held as
Parquet, searched by cosine.

The index is a **projection like the graph** -- rebuilt from the aggregates, never
authoritative. Each vector is registered in the graph as an ``ekg:Representation``
so a fact can point at the representation it was computed from.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS, SNAPSHOT_VERSION
from . import ontology as o

KIND_PROFILE = "daily_profile_24h"


@dataclass
class SearchHit:
    building_id: str
    similarity: float
    rank: int


class VectorIndex:
    """Cosine search over normalised daily load shapes."""

    def __init__(self, path: Path | None = None,
                 snapshot_version: str = SNAPSHOT_VERSION):
        # Deliberately outside graph/: KnowledgeStore.build() removes that
        # directory wholesale, which silently deleted the index.
        self.path = Path(path) if path else PATHS.snapshot / "vectors.parquet"
        self.snapshot_version = snapshot_version
        self._ids: list[str] | None = None
        self._matrix: np.ndarray | None = None

    # -- build -------------------------------------------------------------
    def build(self, db) -> dict:
        # Sum across streams: a building with appliance channels has many
        # streams, and its load shape is their total, not any one of them.
        frame = db.df("""
            SELECT building_id, hour_of_day, SUM(mean_demand) AS mean_demand
            FROM profile_daily GROUP BY 1, 2 ORDER BY 1, 2""")
        wide = frame.pivot(index="building_id", columns="hour_of_day",
                           values="mean_demand").dropna()
        matrix = wide.to_numpy(dtype=float)
        # Normalise by each building's own mean so shape, not magnitude, drives
        # similarity -- a small office and a large one can share a load shape.
        means = matrix.mean(axis=1, keepdims=True)
        matrix = np.divide(matrix, means, out=np.zeros_like(matrix), where=means != 0)
        out = pd.DataFrame(matrix, columns=[f"h{h:02d}" for h in range(matrix.shape[1])])
        out.insert(0, "building_id", wide.index)
        out["snapshot_version"] = self.snapshot_version
        out["kind"] = KIND_PROFILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.path, index=False, compression="zstd")
        self._ids, self._matrix = None, None
        return {"vectors": len(out), "dimension": matrix.shape[1],
                "path": str(self.path), "kind": KIND_PROFILE}

    def graph_triples(self) -> list[tuple]:
        """Register each vector in the graph as a Representation node."""
        self._load()
        triples: list[tuple] = []
        for building_id in self._ids or []:
            node = f"{o.EKGD}representation/{KIND_PROFILE}/{building_id}"
            target = f"{o.EKGD}building/bdg2/{building_id}"
            triples += [
                (node, f"{o.RDF}type", o.term("Representation")),
                (node, o.term("representationKind"), (KIND_PROFILE, "string")),
                (node, o.term("dimension"), (24, "integer")),
                (node, o.term("vectorRef"), (str(self.path), "string")),
                (target, o.term("hasRepresentation"), node),
                (node, o.prov_term("assertionType"), ("derived", "string")),
                (node, o.prov_term("snapshotVersion"), (self.snapshot_version, "string")),
            ]
        return triples

    # -- search ------------------------------------------------------------
    def _load(self) -> None:
        if self._matrix is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"no vector index at {self.path}; build it first")
        frame = pd.read_parquet(self.path)
        stale = frame["snapshot_version"].iloc[0] != self.snapshot_version
        if stale:
            raise ValueError(f"vector index was built for "
                             f"{frame['snapshot_version'].iloc[0]!r}, not "
                             f"{self.snapshot_version!r}; rebuild it")
        self._ids = frame["building_id"].tolist()
        columns = [c for c in frame.columns if c.startswith("h")]
        self._matrix = frame[columns].to_numpy(dtype=float)

    @property
    def size(self) -> int:
        self._load()
        return len(self._ids or [])

    def search(self, building_id: str, k: int = 10,
               candidates: list[str] | None = None) -> list[SearchHit]:
        self._load()
        assert self._ids is not None and self._matrix is not None
        if building_id not in self._ids:
            raise KeyError(f"{building_id} has no representation in the index")
        index = self._ids.index(building_id)
        matrix, ids = self._matrix, self._ids
        if candidates:
            keep = [i for i, b in enumerate(ids) if b in set(candidates) or i == index]
            matrix, ids = matrix[keep], [ids[i] for i in keep]
            index = ids.index(building_id)
        centred = matrix - matrix.mean(axis=1, keepdims=True)
        target = centred[index]
        denominator = np.linalg.norm(centred, axis=1) * np.linalg.norm(target)
        score = np.divide(centred @ target, denominator,
                          out=np.zeros(len(ids)), where=denominator != 0)
        order = [i for i in np.argsort(-score) if i != index][:k]
        return [SearchHit(building_id=ids[i], similarity=float(score[i]), rank=r + 1)
                for r, i in enumerate(order)]
