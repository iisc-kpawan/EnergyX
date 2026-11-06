"""Run-length helpers shared by the Q08 to Q10 checks. Numpy only."""

from __future__ import annotations

import numpy as np


def runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in `mask`."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[1::2] - edges[::2]


def inner_gaps(valid: np.ndarray) -> np.ndarray:
    """Lengths of the runs of missing hours between the first and the last valid hour."""
    if not valid.any():
        return np.array([], dtype=int)
    first, last = np.flatnonzero(valid)[[0, -1]]
    return runs(~valid[first:last + 1])
