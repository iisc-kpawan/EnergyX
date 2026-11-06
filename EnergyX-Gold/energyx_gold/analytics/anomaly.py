"""Anomaly scores for gold tasks."""

from __future__ import annotations

import numpy as np

MODIFIED_Z_THRESHOLD = 3.5


def modified_zscore(values, reference=None) -> np.ndarray:
    """Iglewicz and Hoaglin's modified z-score: 0.6745 * (x - median) / MAD.

    The median and MAD come from `reference` when given (for example, daytime hours), otherwise from
    `values`. When the MAD is 0, 1.253314 * mean absolute deviation is used instead. If that is 0 too,
    every score is 0.
    """
    values = np.asarray(values, dtype=float)
    reference = values if reference is None else np.asarray(reference, dtype=float)
    reference = reference[~np.isnan(reference)]
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    if mad > 0:
        return 0.6745 * (values - median) / mad
    mean_ad = np.mean(np.abs(reference - median))
    if mean_ad > 0:
        return (values - median) / (1.253314 * mean_ad)
    return np.zeros_like(values)
