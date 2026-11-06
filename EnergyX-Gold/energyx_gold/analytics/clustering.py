"""Load profiles, normalisation and K-Means for gold tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

K_RANGE = range(2, 11)
KMEANS_INIT = 10


def mean_profile(hourly: pd.DataFrame, by: str = "hour") -> pd.DataFrame:
    """Mean reading per building by hour of day (24 columns) or hour of week (168 columns, Monday 00:00 first)."""
    index = hourly.index
    key = index.hour if by == "hour" else index.dayofweek * 24 + index.hour
    return hourly.groupby(key).mean().T.rename_axis("building_id")


def minmax_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale each row to 0..1 by its own minimum and maximum. A flat row becomes all 0; missing values stay missing."""
    low, high = frame.min(axis=1), frame.max(axis=1)
    span = high - low
    scaled = frame.sub(low, axis=0).div(span.where(span != 0), axis=0)
    flat = span == 0
    scaled.loc[flat] = frame.loc[flat].where(frame.loc[flat].isna(), 0.0)
    return scaled


def kmeans(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=KMEANS_INIT, random_state=seed).fit_predict(matrix)


def best_k(matrix: np.ndarray, seed: int, k_range=K_RANGE) -> tuple[int, dict[int, float]]:
    """The number of clusters with the highest silhouette score, with every score. Ties go to the smaller k."""
    from sklearn.metrics import silhouette_score

    scores = {}
    for k in k_range:
        if k >= len(matrix):
            break
        scores[k] = float(silhouette_score(matrix, kmeans(matrix, k, seed)))
    best = max(scores, key=lambda k: (round(scores[k], 12), -k))
    return best, scores
