"""Helpers shared by the category 10 tasks (Q46 to Q50).

The method is in docs/ml-protocol.md, "Clustering and similarity". A profile is a building's mean
valid reading per local hour of day (0 to 23), min-max scaled to 0-1 by its own minimum and maximum.
A flat profile becomes all zeros. When a question lets k vary, the highest silhouette score wins,
scores are compared at 12 decimals and a tie goes to the smaller k.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from energyx_gold.analytics import clustering, series
from energyx_gold.core.context import GoldContext

DECIMALS = 6
N_INIT = 10
MAX_ITER = 50  # tslearn TimeSeriesKMeans only
TIE_DIGITS = 12  # silhouette scores equal at this many decimals count as a tie
HOURS = range(24)


def rounded(value) -> float:
    """A float rounded to the result precision."""
    return round(float(value), DECIMALS)


def period_bounds(context: GoldContext) -> tuple[datetime, datetime]:
    """The snapshot period (2016-2017) as start (inclusive) and end (exclusive), site-local time."""
    return context.period.start, context.period.end + timedelta(hours=1)


def year_bounds(year: int) -> tuple[datetime, datetime]:
    """Start (inclusive) and end (exclusive) of a local calendar year."""
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def describe_period(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} (end exclusive), site-local"


def hourly_means(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean valid reading per local hour of day, one row per building, 24 columns. Fails on a missing hour."""
    means = clustering.mean_profile(hourly, by="hour").reindex(columns=list(HOURS))
    incomplete = means.index[means.isna().any(axis=1)].tolist()
    if incomplete:
        raise ValueError(f"profiles with an hour that has no valid reading: {incomplete}")
    return means


def scaled_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    """The mean hourly profiles, each row min-max scaled to 0-1."""
    return clustering.minmax_rows(hourly_means(hourly))


def cohort_profiles(context: GoldContext, start: datetime, end: datetime, **filters) -> pd.DataFrame:
    """Scaled mean hourly profiles over [start, end) for the buildings matching the metadata filters
    that have at least one valid reading (anywhere in the snapshot)."""
    ids = series.cohort(context, **filters)
    return scaled_profiles(series.hourly(context, ids, start, end))


def complete_days(readings: pd.Series) -> pd.DataFrame:
    """One building's complete local days as rows of 24 hourly readings (hour 0 to 23).

    A day is complete when all 24 of its hours have a valid reading.
    """
    readings = readings.dropna()
    frame = pd.DataFrame({"day": readings.index.normalize(), "hour": readings.index.hour, "value": readings.to_numpy()})
    wide = frame.pivot(index="day", columns="hour", values="value").reindex(columns=list(HOURS))
    return wide[wide.notna().all(axis=1)]


def kmeans_model(matrix: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(matrix)


def kmeans_labels(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    return kmeans_model(matrix, k, seed).labels_


def ward_labels(matrix: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative clustering with Ward linkage on Euclidean distance. It has no random part."""
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(matrix)


def euclidean_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(matrix, labels))


def dtw_kmeans_labels(series_3d: np.ndarray, k: int, seed: int) -> np.ndarray:
    """tslearn TimeSeriesKMeans with DTW distance. `series_3d` is (series, time steps, 1)."""
    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_init=N_INIT, max_iter=MAX_ITER, random_state=seed)
    return model.fit_predict(series_3d)


def dtw_silhouette(series_3d: np.ndarray, labels: np.ndarray) -> float:
    from tslearn.clustering import silhouette_score

    return float(silhouette_score(series_3d, labels, metric="dtw"))


def best_k(
    matrix: np.ndarray,
    fit: Callable[[int], np.ndarray],
    k_range: range,
    score: Callable[[np.ndarray, np.ndarray], float] = euclidean_silhouette,
) -> tuple[int, dict[int, float], dict[int, np.ndarray]]:
    """The k with the highest silhouette score, every k's score, and every k's labels.

    `fit` returns the labels for one k. A k of at least the number of rows is skipped. A tie at
    TIE_DIGITS decimals goes to the smaller k.
    """
    scores: dict[int, float] = {}
    labels: dict[int, np.ndarray] = {}
    for k in k_range:
        if k >= len(matrix):
            break
        labels[k] = fit(k)
        scores[k] = score(matrix, labels[k])
    if not scores:
        raise ValueError(f"too few rows to cluster: {len(matrix)}")
    best = max(scores, key=lambda k: (round(scores[k], TIE_DIGITS), -k))
    return best, scores, labels


def silhouettes(scores: dict[int, float]) -> dict[str, float]:
    """Silhouette score for each k, for evidence."""
    return {str(k): rounded(score) for k, score in scores.items()}


def cluster_sizes(labels: np.ndarray) -> list[int]:
    """Cluster sizes, largest first."""
    return sorted((int(n) for n in np.bincount(labels)), reverse=True)


def members(ids: list[str], labels: np.ndarray) -> list[list[str]]:
    """Each cluster's sorted members, clusters ordered by size (largest first) and then by first member."""
    groups = [sorted(ids[i] for i in np.flatnonzero(labels == label)) for label in np.unique(labels)]
    return sorted(groups, key=lambda group: (-len(group), group[0]))
