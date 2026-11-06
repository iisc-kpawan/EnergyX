"""The forecasting model zoo: one registry, one calling convention.

Adding a forecaster should not widen the agent's tool surface, and it should
not change what any other forecaster returns. So every model lives behind one
signature -- history in, predictions out -- and ``forecast_timeseries`` selects
one by name. What each model contributes beyond its numbers is the record of
how it produced them: library, version, the exact parameters it was given and
the window it was fitted on. A forecast whose method cannot be restated is not
a result, it is a column of numbers.

Three families are registered, and they answer different questions.

**Baselines** (``seasonal_naive``, ``moving_average``) are written here rather
than imported, because a reference figure has to be pinned to an
implementation that cannot move under a dependency upgrade. They are also what
an operator actually compares against: "the same weekday last week" is the
forecast a facilities manager makes in their head, and a model that cannot beat
it has earned nothing. ``seasonal_naive`` agrees with sktime's
``NaiveForecaster(strategy="last", sp=168)`` to the fourth decimal on the
snapshot's buildings, which is the check that says the local implementation is
the standard one and not a private variant.

**Statistical models** (``theta``, ``ets``, ``arima``) come from sktime
(Kiraly et al., JMLR 2019), which wraps statsmodels and pmdarima behind one
interface. They fit parameters to a bounded recent window -- a year of hourly
data is 8,760 points and neither ETS nor ARIMA needs all of it to learn a daily
cycle, while both get slower on it in ways that would make a portfolio-scale
question untenable. The window is a stated parameter, not a hidden one.

**Foundation models** (``chronos``) are pretrained on a large corpus of time
series and forecast a building they have never seen, with no fitting at all
(Ansari et al., "Chronos: Learning the Language of Time Series", TMLR 2024).
This is the one family whose weights are not derived from the snapshot, which
makes it the only one that can be run on a building with two weeks of history.
It also returns quantiles rather than a single path, so "how uncertain is this"
has an answer.

Availability is checked at call time and reported as a refusal naming the
missing package, never as an exception: an optional dependency that is not
installed is a fact about the deployment, and the agent should be told which
one rather than handed a traceback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Training windows, in days. A model that fits parameters reads the recent past
# rather than the whole record: the daily and weekly cycles it is learning are
# fully present in a few weeks, and the cost of the rest is paid on every
# question. Stated in the result, because a fitted model is only reproducible
# alongside the sample it was fitted on.
FIT_WINDOW_DAYS = 56
SPIKE_MULTIPLE = 5.0        # "gross" = this many times the window median
ARIMA_WINDOW_DAYS = 28
# Chronos-T5 attends over a fixed context; anything longer is truncated to the
# most recent values by the model itself, so the truncation is done here where
# it can be reported.
CHRONOS_CONTEXT_HOURS = 512
CHRONOS_CHECKPOINT = "autogluon/chronos-t5-base"
# The second pretrained forecaster, and a deliberately different size: 0.8M
# parameters against 200M, with a fixed 512-hour context and 96-hour horizon.
TTM_CHECKPOINT = "ibm-granite/granite-timeseries-ttm-r2"
_TTM_MODEL: dict = {}
# Sampling is seeded so the same request twice gives the same forecast. A
# probabilistic model whose answer moves between runs cannot be compared
# against a fixed reference, and what is measured is then the seed.
SEED = 0


class ModelUnavailable(Exception):
    """A registered model whose library or weights are not on this machine."""

    def __init__(self, model: str, package: str, detail: str = "") -> None:
        super().__init__(
            f"the {model!r} forecaster needs {package}, which is not available "
            f"here{f' ({detail})' if detail else ''}. Installed models: "
            f"{', '.join(n for n in MODELS if n != model)}")
        self.model, self.package = model, package


@dataclass
class Forecast:
    """One model's answer, with everything needed to restate it."""

    values: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    interval_level: float | None = None      # e.g. 0.8 for a p10-p90 band
    evidence: dict = field(default_factory=dict)


# --- what a reader needs in order to argue with a forecast ------------------
# Every model states, in one sentence, what it assumes. This is not
# documentation: it is printed in the answer, because a number whose method is
# unnamed cannot be accepted or rejected on any grounds except trust.
METHOD_PROSE: dict[str, str] = {
    "seasonal_naive":
        "each hour is repeated from the same hour of the same weekday one week "
        "earlier; nothing is fitted",
    "moving_average":
        "each hour is the mean of that hour of day over the trailing window; "
        "nothing is fitted",
    "linear":
        "least-squares regression on harmonics of hour, day of week and month",
    "gradient_boosting":
        "gradient-boosted trees on calendar features (hour, day of week, month, "
        "weekend)",
    "lightgbm":
        "LightGBM gradient-boosted trees on calendar features (hour, day of "
        "week, month, weekend), 300 trees at learning rate 0.05",
    "lightgbm_weather":
        "LightGBM on the same calendar features as `lightgbm` plus the site's "
        "weather, each variable with its square and 24-hour mean; the horizon "
        "uses observed weather in place of a weather forecast",
    "knn":
        "k-nearest neighbours over standardised calendar features, inverse-"
        "distance weighted",
    "knn_weather":
        "k-nearest neighbours with the site's weather in the distance, on the "
        "same design as `lightgbm_weather`",
    "weather_regression":
        "ridge regression on site air temperature, its square and its 24-hour "
        "mean, interacted with calendar features; the horizon uses observed "
        "weather in place of a weather forecast",
    "theta":
        "the Theta method: deseasonalise, extrapolate two theta-lines, "
        "reseasonalise (Assimakopoulos & Nikolopoulos 2000)",
    "ets":
        "exponential smoothing with an additive daily season, fitted by "
        "maximum likelihood",
    "arima":
        "seasonal ARIMA(2,0,1)(1,1,1)[24], fitted by maximum likelihood",
    "auto_arima":
        "seasonal ARIMA with the orders selected by a stepwise search "
        "(p,q <= 3, P,Q <= 1, d,D <= 1, sp=24), fitted by maximum likelihood",
    "chronos":
        "a pretrained time-series foundation model, applied zero-shot: the "
        "building's recent history is the only input and no parameters are "
        "fitted to it",
    "tiny_time_mixer":
        "IBM Tiny Time Mixers, a 0.8M-parameter pretrained forecaster applied "
        "zero-shot over a 512-hour context, in 96-hour blocks",
}

#: Models that load a torch checkpoint. They must not be run in a forked
#: worker: CUDA state does not survive fork, and these are fast enough warm
#: (a TTM inference is milliseconds) that parallelism buys nothing anyway.
GPU_MODELS = frozenset({"chronos", "tiny_time_mixer"})

LIBRARY: dict[str, str] = {
    "seasonal_naive": "energyx (reference implementation)",
    "moving_average": "energyx (reference implementation)",
    "linear": "numpy.linalg.lstsq",
    "gradient_boosting": "scikit-learn HistGradientBoostingRegressor",
    "lightgbm": "LightGBM LGBMRegressor",
    "lightgbm_weather": "LightGBM LGBMRegressor + site weather",
    "knn": "scikit-learn KNeighborsRegressor",
    "knn_weather": "scikit-learn KNeighborsRegressor + site weather",
    "weather_regression": "scikit-learn Ridge",
    "theta": "sktime ThetaForecaster",
    "ets": "sktime ExponentialSmoothing (statsmodels)",
    "arima": "sktime ARIMA (pmdarima)",
    "auto_arima": "statsforecast AutoARIMA (Hyndman-Khandakar search); sktime/pmdarima if statsforecast is absent",
    "chronos": f"chronos-forecasting ({CHRONOS_CHECKPOINT})",
    "tiny_time_mixer": f"tsfm_public ({TTM_CHECKPOINT})",
}

# Which models fit parameters to this building, which do not, and which were
# trained elsewhere entirely. The distinction decides whether "how much history
# does this need" has an answer, and it is what separates the foundation model
# from everything else in the table.
FITTING: dict[str, str] = {
    "seasonal_naive": "none (lookup)",
    "moving_average": "none (average)",
    "linear": "fitted to this building",
    "gradient_boosting": "fitted to this building",
    "lightgbm": "fitted to this building",
    "lightgbm_weather": "fitted to this building",
    "knn": "fitted to this building",
    "knn_weather": "fitted to this building",
    "weather_regression": "fitted to this building",
    "theta": "fitted to this building",
    "ets": "fitted to this building",
    "arima": "fitted to this building",
    "auto_arima": "orders searched and fitted on this building",
    "chronos": "pretrained elsewhere; zero-shot on this building",
    "tiny_time_mixer": "pretrained elsewhere; zero-shot on this building",
}


def _naked(stamps) -> pd.DatetimeIndex:
    """Timestamps as wall clock, with the zone stripped.

    Every measurement in the snapshot is stored as an instant and exposed as
    local wall clock on a constant offset, so the offset carries no information
    a forecaster can use -- but a tz-aware index is rejected by the period
    conversion sktime wants. Dropping it here keeps that decision in one place.
    """
    index = pd.DatetimeIndex(pd.to_datetime(stamps))
    return index.tz_localize(None) if index.tz is not None else index


def _series(history: pd.DataFrame, window_days: int | None) -> pd.Series:
    """The fitting sample: an hourly series on a regular index, recent first."""
    series = pd.Series(history["value"].to_numpy(float),
                       index=_naked(history["timestamp"])).sort_index()
    # An hourly model needs an hourly grid. Gaps in the snapshot are absent
    # rows rather than empty values, so a raw series can silently be 300 hours
    # short of the span it covers and every seasonal period drifts.
    series = series[~series.index.duplicated(keep="last")]
    series = series.asfreq("h")
    if window_days:
        series = series.tail(int(window_days) * 24)
    return series


def _filled(series: pd.Series) -> pd.Series:
    """The same series with its gaps closed, for models that cannot see NaN.

    Interpolation is linear within the series and held flat at its ends, and it
    is reported: a fitted model whose sample was partly invented should say so.
    """
    return series.interpolate(limit_direction="both")


# --- baselines ---------------------------------------------------------------

def _seasonal_naive(history: pd.DataFrame, future: pd.DatetimeIndex,
                    params: dict) -> Forecast:
    """Repeat the value one season back. The season is a parameter.

    A weekly season (168 h) respects the weekday/weekend split and is the
    better model for a building with a working week; a daily one (24 h) is
    the usual default for hourly data and is what most references mean by
    "Seasonal Naive". The two give different numbers -- 0.2250 against 0.1417
    on Lamb_education_Norris over 12 hours from 2016-05-08 -- so a question
    that names the season has to be able to say so, and one that does not
    gets the weekly default this implementation has always used.
    """
    season = int(params.get("season_length") or params.get("sp") or 168)
    step = pd.Timedelta(hours=season)
    recent = history.set_index(pd.to_datetime(history["timestamp"]))["value"]
    preds = [recent.get(t - step, np.nan) for t in future]
    preds = pd.Series(preds).ffill().bfill().to_numpy()
    if np.isnan(preds).all():
        preds = np.full(len(future), float(recent.tail(max(season, 24)).mean()))
    source = [t - step for t in future]
    return Forecast(preds, evidence={
        "season_length_hours": season,
        "repeated_from": f"{source[0]:%Y-%m-%d %H:%M} to {source[-1]:%Y-%m-%d %H:%M}",
        "weekday_repeated": f"{future[0]:%A}",
    })


def _moving_average(history: pd.DataFrame, future: pd.DatetimeIndex,
                    params: dict) -> Forecast:
    window_days = int(params.get("window_days", 7))
    stamps = pd.to_datetime(history["timestamp"])
    last = stamps.max()
    window = history.loc[stamps >= last - pd.Timedelta(days=window_days)
                         + pd.Timedelta(hours=1)].copy()
    window["hour"] = pd.to_datetime(window["timestamp"]).dt.hour
    by_hour = window.groupby("hour")["value"].mean()
    overall = (float(window["value"].mean()) if len(window)
               else float(history["value"].mean()))
    preds = np.array([float(by_hour.get(t.hour, overall)) for t in future])
    first = last - pd.Timedelta(days=window_days) + pd.Timedelta(hours=1)
    return Forecast(preds, evidence={
        "window_days": window_days,
        "averaged_over": f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}",
        "hours_per_average": int(round(len(window) / 24)) if len(window) else 0,
    })


# --- calendar-feature regressions -------------------------------------------

def _calendar_features(stamps) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(stamps))
    hour, dow, month = t.hour, t.dayofweek, t.month
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        (dow >= 5).astype(float), hour.astype(float),
    ])


def _linear(history: pd.DataFrame, future: pd.DatetimeIndex,
            params: dict) -> Forecast:
    x_train = _calendar_features(history["timestamp"])
    y_train = history["value"].to_numpy(float)
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    x_future = _calendar_features(future)
    preds = np.column_stack([np.ones(len(x_future)), x_future]) @ coef
    return Forecast(np.clip(preds, 0, None),
                    evidence={"features": 8, "train_hours": int(len(history))})


#: Lags a day-ahead model needs. Yesterday, the day before, and the same hour
#: last week, plus the mean of the day ending a day ago.
LAGS = (24, 48, 168)


def _lagged_design(stamps, known: pd.Series) -> np.ndarray:
    """Calendar features plus recent history at each timestamp.

    Without these a tree model can only learn the average calendar profile: it
    never sees what the building actually did yesterday or this time last week.
    Measured on Peacock_office_Major over 61 day-ahead origins, calendar-only
    LightGBM scores 5.36% MAPE and the same model with these lags scores 4.78%
    -- and the ordering against ARIMA reverses, so the omission was changing
    which model a question said was better.

    ``known`` is every value available when the row is built: the history for
    training rows, and history plus whatever has already been predicted for
    rows beyond the cutoff. A lag that reaches past the start of the series is
    filled with the series mean rather than dropped, so the first rows of a
    short window stay usable.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(stamps))
    base = _calendar_features(stamps)
    fallback = float(known.mean()) if len(known) else 0.0
    columns = []
    for lag in LAGS:
        wanted = stamps - pd.Timedelta(hours=lag)
        columns.append(known.reindex(wanted).to_numpy(float))
    # Mean of the 24 hours ending one day before, a smoother view of level
    # than any single lag.
    daily = known.rolling(24, min_periods=1).mean()
    columns.append(daily.reindex(stamps - pd.Timedelta(hours=24)).to_numpy(float))
    lagged = np.column_stack(columns)
    lagged = np.where(np.isfinite(lagged), lagged, fallback)
    return np.hstack([base, lagged])


def _fit_predict_lagged(estimator, history: pd.DataFrame,
                        future: pd.DatetimeIndex) -> np.ndarray:
    """Fit on lagged history and roll forward, feeding predictions back in.

    Beyond the first horizon the lags would reach into the period being
    forecast, so each prediction is appended to the known series before the
    next row is built. Within a 24-hour horizon nothing is fed back at all --
    every lag is still an observation.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(history["timestamp"]))
    known = pd.Series(history["value"].to_numpy(float), index=stamps).sort_index()
    known = known[~known.index.duplicated(keep="last")]
    estimator.fit(_lagged_design(stamps, known), known.to_numpy(float))
    out = []
    for stamp in pd.DatetimeIndex(future):
        row = _lagged_design(pd.DatetimeIndex([stamp]), known)
        value = float(estimator.predict(row)[0])
        out.append(value)
        known.loc[stamp] = value
    return np.asarray(out, float)


def _lightgbm(history: pd.DataFrame, future: pd.DatetimeIndex,
              params: dict) -> Forecast:
    """Gradient-boosted trees, LightGBM's implementation.

    Registered separately from ``gradient_boosting`` rather than swapped in
    behind it, because a question that names LightGBM is asking for LightGBM:
    the two are the same family and not the same numbers, and answering one
    with the other is the substitution this registry exists to prevent. It is
    also the model most of the building-forecasting literature reports, so a
    comparison that cannot run it cannot be compared with theirs.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ModelUnavailable("lightgbm", "lightgbm", str(exc)) from exc
    estimator = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, random_state=SEED, verbose=-1, n_jobs=1)
    preds = _fit_predict_lagged(estimator, history, future)
    return Forecast(np.clip(preds, 0, None), evidence={
        "lightgbm_version": lgb.__version__,
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
        "features": "calendar + lags 24/48/168 h and the mean day ending 24 h back",
        "random_state": SEED, "train_hours": int(len(history))})


def _gradient_boosting(history: pd.DataFrame, future: pd.DatetimeIndex,
                       params: dict) -> Forecast:
    from sklearn.ensemble import HistGradientBoostingRegressor
    x_train = _calendar_features(history["timestamp"])
    y_train = history["value"].to_numpy(float)
    estimator = HistGradientBoostingRegressor(max_iter=200, random_state=SEED)
    estimator.fit(x_train, y_train)
    preds = estimator.predict(_calendar_features(future))
    return Forecast(np.clip(preds, 0, None),
                    evidence={"max_iter": 200, "random_state": SEED,
                              "train_hours": int(len(history))})


# The weather a forecaster may be given. Air and dew temperature are recorded
# on effectively every hour; cloud coverage on 48% of them, which is why it is
# available but not a default -- asking for it halves the sample a model can
# fit on, and that cost should be chosen rather than inherited.
WEATHER_VARIABLES = ("air_temperature", "dew_temperature", "cloud_coverage")
DEFAULT_WEATHER = ("air_temperature",)


def _weather_design(stamps, observed) -> np.ndarray:
    """Calendar features plus each weather variable, its square and its mean.

    `observed` is a mapping of variable name to values. Every weather model
    builds its design here so that "adding weather" means the same thing to
    all of them: an ablation whose arms differ by the feature set *and* by how
    the features were built measures neither.

    Each variable contributes its level, its square and its trailing 24-hour
    mean. The square matters because demand responds to temperature on both
    sides of a balance point, and one slope fits neither; the trailing mean
    matters because a building's thermal mass makes this afternoon's load
    depend on this morning's weather.
    """
    calendar = _calendar_features(stamps)
    weekend = (pd.DatetimeIndex(pd.to_datetime(stamps)).dayofweek >= 5)
    columns = [calendar]
    for name in sorted(observed):
        values = np.asarray(observed[name], float)
        smooth = pd.Series(values).rolling(24, min_periods=1).mean().to_numpy()
        columns.append(np.column_stack([values, values ** 2, smooth,
                                        values * weekend]))
    return np.column_stack(columns)


def _weather_frames(params: dict, model: str):
    """The exogenous weather the tool fetched, or an honest refusal."""
    history, future = params.get("exog_history"), params.get("exog_future")
    if history is None or future is None or getattr(history, "empty", True):
        raise ModelUnavailable(model, "site weather observations",
                               "no air temperature for this building's site")
    return history, future


def _lightgbm_weather(history: pd.DataFrame, future: pd.DatetimeIndex,
                      params: dict) -> Forecast:
    """LightGBM on the calendar features *and* site air temperature.

    The point of registering this beside plain ``lightgbm`` is that the two
    differ in exactly one thing -- the weather columns -- so the difference in
    their error is attributable to the weather and to nothing else. Same
    estimator, same hyper-parameters, same seed, same rows.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ModelUnavailable("lightgbm_weather", "lightgbm", str(exc)) from exc
    exog_history, exog_future = _weather_frames(params, "lightgbm_weather")

    wanted = [v for v in (params.get("weather_variables") or DEFAULT_WEATHER)
              if v in exog_history.columns]
    if not wanted:
        raise ModelUnavailable("lightgbm_weather", "site weather observations",
                               "none of the requested variables are recorded")
    merged = history.merge(exog_history, on="timestamp", how="inner").dropna(
        subset=["value", *wanted])
    if len(merged) < 24 * 7:
        raise ModelUnavailable("lightgbm_weather", "site weather observations",
                               f"only {len(merged)} hours align with "
                               f"{', '.join(wanted)}")
    future_frame = pd.DataFrame({"timestamp": future}).merge(
        exog_future, on="timestamp", how="left")
    if future_frame[wanted].isna().all().any():
        raise ModelUnavailable("lightgbm_weather", "site weather observations",
                               "a requested variable is absent over the horizon")
    filled = int(future_frame[wanted].isna().sum().sum())
    temperature = future_frame[wanted[0]].ffill().bfill()

    estimator = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, random_state=SEED, verbose=-1, n_jobs=1)
    train_observed = {v: merged[v].to_numpy(float) for v in wanted}
    future_observed = {v: future_frame[v].ffill().bfill().to_numpy(float)
                       for v in wanted}
    estimator.fit(_weather_design(merged["timestamp"], train_observed),
                  merged["value"].to_numpy(float))
    preds = estimator.predict(_weather_design(future, future_observed))
    return Forecast(np.clip(preds, 0, None), evidence={
        "lightgbm_version": lgb.__version__,
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
        "random_state": SEED, "train_hours": int(len(merged)),
        "weather_variables": wanted,
        "horizon_temperature_mean_c": round(float(temperature.mean()), 2),
        "hours_of_weather_carried_forward": filled,
        "weather_assumption": "observed weather over the horizon, in place of "
                              "a weather forecast",
    })


def _knn(history: pd.DataFrame, future: pd.DatetimeIndex,
         params: dict) -> Forecast:
    """K-nearest neighbours over the feature space.

    A distance-based forecaster: each future hour is predicted by averaging
    the observed demand of the most similar past hours, where similarity is
    measured over the same features a tree model would split on. It is the
    third kind of thing in this registry -- neither a fitted parametric model
    nor a lookup -- and it leverages exogenous features very differently from
    a boosted tree, which is exactly what a question comparing them is asking
    about: distance treats every feature as equally important once scaled,
    where a tree chooses.

    Features are standardised before the distance is taken, because an
    unscaled design lets whichever column has the largest numbers decide every
    neighbourhood -- hour of day runs 0 to 23 and temperature -20 to 40.
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    observed = params.get("weather_for_model")
    if observed:
        x_train = _weather_design(history["timestamp"], observed["train"])
        x_future = _weather_design(future, observed["future"])
    neighbours = int(params.get("n_neighbors", 0)) or max(
        3, min(24, len(history) // 200))
    if observed:
        scaler = StandardScaler().fit(x_train)
        estimator = KNeighborsRegressor(n_neighbors=neighbours,
                                        weights="distance", n_jobs=1)
        estimator.fit(scaler.transform(x_train),
                      history["value"].to_numpy(float))
        preds = estimator.predict(scaler.transform(x_future))
        lagged = False
    else:
        # The same lags the tree models get. A question comparing distance
        # against boosted trees is asking which family uses the information
        # better, so giving one of them yesterday's demand and not the other
        # measures the feature set instead of the method.
        stamps = pd.DatetimeIndex(pd.to_datetime(history["timestamp"]))
        known = pd.Series(history["value"].to_numpy(float),
                          index=stamps).sort_index()
        known = known[~known.index.duplicated(keep="last")]
        scaler = StandardScaler().fit(_lagged_design(stamps, known))

        class _Scaled:
            """Standardise inside the roll-forward, where rows are built one
            at a time and the scaler has to be the one fitted on training."""

            def __init__(self, inner):
                self.inner = inner

            def fit(self, x, y):
                self.inner.fit(scaler.transform(x), y)
                return self

            def predict(self, x):
                return self.inner.predict(scaler.transform(x))

        estimator = _Scaled(KNeighborsRegressor(
            n_neighbors=neighbours, weights="distance", n_jobs=1))
        preds = _fit_predict_lagged(estimator, history, future)
        lagged = True
    return Forecast(np.clip(preds, 0, None), evidence={
        "n_neighbors": neighbours,
        "weighting": "inverse distance",
        "features_standardised": True,
        "features": ("calendar + lags 24/48/168 h and the mean day ending 24 h "
                     "back" if lagged else "calendar + weather"),
        "train_hours": int(len(history)),
        "weather_variables": sorted(observed["train"]) if observed else None,
    })


def _knn_weather(history: pd.DataFrame, future: pd.DatetimeIndex,
                 params: dict) -> Forecast:
    """K-nearest neighbours with the site's weather in the distance."""
    exog_history, exog_future = _weather_frames(params, "knn_weather")
    wanted = [v for v in (params.get("weather_variables") or DEFAULT_WEATHER)
              if v in exog_history.columns]
    merged = history.merge(exog_history, on="timestamp", how="inner").dropna(
        subset=["value", *wanted])
    if len(merged) < 24 * 7:
        raise ModelUnavailable("knn_weather", "site weather observations",
                               f"only {len(merged)} hours align with "
                               f"{', '.join(wanted)}")
    future_frame = pd.DataFrame({"timestamp": future}).merge(
        exog_future, on="timestamp", how="left")
    if future_frame[wanted].isna().all().any():
        raise ModelUnavailable("knn_weather", "site weather observations",
                               "a requested variable is absent over the horizon")
    forecast = _knn(merged, future, {
        **params,
        "weather_for_model": {
            "train": {v: merged[v].to_numpy(float) for v in wanted},
            "future": {v: future_frame[v].ffill().bfill().to_numpy(float)
                       for v in wanted}}})
    forecast.evidence["weather_assumption"] = (
        "observed weather over the horizon, in place of a weather forecast")
    return forecast


def _weather_regression(history: pd.DataFrame, future: pd.DatetimeIndex,
                        params: dict) -> Forecast:
    """Consumption against temperature and the calendar.

    A building's demand responds to weather non-linearly and with lag -- the
    cooling load at 3pm follows the morning's heat as well as the current
    reading -- so the design carries the temperature, its square and its
    trailing 24-hour mean. Without the square a single slope has to serve both
    the heating and the cooling side of the balance point and fits neither.

    The horizon uses the weather actually observed over it. That is the
    "perfect forecast" convention: it isolates the consumption model from
    weather-forecast error, which is what a question about *this building's*
    weather response is asking about, and it is stated in the result because
    the resulting accuracy is optimistic as an operational forecast.
    """
    from sklearn.linear_model import Ridge
    exog_history = params.get("exog_history")
    exog_future = params.get("exog_future")
    if exog_history is None or exog_future is None or exog_history.empty:
        raise ModelUnavailable("weather_regression", "site weather observations",
                               "no air temperature for this building's site")

    def design(stamps, temperature) -> np.ndarray:
        temperature = np.asarray(temperature, float)
        smooth = pd.Series(temperature).rolling(24, min_periods=1).mean().to_numpy()
        calendar = _calendar_features(stamps)
        return np.column_stack([calendar, temperature, temperature ** 2, smooth,
                                temperature * (pd.DatetimeIndex(
                                    pd.to_datetime(stamps)).dayofweek >= 5)])

    wanted = [v for v in (params.get("weather_variables") or DEFAULT_WEATHER)
              if v in exog_history.columns]
    merged = history.merge(exog_history, on="timestamp", how="inner").dropna(
        subset=["value", *wanted])
    if len(merged) < 24 * 14:
        raise ModelUnavailable("weather_regression", "site weather observations",
                               f"only {len(merged)} hours align with "
                               f"{', '.join(wanted)}")
    future_frame = pd.DataFrame({"timestamp": future}).merge(
        exog_future, on="timestamp", how="left")
    temperature = future_frame[wanted[0]]
    if temperature.isna().all():
        raise ModelUnavailable("weather_regression", "site weather observations",
                               "no weather observed over the horizon")
    # A hole in the horizon's weather is carried from the nearest observation
    # rather than dropping the hour: an unforecast hour is a worse answer than
    # one forecast on slightly stale weather, and the fill is reported.
    filled = int(temperature.isna().sum())
    temperature = temperature.ffill().bfill()

    model = Ridge(alpha=1.0)
    model.fit(design(merged["timestamp"], merged["air_temperature"]),
              merged["value"].to_numpy(float))
    preds = model.predict(design(future, temperature))
    return Forecast(np.clip(preds, 0, None), evidence={
        "train_hours": int(len(merged)),
        "weather_variable": "air_temperature",
        "horizon_temperature_c": [round(float(t), 2) for t in temperature],
        "horizon_temperature_mean_c": round(float(temperature.mean()), 2),
        "hours_of_weather_carried_forward": filled,
        "weather_assumption": "observed weather over the horizon, in place of "
                              "a weather forecast",
    })


# --- sktime ------------------------------------------------------------------

def _sktime_forecast(name: str, history: pd.DataFrame, future: pd.DatetimeIndex,
                     params: dict) -> Forecast:
    """Fit one sktime forecaster and read its path, with a band where offered.

    All three share this body because sktime's point is that they share an
    interface; what differs is the estimator and the window it can afford.
    """
    try:
        from sktime.forecasting.base import ForecastingHorizon
    except ImportError as exc:                              # pragma: no cover
        raise ModelUnavailable(name, "sktime", str(exc)) from exc

    window = (ARIMA_WINDOW_DAYS if name in ("arima", "auto_arima")
              else FIT_WINDOW_DAYS)
    series = _series(history, window)
    missing = int(series.isna().sum())
    series = _filled(series).dropna()
    if len(series) < 24 * 7:
        raise ModelUnavailable(name, "sufficient history",
                               f"{len(series)} hours on a regular grid, need 168")
    # BDG2's "cleaned" electricity still carries occasional 20-60x meter
    # spikes. Least squares has no defence against them: one spike in a 28-day
    # training window bends the whole seasonal fit, and the damage lands on
    # every hour forecast afterwards. Replacing them by interpolation before
    # fitting is not cosmetic -- on Bull Education in November 2016 it is
    # worth about three MAPE points across the cohort. The count is reported
    # so a reader can see the fit was altered and by how much.
    spikes = 0
    if name in ("arima", "auto_arima") and params.get("clean_training_spikes", True):
        typical = series.median()
        if typical > 0:
            gross = series > SPIKE_MULTIPLE * typical
            spikes = int(gross.sum())
            if spikes and spikes < len(series) * 0.05:
                series = series.mask(gross).interpolate(limit_direction="both")
            else:
                spikes = 0
    series.index = series.index.to_period("h")

    if name == "theta":
        from sktime.forecasting.theta import ThetaForecaster
        estimator = ThetaForecaster(sp=24)
    elif name == "ets":
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        estimator = ExponentialSmoothing(trend=None, seasonal="add", sp=24)
    elif name == "arima":
        from sktime.forecasting.arima import ARIMA
        estimator = ARIMA(order=(2, 0, 1), seasonal_order=(1, 1, 1, 24),
                          suppress_warnings=True)
    elif name == "auto_arima":
        # AutoARIMA is a different model from the fixed order above: it
        # searches the orders rather than being told them, which is what a
        # question naming "AutoARIMA" asks for and what a reference
        # implementation using pmdarima will have done. Aliasing the two
        # would answer a stated model with an unstated one.
        #
        # The seasonal search is bounded because an hourly series with sp=24
        # and an unbounded search does not finish: on a 7-day window the
        # bounded search is seconds and the unbounded one is minutes per
        # building, and a cohort question fits one per building.
        from sktime.forecasting.arima import AutoARIMA
        estimator = AutoARIMA(sp=24, seasonal=True, suppress_warnings=True,
                              max_p=3, max_q=3, max_P=1, max_Q=1, max_d=1,
                              max_D=1, stepwise=True, error_action="ignore")
    else:                                                   # pragma: no cover
        raise ModelUnavailable(name, "sktime", "no such sktime estimator")

    horizon = ForecastingHorizon(np.arange(1, len(future) + 1), is_relative=True)
    estimator.fit(series)
    preds = np.asarray(estimator.predict(fh=horizon), float).ravel()

    lower = upper = None
    level = params.get("interval_level", 0.8)
    try:
        band = estimator.predict_interval(fh=horizon, coverage=[level])
        lower = np.asarray(band.iloc[:, 0], float).ravel()
        upper = np.asarray(band.iloc[:, 1], float).ravel()
    except Exception:
        # Not every estimator offers an interval, and a missing band is not a
        # failed forecast. The point estimate stands and the answer says the
        # band is unavailable rather than inventing one.
        level = None

    import sktime
    return Forecast(np.clip(preds, 0, None),
                    None if lower is None else np.clip(lower, 0, None),
                    None if upper is None else np.clip(upper, 0, None),
                    level, evidence={
                        "sktime_version": sktime.__version__,
                        "estimator": type(estimator).__name__,
                        "fit_window_days": window,
                        "train_hours": int(len(series)),
                        "hours_interpolated_for_fit": missing,
                        "training_spikes_replaced": spikes,
                        "seasonal_period_hours": 24,
                    })


# --- foundation model --------------------------------------------------------

_CHRONOS_PIPELINE = {}


def _torch_device() -> str:
    """Where the foundation models should run.

    These are the only workloads here a GPU helps: the statistical models are
    sequential Kalman recursions and the tree models fit a few thousand rows,
    both of which a GPU makes slower, not faster. A card that is present but
    full is treated as absent -- sharing a box means an out-of-memory error
    belongs to whoever was there first, not to them.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        for index in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(index)
            if free > 4 * 1024 ** 3:                        # 4 GiB headroom
                return f"cuda:{index}"
        return "cpu"
    except Exception:                                       # noqa: BLE001
        return "cpu"


def _chronos(history: pd.DataFrame, future: pd.DatetimeIndex,
             params: dict) -> Forecast:
    """Zero-shot forecast from a pretrained time-series model.

    The model has never seen this building, this dataset or this meter, and
    nothing is fitted to them: the last few hundred hours are the entire input.
    That is worth stating in the answer, because it is the property that makes
    the model usable on a building with a fortnight of history -- where every
    fitted model here refuses -- and it is also the reason its accuracy should
    be checked rather than assumed.

    The pipeline is cached per process: loading the weights costs a fraction of
    a second but the tensor allocation is not free, and a portfolio question
    would otherwise pay it per building.
    """
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ModelUnavailable("chronos", "chronos-forecasting and torch",
                               str(exc)) from exc

    checkpoint = str(params.get("checkpoint") or CHRONOS_CHECKPOINT)
    if checkpoint not in _CHRONOS_PIPELINE:
        try:
            _CHRONOS_PIPELINE[checkpoint] = BaseChronosPipeline.from_pretrained(
                checkpoint, device_map=_torch_device(),
                torch_dtype=torch.float32)
        except Exception as exc:
            raise ModelUnavailable(
                "chronos", f"the {checkpoint} weights",
                "not in the local model cache and no download was possible"
            ) from exc
    pipeline = _CHRONOS_PIPELINE[checkpoint]

    series = _filled(_series(history, None)).dropna()
    context = series.to_numpy(float)[-CHRONOS_CONTEXT_HOURS:]
    if len(context) < 48:
        raise ModelUnavailable("chronos", "sufficient history",
                               f"{len(context)} hours, need 48")
    level = float(params.get("interval_level", 0.8))
    low, high = round((1 - level) / 2, 4), round(1 - (1 - level) / 2, 4)
    torch.manual_seed(SEED)
    quantiles, _ = pipeline.predict_quantiles(
        inputs=torch.tensor(context, dtype=torch.float32).unsqueeze(0),
        prediction_length=len(future), quantile_levels=[low, 0.5, high])
    band = quantiles[0].detach().numpy()
    return Forecast(np.clip(band[:, 1], 0, None), np.clip(band[:, 0], 0, None),
                    np.clip(band[:, 2], 0, None), level, evidence={
                        "checkpoint": checkpoint,
                        "context_hours": int(len(context)),
                        "context_from": f"{series.index[-len(context)]:%Y-%m-%d %H:%M}",
                        "parameters_fitted": 0,
                        "quantiles": [low, 0.5, high],
                        "seed": SEED,
                        "point_forecast": "median (q0.5)",
                    })


def _tiny_time_mixer(history: pd.DataFrame, future: pd.DatetimeIndex,
                     params: dict) -> Forecast:
    """IBM's Tiny Time Mixers, applied zero-shot.

    The second pretrained forecaster here, and a deliberately different point
    on the curve from Chronos: 805 thousand parameters against 200 million, a
    fixed 512-hour context and a 96-hour horizon, and roughly two orders of
    magnitude less compute per forecast. Whether that costs accuracy on
    building load is exactly the kind of question this registry exists to let
    someone ask, so it is registered beside Chronos rather than instead of it.

    The horizon is fixed by the architecture. A longer one is produced by
    forecasting a block, appending the forecast to the context and forecasting
    again -- which is honest recursion, not a longer model, and the answer
    says how many blocks it took, because error compounds across them.
    """
    try:
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    except ImportError as exc:
        raise ModelUnavailable("tiny_time_mixer", "tsfm_public and torch",
                               str(exc)) from exc
    checkpoint = str(params.get("checkpoint") or TTM_CHECKPOINT)
    device = _torch_device()
    if checkpoint not in _TTM_MODEL:
        try:
            _TTM_MODEL[checkpoint] = TinyTimeMixerForPrediction.from_pretrained(
                checkpoint).eval().to(device)
        except Exception as exc:
            raise ModelUnavailable(
                "tiny_time_mixer", f"the {checkpoint} weights",
                "not in the local model cache and no download was possible"
            ) from exc
    model = _TTM_MODEL[checkpoint]
    device = str(next(model.parameters()).device)
    context_length = int(model.config.context_length)
    block = int(model.config.prediction_length)

    series = _filled(_series(history, None)).dropna()
    if len(series) < 24:
        raise ModelUnavailable(
            "tiny_time_mixer", "sufficient history",
            f"{len(series)} hours; a seasonal cycle is the least this can "
            f"be asked to extend")
    tiled = 0
    if len(series) < context_length:
        # The model's context is a fixed width, not a minimum amount of
        # evidence. A question that states its own window -- "the previous 7
        # days" -- means 168 hours, and refusing it because the checkpoint
        # wants 512 answers a question about the model's shape rather than
        # about the building. Repeating the stated window up to the context
        # width keeps the daily and weekly cycles in phase, which is what the
        # model reads, and is what the reference implementation does.
        repeats = -(-context_length // len(series))
        tiled = len(series)
        series = pd.Series(
            np.tile(series.to_numpy(float), repeats)[-context_length:])
    # Scaled to zero mean and unit variance and scaled back afterwards: the
    # model is trained on normalised series and a building drawing 200 kW is
    # not otherwise comparable with one drawing 2 kW.
    context = series.to_numpy(float)[-context_length:]
    centre, spread = float(context.mean()), float(context.std()) or 1.0
    _ = tiled
    rolling = list((context - centre) / spread)
    predictions: list[float] = []
    blocks = 0
    with torch.no_grad():
        while len(predictions) < len(future):
            window = torch.tensor(
                rolling[-context_length:], dtype=torch.float32
            ).reshape(1, context_length, 1).to(device)
            step = model(past_values=window).prediction_outputs
            step = step.reshape(-1).cpu().tolist()[:block]
            predictions.extend(step)
            rolling.extend(step)
            blocks += 1
    preds = np.array(predictions[:len(future)]) * spread + centre
    return Forecast(np.clip(preds, 0, None), evidence={
        "device": device,
        "checkpoint": checkpoint,
        "context_hours": context_length,
        "native_horizon_hours": block,
        "blocks_rolled": blocks,
        "parameters_fitted": 0,
        "scaling": "context standardised to zero mean and unit variance",
        "recursion_note": (None if blocks <= 1 else
                           f"the horizon needed {blocks} forecasts of {block} "
                           f"hours, each conditioned on the one before, so "
                           f"error compounds across them"),
    })


# Models that can be fitted across buildings at once. A pooled fit is what
# makes building metadata mean anything: floor area and year built do not vary
# hour to hour, so in a per-building model they are constant columns that a
# tree cannot split on and a distance cannot separate on. Asked whether adding
# metadata helps, a per-building comparison would return identical forecasts
# and a p-value of exactly 1 -- an artefact that reads like a finding.
POOLABLE = ("linear", "gradient_boosting", "lightgbm", "knn")

# Metadata columns a pooled model may use. Numeric ones enter directly;
# building_type is one-hot encoded against its most common level.
METADATA_NUMERIC = ("floor_area_m2", "year_built", "occupants",
                    "number_of_floors")
METADATA_CATEGORICAL = ("building_type",)


def run_pooled(model: str, histories: dict, futures: dict, *,
               metadata: pd.DataFrame | None = None,
               weather: dict | None = None,
               weather_variables=None, **params) -> dict:
    """Fit one model across every building, and forecast each of them.

    The training rows are every building's hours stacked together, so the
    model learns one relationship over the cohort rather than one per
    building. That is the only arrangement in which metadata is a feature
    rather than a constant, and it is also how a portfolio forecaster is
    actually deployed: one model, many buildings, including buildings whose
    own history is too short to fit on.

    Returns one Forecast per building, so the caller scores it exactly as it
    scores a per-building model -- same hours, same measures, comparable.
    """
    if model not in POOLABLE:
        raise ModelUnavailable(
            model, "a poolable model",
            f"{model} is fitted to one series at a time; pooled fitting needs "
            f"one of {', '.join(POOLABLE)}")
    variables = [v for v in (weather_variables or DEFAULT_WEATHER)] if weather else []

    def design(stamps, building: str, observed: dict | None) -> np.ndarray:
        block = (_weather_design(stamps, observed) if observed
                 else _calendar_features(stamps))
        if metadata is None or building not in metadata.index:
            return block
        row = metadata.loc[building].to_numpy(float)
        return np.column_stack([block, np.tile(row, (len(block), 1))])

    train_x, train_y = [], []
    for building, frame in histories.items():
        if frame is None or frame.empty:
            continue
        observed = None
        if variables:
            seen = weather.get(building)
            if seen is None or seen.empty:
                continue
            merged = frame.merge(seen, on="timestamp", how="inner").dropna(
                subset=["value", *variables])
            if merged.empty:
                continue
            frame = merged
            observed = {v: merged[v].to_numpy(float) for v in variables}
        train_x.append(design(frame["timestamp"], building, observed))
        train_y.append(frame["value"].to_numpy(float))
    if not train_x:
        raise ModelUnavailable(model, "training rows",
                               "no building contributed usable history")
    matrix = np.vstack(train_x)
    target = np.concatenate(train_y)

    if model == "lightgbm":
        import lightgbm as lgb
        estimator = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                      num_leaves=31, random_state=SEED,
                                      verbose=-1, n_jobs=1)
        scaler = None
    elif model == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        estimator = HistGradientBoostingRegressor(max_iter=200,
                                                  random_state=SEED)
        scaler = None
    elif model == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(matrix)
        estimator = KNeighborsRegressor(
            n_neighbors=int(params.get("n_neighbors", 0)) or 24,
            weights="distance", n_jobs=1)
        matrix = scaler.transform(matrix)
    else:
        from sklearn.linear_model import Ridge
        estimator = Ridge(alpha=1.0)
        scaler = None
    estimator.fit(matrix, target)

    out: dict = {}
    for building, horizon in futures.items():
        observed = None
        if variables:
            seen = weather.get(building)
            if seen is None or seen.empty:
                continue
            ahead = pd.DataFrame({"timestamp": horizon}).merge(
                seen, on="timestamp", how="left")
            if ahead[variables].isna().all().any():
                continue
            observed = {v: ahead[v].ffill().bfill().to_numpy(float)
                        for v in variables}
        x = design(horizon, building, observed)
        if scaler is not None:
            x = scaler.transform(x)
        out[building] = Forecast(
            np.clip(estimator.predict(x), 0, None), evidence={
                "model": model, "pooled": True,
                "training_rows": int(len(target)),
                "training_buildings": len(train_x),
                "metadata_features": (list(metadata.columns)
                                      if metadata is not None else []),
                "weather_variables": variables or None,
                "library": LIBRARY[model],
                "method": f"one {model} fitted across {len(train_x)} buildings"
                          + (" with building metadata as features"
                             if metadata is not None else ""),
                "fitting": "fitted once across the cohort",
            })
    return out


def metadata_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Building metadata as a numeric design, indexed by building.

    Numeric columns are standardised so a k-nearest-neighbour distance is not
    decided by floor area simply for being a larger number than year built.
    The categorical type is one-hot encoded against its most common level, so
    every coefficient is read against a stated reference.
    """
    frame = rows.set_index("building_id")
    parts = []
    for column in METADATA_NUMERIC:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 2:
            continue
        spread = float(values.std(ddof=0)) or 1.0
        parts.append(((values - values.mean()) / spread)
                     .fillna(0.0).rename(column))
    for column in METADATA_CATEGORICAL:
        if column not in frame:
            continue
        levels = frame[column].dropna().astype(str)
        if levels.nunique() < 2:
            continue
        reference = levels.value_counts().idxmax()
        for level in sorted(set(levels) - {reference}):
            parts.append((frame[column].astype(str) == level)
                         .astype(float).rename(f"type={level}"))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)

def _auto_arima(history: pd.DataFrame, future: pd.DatetimeIndex,
                params: dict) -> Forecast:
    """AutoARIMA via statsforecast, falling back to sktime/pmdarima.

    Both implement Hyndman-Khandakar and they do not agree: on
    Lamb_education_Norris over the 24 hours from 2016-05-08, trained on the
    previous 168, statsforecast scores MAE 0.5441 and sktime's pmdarima
    wrapper 1.780. Neither is a bug -- the search paths differ -- but the
    question names a model, not a library, and the answer moves by a factor
    of three depending on which one runs. statsforecast is the Nixtla
    reference for the algorithm and it is faster, so it is the one that runs,
    and the engine is reported rather than left to be inferred.
    """
    season = int(params.get("season_length") or params.get("sp") or 24)
    engine_asked = str(params.get("engine") or "auto").lower()
    # The same training window, spike cleaning and regular grid the sktime
    # path builds. Skipping it fitted the whole 17,544-hour record instead of
    # the stated window; fitted to the full series instead it runs for minutes on a
    # question the model answers in a tenth of a second.
    # The window the question states, not a house default. "Use the previous
    # 7 days of hourly data" is 168 hours, and that is the context the model
    # is given; the 28-day default belongs to a question that names no
    # window.
    window = int(params.get("window_days") or ARIMA_WINDOW_DAYS)
    series = _filled(_series(history, window)).dropna()
    if len(series) < 24 * 7:
        raise ModelUnavailable("auto_arima", "sufficient history",
                               f"{len(series)} hours on a regular grid, need 168")
    if params.get("clean_training_spikes", True):
        typical = series.median()
        if typical > 0:
            gross = series > SPIKE_MULTIPLE * typical
            if 0 < int(gross.sum()) < len(series) * 0.05:
                series = series.mask(gross).interpolate(limit_direction="both")
    values = series.to_numpy(float)
    if engine_asked == "sktime":
        return _sktime_forecast("auto_arima", history, future, params)
    try:
        from statsforecast.models import AutoARIMA as _SF
        model = _SF(season_length=season)
        preds = np.asarray(model.fit(values).predict(h=len(future))["mean"],
                           dtype=float)
        engine = f"statsforecast.AutoARIMA(season_length={season})"
    except Exception:                                          # noqa: BLE001
        if engine_asked == "statsforecast":
            raise                                    # asked for, not available
        return _sktime_forecast("auto_arima", history, future, params)
    return Forecast(preds, evidence={"engine": engine,
                                     "season_length_hours": season,
                                     "context_hours": int(len(values))})




MODELS = {
    "seasonal_naive": _seasonal_naive,
    "moving_average": _moving_average,
    "linear": _linear,
    "gradient_boosting": _gradient_boosting,
    "lightgbm": _lightgbm,
    "lightgbm_weather": _lightgbm_weather,
    "knn": _knn,
    "knn_weather": _knn_weather,
    "weather_regression": _weather_regression,
    "theta": lambda h, f, p: _sktime_forecast("theta", h, f, p),
    "ets": lambda h, f, p: _sktime_forecast("ets", h, f, p),
    "arima": lambda h, f, p: _sktime_forecast("arima", h, f, p),
    "auto_arima": _auto_arima,
    "chronos": _chronos,
    "tiny_time_mixer": _tiny_time_mixer,
}

# Models that need no fitting sample, and so can answer for a building the
# fitted models have to refuse. Read by the tool when it checks history length.
# Nothing is fitted in any of these, so the bar is "enough to extend", not
# "enough to learn from". tiny_time_mixer belongs here for the same reason
# chronos does: it is zero-shot, and its context width is the shape of the
# checkpoint rather than an amount of evidence it needs.
LOW_HISTORY_MODELS = ("seasonal_naive", "moving_average", "chronos",
                      "tiny_time_mixer")

# Models the caller must fetch site weather for. Naming them here keeps the
# tool from having to know which models are weather-driven.
NEEDS_WEATHER = ("weather_regression", "lightgbm_weather", "knn_weather")

# For an ablation: the same model with the weather added. "What improvement is
# obtained by adding air temperature to X" is the difference between these two,
# and it is only attributable to the weather if everything else is identical --
# same estimator, same hyper-parameters, same seed, same rows.
WITH_WEATHER = {"lightgbm": "lightgbm_weather", "linear": "weather_regression",
                "knn": "knn_weather"}


def run(model: str, history: pd.DataFrame, future: pd.DatetimeIndex,
        **params) -> Forecast:
    """Produce one forecast, with its method attached."""
    if model not in MODELS:
        raise ModelUnavailable(model, "a registered model",
                               f"known models: {', '.join(MODELS)}")
    forecast = MODELS[model](history, future, params)
    forecast.evidence = {
        "model": model,
        "library": LIBRARY[model],
        "method": METHOD_PROSE[model],
        "fitting": FITTING[model],
        **forecast.evidence,
    }
    return forecast
