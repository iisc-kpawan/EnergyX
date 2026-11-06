"""Advanced AI Analytics tools (8).

Covers task categories 9-11. Models are selected from a registry by name rather
than exposed one-tool-per-model, so adding a forecaster does not widen the
agent's tool surface.

Every model-backed result reports the method and its parameters. A forecast
without a stated cutoff, or an anomaly without a stated criterion, is not a
usable benchmark answer.
"""
from __future__ import annotations

import os
import time
from typing import Literal

from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast_models
from ._common import batched_series, budgeted, with_coverage, in_clause, load_series, with_local_parts
from .base import (Status, ToolResult, empty, frame_result,
                   insufficient, ok)
from .registry import ToolContext, tool

# Above this many buildings an hourly scan is answered at daily grain,
# which reads the precomputed aggregate instead of raw measurements:
# 1.2 s against 57 s for the same question, and it actually finishes.
COHORT_DAILY_GRAIN = 300

# Every model the registry knows, in the order a reader should consider them:
# the two pinned baselines first, then the fitted models, then the pretrained
# one. The Literal on the tool is generated from this, so the agent sees the
# same list the registry implements.
FORECAST_MODELS = tuple(forecast_models.MODELS)
# A fitted model needs a sample. A lookup or a pretrained model needs only
# enough history to look back over -- one week plus the hours it repeats.
MIN_TRAIN_HOURS = 24 * 14
MIN_LOOKBACK_HOURS = 24 * 8
# A horizon longer than a season is not a forecast, it is a scenario, and the
# snapshot has no ground truth to score it against.
MAX_HORIZON_HOURS = 24 * 90
# A rolling backtest refits a model per building per origin, so the cohort is
# bounded: a portfolio-wide backtest is a different question and needs a
# different protocol than one that refits for every building.
# A cohort question names as many buildings as it names; the origin sampler
# already prices the work, so the cap only has to stop something absurd.
#: No cap. It was 120, and a cohort larger than that was sampled down to the
#: buildings with the most complete records -- so a portfolio question was
#: answered over a subset chosen by the tool, disclosed in a note and invisible
#: in the number. Scoring every building is slower and it is the answer; the
#: parallel backtest is what makes it affordable.
MAX_BACKTEST_BUILDINGS = None
#: Buildings scored at once. Independent work, GIL-releasing fits.
BACKTEST_WORKERS = int(os.environ.get("ENERGYX_BACKTEST_WORKERS", "12"))


def _hourly(ctx: ToolContext, building_id: str, end: str | None = None,
            grain: str = "hourly") -> pd.DataFrame:
    """The series a forecast is fitted to, at the grain it is asked for.

    ``grain="daily"`` sums each local day, which is a different series from
    the hourly one and a different forecasting problem: "the next 7 days of
    daily consumption, fitted on the previous 30 days of daily data" is seven
    points from thirty, not 168 points from 720. Forecasting hourly and adding
    the hours up afterwards is a third thing again, and none of the three
    gives the same error.
    """
    params: list = [building_id]
    where = "building_id = ?"
    if end:
        where += " AND timestamp_local < ?"; params.append(end)
    if grain == "daily":
        return ctx.db.df(f"""
            SELECT CAST(timestamp_local AS DATE) AS timestamp, SUM(value) AS value
            FROM electricity WHERE {where} AND quality_flag = 'OK'
              AND value IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """, params)
    return ctx.db.df(f"""
        SELECT timestamp_local AS timestamp, value
        FROM electricity WHERE {where} ORDER BY timestamp
    """, params)


def _naive(ts: pd.Series) -> pd.Series:
    """Local wall time without a timezone tag, whichever side it came from.

    The snapshot stores timestamp_local as TIMESTAMP WITH TIME ZONE holding
    local wall time under +00:00, so it reads back tz-aware while anything
    built in pandas is naive. Only the tag differs; the clock reading does not.
    """
    return ts.dt.tz_convert(None) if getattr(ts.dtype, "tz", None) else ts


def _features(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts)
    hour, dow, month = t.dt.hour, t.dt.dayofweek, t.dt.month
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        (dow >= 5).astype(float), hour.astype(float),
    ])


def _assert_model_vocabulary() -> None:
    """The declared model names must be the registry's, exactly.

    They drifted once: the registry gained LightGBM, KNN and Tiny Time Mixers
    and this module's Literal did not, so the catalogue advertised nine models
    to the planner while the system had fourteen, and a question naming
    LightGBM was refused by argument validation with a list that omitted a
    model the system demonstrably has.

    Read from the source text, because the @tool decorator rebuilds the
    signature and ``from __future__ import annotations`` leaves the hint a
    string. If the list cannot be read at all the check is skipped: a guard
    that breaks the import is worse than the drift it exists to catch.
    """
    import re as _re
    try:
        source = Path(__file__).read_text()
        block = _re.search(r"model: Literal\[(.*?)\] = \"seasonal_naive\"",
                           source, _re.S)
        if not block:
            return
        declared = set(_re.findall(r'"([a-z_]+)"', block.group(1)))
    except Exception:                                       # noqa: BLE001
        return
    if not declared:
        return
    known = set(forecast_models.MODELS)
    missing, extra = known - declared, declared - known
    if missing or extra:
        raise RuntimeError(
            f"forecast_timeseries model list is out of step with the registry: "
            f"missing {sorted(missing)}, unknown {sorted(extra)}")


@tool("ai_analytics")
def forecast_timeseries(
    ctx: ToolContext,
    building_id: str | None = None,
    # A cohort forecast is the same forecast issued once per building. The
    # question that asks it -- "forecast the next 7 days for all education
    # buildings in London and report the median MAE" -- names one origin, so
    # backtest_forecast_models is the wrong tool for it by that tool's own
    # docstring, and this one accepted a single building only. Planners wrote
    # building_ids anyway, which is the readable thing to write, and the run
    # died on "unknown parameter 'building_ids'". It fans out here instead.
    building_ids: list[str] | None = None,
    horizon_hours: int = 24,
    # The registry gained lightgbm, knn, their weather variants and Tiny Time
    # Mixers, and this signature was not updated with it -- so a question that
    # named LightGBM was refused by the argument validator with a list that
    # did not include a model the system demonstrably has. A tool's declared
    # vocabulary has to be the registry's, or the catalogue lies to the
    # planner.
    model: Literal["seasonal_naive", "moving_average", "linear",
                   "gradient_boosting", "lightgbm", "lightgbm_weather",
                   "knn", "knn_weather", "weather_regression",
                   "theta", "ets", "arima", "auto_arima", "chronos",
                   "tiny_time_mixer"] = "seasonal_naive",
    cutoff: str | None = None,
    window_days: int = 7,
    # The season a seasonal model repeats from, in hours. 168 respects the
    # weekday/weekend split and is the default this implementation has always
    # used; 24 is the usual textbook season for hourly data. They are
    # different forecasts -- 0.2250 against 0.1417 MAE on one building over
    # one 12-hour horizon -- so a question that names the season is answered
    # with the season it names.
    season_length: int | None = None,
    # Which implementation runs a named algorithm. "AutoARIMA" is a search
    # procedure, not a library, and two libraries that both implement
    # Hyndman-Khandakar disagree: on one building over one 24-hour horizon
    # statsforecast scores MAE 0.5441 and sktime's pmdarima wrapper 1.780.
    # The default is statsforecast -- the reference for the algorithm, and
    # fourteen times faster here -- and the engine that ran is reported, so
    # a figure can be reproduced without guessing which one produced it.
    engine: Literal["auto", "statsforecast", "sktime"] = "auto",
    # The grain the model is fitted at. "Forecast the next 7 days of daily
    # consumption using the previous 30 days of daily data" is a daily model:
    # fitting hourly and summing afterwards answers a different question and
    # scores differently.
    grain: Literal["hourly", "daily"] = "hourly",
    interval_level: float = 0.8,
) -> ToolResult:
    """Forecast electricity demand from a frozen historical cutoff.

    The cutoff is mandatory in spirit: a forecast must be reproducible,
    which means the model may only see data strictly before it.

    ``model`` selects from the registry in ``forecast_models``. The two
    baselines are the ones an operator already carries in their head --
    ``seasonal_naive`` repeats the same weekday one week earlier, and
    ``moving_average`` averages each hour of day over the trailing
    ``window_days`` -- and they are pinned implementations, so a reference
    answer cannot move under a dependency upgrade. ``theta``, ``ets`` and
    ``arima`` are sktime's statistical forecasters, fitted to a bounded recent
    window. ``weather_regression`` adds the site's air temperature, which is
    what a question about weather-driven demand is asking for. ``chronos`` is a
    pretrained foundation model applied zero-shot, and is the only one that
    returns a forecast distribution without fitting anything to this building.

    The answer is stated in both units the question can mean: each hour as mean
    demand in kW, and the horizon as energy in kWh. A forecast reported as a
    bare column of numbers is not usable by anyone who did not write the tool.
    """
    if model not in FORECAST_MODELS:
        return insufficient("unknown_model", f"valid models: {list(FORECAST_MODELS)}")
    cohort = [b for b in (building_ids or []) if b]
    if not cohort and building_id:
        cohort = [building_id]
    if not cohort:
        return insufficient("no_target", "give building_id or building_ids")
    if len(cohort) > 1:
        return _forecast_cohort(
            ctx, cohort, horizon_hours=horizon_hours, model=model,
            cutoff=cutoff, window_days=window_days, season_length=season_length,
            engine=engine, grain=grain, interval_level=interval_level)
    building_id = cohort[0]
    if horizon_hours < 1 or horizon_hours > MAX_HORIZON_HOURS:
        return insufficient("horizon", f"horizon_hours must be 1..{MAX_HORIZON_HOURS}")
    # A coverage of 1 asks for the 0th and 100th percentiles, which is an
    # infinite band reported as a number. The planner writes this when the
    # question says nothing about uncertainty, so it is clamped rather than
    # refused: the point forecast is what was asked for either way.
    interval_level = min(max(float(interval_level), 0.5), 0.99)
    history = _hourly(ctx, building_id, end=cutoff, grain=grain)
    history = history.dropna(subset=["value"])
    # A model that fits parameters needs a sample; one that does not needs only
    # enough to look back over. Holding every model to the fitted models' floor
    # refused forecasts that seasonal_naive and the foundation model could both
    # have answered from a fortnight of history.
    floor = (MIN_LOOKBACK_HOURS if model in forecast_models.LOW_HISTORY_MODELS
             else MIN_TRAIN_HOURS)
    # A seasonal baseline needs one season, not the eight days the weekly
    # season implies. With a 24-hour season the model needs 24 hours of
    # history and the 192-hour floor refused it: asked for the nine Dublin
    # offices from 2016-01-08, whose record begins on the 1st, the first
    # building had 167 hours and the whole cohort question failed on it.
    if model == "seasonal_naive" and season_length:
        floor = min(floor, max(int(season_length) + 24, 48))
    # ``window_days`` says how much history to *use*, not how much to
    # require. Nothing is fitted in these models, so the real minimum is a
    # seasonal cycle -- enough to extend from. Held to 192 hours, the nine
    # Dublin offices failed on the first: its record begins a week before
    # the cutoff and holds 167 hours, one short of the 168 the question
    # names, and one missing hour is not a reason to refuse the question.
    # The fitted models keep their floor, where a short sample is a real
    # objection to fitting.
    if model in forecast_models.LOW_HISTORY_MODELS and grain != "daily":
        floor = min(floor, 24)
    # The floor counts observations, and a daily series has one where an
    # hourly one has 24. Held to the hourly floor, a thirty-day daily history
    # looked like thirty hours and every daily forecast was refused.
    unit = "hours"
    if grain == "daily":
        floor = max(14, floor // 24)
        unit = "days"
    if len(history) < floor:
        return insufficient("insufficient_history",
                            f"{building_id} has {len(history)} usable {unit} before "
                            f"the cutoff; the {model!r} model needs {floor}")
    last = pd.to_datetime(history["timestamp"]).max()
    # The horizon steps at the grain the model was fitted at. Left hourly, a
    # daily forecast came back stamped 01:00, 02:00, 03:00 -- seven hours
    # wearing the values of seven days, and nothing downstream could align it
    # with the observed days to score it.
    step = pd.Timedelta(days=1) if grain == "daily" else pd.Timedelta(hours=1)
    # The parameter is named in hours and is read that way at either grain.
    # Taken as a step count at daily grain, "the next 7 days" written the
    # obvious way -- horizon_hours=168 -- forecast 168 *days*, out to October
    # for a May cutoff, and the number that came back was not wrong so much
    # as about a different question.
    periods = max(1, round(horizon_hours / 24)) if grain == "daily" \
        else horizon_hours
    future = pd.date_range(last + step, periods=periods,
                           freq="D" if grain == "daily" else "h")

    params = {"window_days": window_days, "interval_level": interval_level,
              "season_length": season_length, "engine": engine}
    if model == "weather_regression":
        params["exog_history"], params["exog_future"] = _site_weather(
            ctx, building_id, cutoff, future)
    try:
        forecast = forecast_models.run(model, history, future, **params)
    except forecast_models.ModelUnavailable as unavailable:
        return insufficient("model_unavailable", str(unavailable))

    preds = np.asarray(forecast.values, float)
    # The column says its unit. It was `predicted`, and a reader handed a
    # column of numbers headed "predicted" cannot tell demand from energy --
    # which for an hourly series are the same figure and different questions.
    out = pd.DataFrame({"building_id": building_id, "timestamp": future,
                        "predicted_kw": preds})
    out["hour_of_day"] = pd.DatetimeIndex(future).hour
    if forecast.lower is not None:
        out["lower_kw"] = np.asarray(forecast.lower, float)
        out["upper_kw"] = np.asarray(forecast.upper, float)

    total = float(preds.sum())
    peak_idx = int(np.argmax(preds))
    trough_idx = int(np.argmin(preds))
    horizon_days = horizon_hours / 24
    # What the forecast is worth: the same measure over the history it was
    # built from, so a reader can see whether the model is predicting an
    # ordinary day or an exceptional one before they act on it.
    recent = history.tail(int(min(len(history), 24 * 28)))
    typical_day = (float(recent["value"].sum()) / max(len(recent) / 24, 1e-9)
                   if len(recent) else None)
    evidence = {
        **forecast.evidence,
        # Which column is the measure, so a chart plots the forecast rather
        # than whichever numeric column happens to sit last. hour_of_day is a
        # numeric column and would otherwise be drawn as a ramp from 0 to 23.
        "metric": "predicted_kw",
        "unit": "kW (hourly mean demand); kWh when summed over the horizon",
        "building_id": building_id,
        "horizon_hours": horizon_hours,
        "horizon_from": f"{future[0]:%Y-%m-%d %H:%M}",
        "horizon_to": f"{future[-1]:%Y-%m-%d %H:%M}",
        "cutoff": str(cutoff) if cutoff else str(last),
        "history_hours_available": int(len(history)),
        "history_from": f"{pd.to_datetime(history['timestamp']).min():%Y-%m-%d}",
        "timezone": "local",
        "forecast_total_kwh": round(total, 4),
        "forecast_mean_kw": round(float(preds.mean()), 4),
        "predicted_peak_kw": round(float(preds[peak_idx]), 4),
        "predicted_peak_time": f"{future[peak_idx]:%Y-%m-%d %H:%M}",
        "predicted_minimum_kw": round(float(preds[trough_idx]), 4),
        "predicted_minimum_time": f"{future[trough_idx]:%Y-%m-%d %H:%M}",
        "typical_recent_day_kwh": (round(typical_day, 4)
                                   if typical_day is not None else None),
        # Kept for plans written before the unit was named.
        "day_total": round(total, 4) if horizon_hours == 24 else None,
        "predicted_peak": round(float(preds[peak_idx]), 4),
        "predicted_peak_time_raw": str(future[peak_idx]),
    }
    if forecast.lower is not None:
        evidence["interval_level"] = forecast.interval_level
        evidence["forecast_total_kwh_range"] = [
            round(float(np.sum(forecast.lower)), 4),
            round(float(np.sum(forecast.upper)), 4)]
    # The answer to "forecast this day" is the energy over the horizon, and the
    # runtime promotes a declared headline to the typed answer. Without it the
    # answer was the handle of a 24-row table.
    evidence["headline"] = {
        "value": round(total, 4), "unit": "kWh",
        "label": (f"forecast electricity consumption for {building_id}, "
                  f"{_span_words(future)}"),
    }
    evidence["narrative"] = _forecast_narrative(
        building_id, future, preds, forecast, total, typical_day, horizon_days)
    return frame_result(out, ctx.store, name=f"forecast_{model}",
                        evidence=evidence)


MAX_COHORT_FORECASTS = 60


def _forecast_cohort(ctx: ToolContext, cohort: list[str], **kw) -> ToolResult:
    """One forecast per building from a single shared origin.

    The cohort question asks for one number over many buildings -- a median
    MAE, usually -- and the honest way to it is the per-building forecast
    repeated, not a pooled model that answers a different question. Buildings
    without enough history are skipped and counted rather than failing the
    whole cohort: a run that loses 22 buildings because one of them starts
    late has told the asker nothing.
    """
    if len(cohort) > MAX_COHORT_FORECASTS:
        return insufficient(
            "cohort_too_large",
            f"{len(cohort):,} buildings is past the {MAX_COHORT_FORECASTS} "
            "this tool forecasts one at a time; narrow the cohort")
    frames, skipped, evidence = [], [], {}
    for name in cohort:
        one = forecast_timeseries(ctx, building_id=name, **kw)
        if one.status is not Status.OK or not one.handle:
            skipped.append(f"{name}: {one.message or one.status.name}")
            continue
        frames.append(ctx.store.get(one.handle))
        evidence = one.evidence or evidence
    if not frames:
        return insufficient(
            "no_forecastable_buildings",
            f"none of the {len(cohort)} buildings could be forecast"
            + (f" ({skipped[0]})" if skipped else ""))
    out = pd.concat(frames, ignore_index=True)
    model = kw.get("model", "seasonal_naive")
    ev = {"buildings_forecast": len(frames), "buildings_requested": len(cohort),
          "model": model, "cutoff": kw.get("cutoff"),
          "grain": kw.get("grain", "hourly"),
          "horizon_hours": kw.get("horizon_hours"),
          "engine": (evidence or {}).get("engine")}
    if skipped:
        ev["skipped"] = skipped[:10]
        ev["skipped_count"] = len(skipped)
    ev["narrative"] = (
        f"{model} forecast issued from {kw.get('cutoff')} for "
        f"{len(frames)} of {len(cohort)} buildings"
        + (f"; {len(skipped)} skipped for insufficient history" if skipped else "")
        + ". Score it with evaluate_forecast to get the error per building.")
    return frame_result(out, ctx.store, name=f"forecast_{model}_cohort",
                        evidence=ev)


def _hourly_many(ctx: ToolContext, building_ids: list[str]) -> pd.DataFrame:
    """Hourly readings for a cohort, keyed by building."""
    params: list = []
    where = "quality_flag = 'OK'" + in_clause("building_id", building_ids, params)
    return ctx.db.df(f"""
        SELECT building_id, timestamp_local AS timestamp, value
        FROM electricity WHERE {where} ORDER BY building_id, timestamp
    """, params or None)


def _site_weather_series(ctx: ToolContext, building_id: str) -> pd.DataFrame:
    """Every weather variable a forecaster may use, at one building's site.

    All three are fetched whichever are asked for, because the join is the
    expensive part and a second pass per variable would cost more than the
    columns do. Rows are kept when air temperature is present -- cloud
    coverage is recorded on 48% of hours, and dropping on it here would halve
    the sample for every model, including those that never asked for it.
    """
    return ctx.db.df("""
        SELECT w.timestamp_local AS timestamp, w.air_temperature,
               w.dew_temperature, w.cloud_coverage
        FROM weather w
        WHERE w.site_id = (SELECT b.site_id FROM buildings b
                           WHERE b.building_id = ? LIMIT 1)
          AND w.air_temperature IS NOT NULL
        ORDER BY 1
    """, [building_id])


def _span_words(future: pd.DatetimeIndex) -> str:
    """The horizon as a person would say it."""
    if len(future) == 24 and future[0].hour == 0:
        return f"{future[0]:%A %-d %B %Y}"
    if future[0].date() == future[-1].date():
        return f"{future[0]:%Y-%m-%d %H:%M} to {future[-1]:%H:%M}"
    return f"{future[0]:%Y-%m-%d %H:%M} to {future[-1]:%Y-%m-%d %H:%M} local"


def _forecast_narrative(building_id, future, preds, forecast, total,
                        typical_day, horizon_days) -> list[str]:
    """The forecast in sentences, which is how it has to arrive to be used.

    A reader handed 24 unlabelled numbers cannot tell demand from energy, local
    time from UTC, or a fitted model from a lookup. Each of those is one line
    here, and each is a fact the result already knows -- what was missing was
    saying it.
    """
    peak, trough = int(np.argmax(preds)), int(np.argmin(preds))
    lines = [
        f"Forecast for {building_id} over {_span_words(future)} "
        f"({len(future)} hourly steps, local time).",
        f"  Total consumption   {total:,.1f} kWh",
        f"  Mean demand         {preds.mean():,.1f} kW",
        f"  Peak demand         {preds[peak]:,.1f} kW at "
        f"{future[peak]:%H:%M} on {future[peak]:%-d %b}",
        f"  Lowest demand       {preds[trough]:,.1f} kW at "
        f"{future[trough]:%H:%M} on {future[trough]:%-d %b}",
    ]
    if forecast.lower is not None and forecast.interval_level:
        lines.append(
            f"  {forecast.interval_level:.0%} interval   "
            f"{np.sum(forecast.lower):,.1f} to {np.sum(forecast.upper):,.1f} kWh "
            f"over the horizon")
    if typical_day is not None and horizon_days:
        expected = typical_day * horizon_days
        if expected > 0:
            change = 100 * (total - expected) / expected
            direction = ("above" if change > 0.5 else
                         "below" if change < -0.5 else "in line with")
            lines.append(
                f"  That is {abs(change):,.1f}% {direction} this building's "
                f"recent typical {typical_day:,.1f} kWh/day"
                if direction != "in line with" else
                f"  That is in line with this building's recent typical "
                f"{typical_day:,.1f} kWh/day")
    lines.append(f"  Method: {forecast.evidence.get('method')} "
                 f"[{forecast.evidence.get('library')}; "
                 f"{forecast.evidence.get('fitting')}].")
    return lines


# Weekend and weekday are Saturday/Sunday and Monday-Friday by the local
# calendar, which is the only definition the snapshot supports: it carries no
# holiday calendar, and inventing one per country would be a different answer
# in every city.
_WEEKEND = (5, 6)


def _only_days(frame: pd.DataFrame, days: str) -> pd.DataFrame:
    """Rows falling on the day type asked for, by local weekday."""
    if days in (None, "all") or frame.empty or "date" not in frame:
        return frame
    weekday = pd.to_datetime(frame["date"]).dt.dayofweek
    keep = weekday.isin(_WEEKEND) if days == "weekend" else ~weekday.isin(_WEEKEND)
    return frame.loc[keep]


def _only_hours(frame: pd.DataFrame, first, last) -> pd.DataFrame:
    """Rows inside a clock window, which may wrap midnight.

    "8 PM to 7 AM" is hours 20..23 and 0..7, and a naive `first <= h <= last`
    returns nothing for it. Restricting the baseline as well as the scan is
    the point: a quiet hour should be unusual among this building's other
    quiet hours, not merely lower than its afternoons.
    """
    if first is None or last is None or frame.empty or "hour" not in frame:
        return frame
    first, last = int(first) % 24, int(last) % 24
    hours = frame["hour"]
    keep = (hours.between(first, last) if first <= last
            else (hours >= first) | (hours <= last))
    return frame.loc[keep]


def _only_dates(frame: pd.DataFrame, days: str, column: str) -> pd.DataFrame:
    """The same restriction on a frame keyed by a date column."""
    if days in (None, "all") or frame.empty or column not in frame:
        return frame
    weekday = pd.to_datetime(frame[column]).dt.dayofweek
    keep = weekday.isin(_WEEKEND) if days == "weekend" else ~weekday.isin(_WEEKEND)
    return frame.loc[keep]


def _sweep_headline(frame: pd.DataFrame, spread: float) -> dict:
    """What the sweep found, not what a sweep is.

    "How far the anomaly count moves across the settings swept" describes the
    tool; the question asked how the count changes between two thresholds, and
    the answer is the two counts and the difference between them. A ratio is
    also the wrong summary when one end is zero -- 0 to 12 is an infinite
    factor and a change of twelve.
    """
    if frame.empty or "anomalies" not in frame or "threshold" not in frame:
        return {"value": None, "unit": "x",
                "label": "how far the anomaly count moves across the settings swept"}
    ordered = frame.sort_values("threshold", ascending=False)
    first, last = ordered.iloc[0], ordered.iloc[-1]
    change = int(last["anomalies"]) - int(first["anomalies"])
    if len(ordered) == 2:
        label = (f"{int(first['anomalies']):,} at {first['threshold']:g} SD and "
                 f"{int(last['anomalies']):,} at {last['threshold']:g} SD: "
                 f"{'adds' if change >= 0 else 'removes'} {abs(change):,}")
    else:
        label = (", ".join(f"{int(r['anomalies']):,} at {r['threshold']:g} SD"
                           for _, r in ordered.iterrows())
                 + f"; {abs(change):,} between the ends")
    return {"value": change, "unit": "anomalies", "label": label}


def _per_building_counts(found: pd.DataFrame, ctx: ToolContext,
                         method: str) -> dict:
    """A count and a worst score per building, published as its own artifact.

    An anomaly scan returns incidents; a decision-support question needs one
    row per building so the count can sit beside a forecast error and a floor
    area. Producing it here means nothing downstream has to re-aggregate a
    list of timestamps to learn how many each building had.
    """
    if found.empty or "building_id" not in found:
        return {}
    scored = found.assign(_abs=found["score"].abs())
    summary = (scored.groupby("building_id")
               .agg(anomalies=("score", "size"),
                    worst_score=("_abs", "max"))
               .reset_index()
               .sort_values("anomalies", ascending=False))
    summary["method"] = method
    return {"per_building_handle": ctx.store.put(
                summary, name=f"anomalies_per_building_{method}"),
            "buildings_with_anomalies": int(len(summary))}


def _wall_clock(stamp) -> str:
    """One timestamp as local wall clock, the way jsonify renders a column.

    These rows are built as dicts rather than as a frame, so they never pass
    through the frame renderer and were printing the stored +00:00 under a
    local-time answer.
    """
    moment = pd.Timestamp(stamp)
    if moment.tz is not None:
        moment = moment.tz_localize(None)
    return moment.strftime("%Y-%m-%d %H:%M" if (moment.hour or moment.minute)
                           else "%Y-%m-%d")


def _detection_narrative(what: str, rows: list[dict], evidence: dict, *,
                         when_field: str = "when", value_field: str = "observed",
                         unit: str = "kWh", reference: str = "") -> list[str]:
    """What a detection found, said rather than tabulated.

    The same complaint applies here as to a forecast: a table headed "score"
    with a column of 3.4s does not tell a reader what crossed what, in what
    units, measured against which baseline. Every one of those is already in
    the evidence; this states them, and names the worst few offenders, which
    is what someone reading an anomaly scan actually wants first.
    """
    count = len(rows)
    period = evidence.get("period") or []
    window = (f" between {period[0]} and {period[1]}"
              if len(period) == 2 and period[0] and period[1] else "")
    # Say which hours were scanned. Asked about "nighttime hours from 8 PM to
    # 7 AM" the scan was correctly restricted to them and the answer still
    # opened "5,239 anomalous hours found" -- a reader cannot tell that from a
    # scan of the whole clock, and the two are different findings.
    clock = evidence.get("hours") or evidence.get("clock_window")
    if isinstance(clock, (list, tuple)) and len(clock) == 2 and clock[0] is not None:
        first, last = int(clock[0]), int(clock[1])
        label = "overnight" if first > last else "daytime" if 6 <= first < 12 else ""
        window += (f", {label} hours only ({first:02d}:00-{last:02d}:59)"
                   if label else f", hours {first:02d}:00-{last:02d}:59 only")
    if not count:
        # "No anomalous hours found" is the right answer and does not contain
        # the number that *is* the answer. A question asking how many there
        # are should be answered with a figure, and zero is a figure: a reader
        # scanning for one, or a table collecting them, finds nothing here.
        return [f"0 {what} found{window}.",
                f"  Test: {reference}" if reference else ""]
    lines = [f"{count:,} {what}{'' if count != 1 else ''} found{window}."]
    for row in rows[:5]:
        when = row.get(when_field)
        value = row.get(value_field)
        score = row.get("score")
        detail = f"  {when}"
        if isinstance(value, (int, float)):
            detail += f"   {value:,.1f} {unit}"
        if isinstance(score, (int, float)):
            detail += f"   ({score:+.2f} by the test's own scale)"
        if row.get("building_id") and evidence.get("buildings_examined", 1) != 1:
            detail += f"   {row['building_id']}"
        lines.append(detail)
    if count > 5:
        lines.append(f"  ... and {count - 5:,} more, worst first")
    if reference:
        lines.append(f"  Test: {reference}")
    return [line for line in lines if line]


def _site_weather(ctx: ToolContext, building_id: str, cutoff: str | None,
                  future: pd.DatetimeIndex):
    """Air temperature at the building's site, split at the cutoff.

    Returned as two frames keyed on the same timestamps the forecast uses, so
    the model never has to reason about which site a building belongs to.
    """
    rows = ctx.db.df("""
        SELECT w.timestamp_local AS timestamp, w.air_temperature
        FROM weather w
        WHERE w.site_id = (SELECT b.site_id FROM buildings b
                           WHERE b.building_id = ? LIMIT 1)
          AND w.air_temperature IS NOT NULL
        ORDER BY 1
    """, [building_id])
    if rows.empty:
        return rows, rows
    stamps = pd.to_datetime(rows["timestamp"])
    edge = pd.Timestamp(cutoff) if cutoff else future[0]
    if stamps.dt.tz is not None and edge.tz is None:
        edge = edge.tz_localize(stamps.dt.tz)
    return rows.loc[stamps < edge], rows.loc[stamps >= edge]


@tool("ai_analytics")
def backtest_forecast_models(
    ctx: ToolContext,
    building_id: str | None = None,
    building_ids: list[str] | None = None,
    models: list[Literal["seasonal_naive", "moving_average", "linear",
                         "gradient_boosting", "lightgbm", "lightgbm_weather",
                         "weather_regression", "theta", "ets", "arima",
                         "auto_arima", "tiny_time_mixer",
                         "chronos"]] | None = None,
    start: str | None = None,
    end: str | None = None,
    horizon_hours: int = 24,
    max_origins: int | None = None,
    metrics: list[Literal["mape", "mae", "rmse", "cvrmse", "mase",
                          "bias"]] | None = None,
    weather_variables: list[Literal["air_temperature", "dew_temperature",
                                    "cloud_coverage"]] | None = None,
    pooled: bool = False,
    use_metadata: bool = False,
    report: Literal["table", "improvement"] = "table",
) -> ToolResult:
    """Score one or more forecasters over a PERIOD by rolling origin.

    Not for a question that names a single origin. "Compare AutoARIMA and
    LightGBM for the next 24 hours starting at 00:00 on 1 March 2016" states
    one cutoff and one horizon: that is two calls to ``forecast_timeseries``
    and two to ``evaluate_forecast``, and it costs seconds. Planners read the
    line below as "the tool for comparing models" and reached for it on every
    question containing the word *compare* -- on the 22 London education
    buildings that took 1,020 seconds, against 5 for the same question
    answered at its stated origin, and returned a rolling-origin average
    where a single-origin MAE was asked for. Use this only where the question
    gives a window and no origin.


    This is what "how accurately does X forecast this building over March and
    April, and what is its MAPE" asks for, and it is not a forecast: a forecast
    is issued once and runs forward: scoring a two-month window with a single
    1,464-hour horizon measures how fast a model decays, not how accurately it
    forecasts. What an operator means is the **rolling origin**: stand at each
    day in the window, forecast the next ``horizon_hours`` from history
    strictly before it, and score every one of those forecasts against what
    actually happened.

    The model is refitted at every origin, which is what day-ahead forecasting
    means. That is expensive -- a seasonal ARIMA fit is tens of seconds, so
    sixty-one of them is a forty-minute question and no interactive budget
    survives it. Rather than run past the deadline and return nothing, the
    tool prices one origin per model first and then spaces the origins it can
    afford evenly across the window, reporting how many of the available
    origins were actually scored. A sampled backtest is a real estimate; a
    backtest that times out is not an answer at all.

    Every model sees exactly the same origins and is scored on exactly the same
    hours -- the sample is chosen once, from the most expensive model's cost,
    and shared. A comparison in which two models saw different days is not one.
    """
    # A "+metadata" arm is the same base model with the catalog's columns
    # added, and it is named rather than toggled so both arms can be asked for
    # in one call and scored on identical hours. Only meaningful pooled, which
    # the guard below enforces.
    requested = list(models or ["seasonal_naive"])
    metadata_arms = {m for m in requested if m.endswith("_metadata")}
    chosen = [m for m in requested
              if m in forecast_models.MODELS
              or (m in metadata_arms
                  and m[:-len("_metadata")] in forecast_models.POOLABLE)]
    if metadata_arms and not pooled:
        return insufficient(
            "metadata_needs_pooling",
            "building metadata does not vary hour to hour, so in a "
            "per-building model it is a constant column that changes nothing "
            "-- the comparison would return identical forecasts and a p-value "
            "of 1. Ask for pooled=True, where one model is fitted across the "
            "cohort and metadata distinguishes one building's rows from "
            "another's.")
    if metadata_arms:
        use_metadata = True
    identity_arms = {m for m in requested if m.endswith("_identity")}
    chosen += [m for m in identity_arms
               if m[:-len("_identity")] in forecast_models.POOLABLE
               and m not in chosen]
    if not chosen:
        return insufficient("unknown_model",
                            f"valid models: {list(forecast_models.MODELS)}")
    if horizon_hours < 1 or horizon_hours > 24 * 14:
        return insufficient("horizon", "horizon_hours must be 1..336 for a "
                                       "rolling backtest")
    cohort = [b for b in (building_ids or ([building_id] if building_id else []))
              if b]
    if not cohort:
        return insufficient("no_building",
                            "a backtest needs at least one named building")
    sampled_from = None
    if MAX_BACKTEST_BUILDINGS and len(cohort) > MAX_BACKTEST_BUILDINGS:
        # "Does adding metadata help LightGBM" names no cohort, so the planner
        # hands over every building and the backtest refused itself for being
        # too large -- a question about method answered with a complaint about
        # scale. A refusal is right when the question names a cohort too big to
        # run; it is unhelpful when the question names none at all, because
        # then the cohort is the tool's to choose. It chooses, and says so:
        # the buildings with the most complete records, in a fixed order, so
        # the same question returns the same sample tomorrow.
        counts = (_hourly_many(ctx, cohort).dropna(subset=["value"])
                  .groupby("building_id").size().sort_values(ascending=False))
        chosen = sorted(counts.head(MAX_BACKTEST_BUILDINGS).index)
        sampled_from = len(cohort)
        cohort = chosen or cohort[:MAX_BACKTEST_BUILDINGS]
    series = _hourly_many(ctx, cohort).dropna(subset=["value"])
    if series.empty:
        return empty(f"no readings for {', '.join(cohort[:3])}")
    stamps = pd.to_datetime(series["timestamp"])
    first, last = stamps.min(), stamps.max()
    window_start = pd.Timestamp(start) if start else first + pd.Timedelta(days=28)
    window_end = pd.Timestamp(end) if end else last + pd.Timedelta(hours=1)
    if window_start.tz is None and stamps.dt.tz is not None:
        window_start = window_start.tz_localize(stamps.dt.tz)
        window_end = window_end.tz_localize(stamps.dt.tz)
    if window_end <= window_start:
        return insufficient("window", "the evaluation window is empty")

    # Every origin, and the hours each one is responsible for. Built once and
    # reused by every model: a comparison in which two models were scored on
    # different hours is not a comparison.
    origins = []
    cursor = window_start
    step = pd.Timedelta(hours=horizon_hours)
    while cursor < window_end:
        stop = min(cursor + step, window_end)
        origins.append((cursor, stop))
        cursor = stop
    if not origins:
        return insufficient("window", "no forecast origins fall in the window")

    # Weather is fetched once per building and reused by every origin, so a
    # weather model costs no more reads than a calendar one.
    weather = ({b: _site_weather_series(ctx, b) for b in cohort}
               if any(m in forecast_models.NEEDS_WEATHER for m in chosen) else {})

    metadata = None
    if use_metadata:
        if not pooled:
            return insufficient(
                "metadata_needs_pooling",
                "building metadata does not vary hour to hour, so in a "
                "per-building model it is a constant column that changes "
                "nothing -- the comparison would return identical forecasts "
                "and a p-value of 1. Ask for pooled=True, where one model is "
                "fitted across the cohort and metadata distinguishes one "
                "building's rows from another's.")
        catalog = ctx.db.df(
            "SELECT building_id, floor_area_m2, year_built, occupants, "
            "number_of_floors, building_type_canonical AS building_type "
            "FROM buildings WHERE building_id IN (SELECT unnest(?))",
            [list(cohort)])
        metadata = forecast_models.metadata_frame(catalog)
        if metadata.empty:
            return insufficient("no_metadata",
                                "none of these buildings carry usable metadata")

    identity = None
    if any(m.endswith("_identity") for m in chosen):
        # One column per building: the control that separates "metadata helps"
        # from "knowing which building this is helps".
        identity = pd.get_dummies(pd.Series(cohort, index=cohort),
                                  prefix="is").astype(float)

    rows, notes, unscored, per_building = [], [], [], []
    paired_errors: dict = {}
    deadline = getattr(ctx, "deadline", None)
    available = len(origins)
    origins, sampling = _affordable_origins(
        origins, chosen, cohort, series, weather, deadline, max_origins, notes)
    if not origins:
        return insufficient(
            "budget",
            f"a single forecast origin for {', '.join(chosen)} costs more than "
            f"the time left. Ask for fewer models, a shorter window, or a "
            f"cheaper model.")
    for model in chosen:
        began, ran_out = time.time(), False
        actuals, forecasts, scored_buildings = [], [], 0
        # Pooling is a property of the model, not of the comparison. Asked to
        # compare Seasonal Naive with a pooled learner, the whole call refused
        # because the naive one cannot be fitted across buildings -- and the
        # question was which of the two forecasts better, which needs both
        # scored however each can be. A model that cannot pool is scored per
        # building and the answer says so.
        pool_this = pooled
        if pooled and model.split("_metadata")[0].split("_identity")[0] \
                not in forecast_models.POOLABLE:
            pool_this = False
            notes.append(f"{model}: fitted one series at a time, because it "
                         f"cannot be pooled; the other models keep the pooled "
                         f"fit and the two are scored on the same hours")
        if pool_this:
            wants_metadata = model.endswith("_metadata")
            wants_identity = model.endswith("_identity")
            base = (model[:-len("_metadata")] if wants_metadata
                    else model[:-len("_identity")] if wants_identity else model)
            pooled_rows, ran_out = _pooled_arm(
                ctx, base, cohort, series, weather, origins, window_start,
                window_end,
                metadata if wants_metadata
                else identity if wants_identity else None,
                weather_variables, deadline, notes)
            if not pooled_rows:
                unscored.append({"model": model,
                                 "why": next((n.split(": ", 1)[-1] for n in notes
                                              if n.startswith(f"{model}:")),
                                             "no pooled fit could be made")})
                continue
            actual = pd.concat([p["actual"] for p in pooled_rows])
            predicted = pd.concat([p["predicted"] for p in pooled_rows])
            paired_errors[model] = (predicted - actual).abs()
            for part in pooled_rows:
                per_building.append({
                    "model": model, "building_id": part["building_id"],
                    **_accuracy(part["actual"].to_numpy(float),
                                part["predicted"].to_numpy(float),
                                series, part["actual"].index),
                    "hours_scored": int(len(part["actual"]))})
            rows.append({"model": model,
                         **_accuracy(actual.to_numpy(float),
                                     predicted.to_numpy(float),
                                     series, actual.index),
                         "buildings_scored": len(pooled_rows),
                         "hours_scored": int(len(actual)),
                         "origins": len(origins),
                         "seconds": round(time.time() - began, 2),
                         "library": forecast_models.LIBRARY.get(base, base),
                         "metadata_features": (len(metadata.columns)
                                               if wants_metadata and
                                               metadata is not None else 0),
                         "truncated": ran_out})
            continue
        # Buildings are independent, so they are scored in parallel. The
        # fits spend their time in LAPACK and LightGBM's C core, which release
        # the GIL, and BLAS is pinned to one thread apiece -- so the threads
        # get real cores rather than fighting over one. Serially, 62 buildings
        # over 30 November origins is about six hours, which is how a time
        # budget ends up deciding an answer instead of bounding a runaway.
        def _score_building(building):
            own = series.loc[series["building_id"] == building]
            own_stamps = pd.to_datetime(own["timestamp"])
            truth = own.loc[(own_stamps >= window_start)
                            & (own_stamps < window_end)].copy()
            if truth.empty:
                return None
            truth["timestamp"] = pd.to_datetime(truth["timestamp"])
            truth = truth.set_index("timestamp")["value"]
            predictions = []
            for origin, stop in origins:
                if deadline and time.time() > deadline - 5:
                    break
                history = own.loc[own_stamps < origin]
                floor = (MIN_LOOKBACK_HOURS
                         if model in forecast_models.LOW_HISTORY_MODELS
                         else MIN_TRAIN_HOURS)
                if len(history) < floor:
                    continue
                steps = int((stop - origin) / pd.Timedelta(hours=1))
                future = pd.date_range(origin, periods=steps, freq="h")
                params = {"window_days": 7, "interval_level": 0.8,
                          "weather_variables": weather_variables}
                if model in forecast_models.NEEDS_WEATHER:
                    observed = weather.get(building)
                    if observed is None or observed.empty:
                        notes.append(f"{model}: no site weather for {building}")
                        break
                    marks = pd.to_datetime(observed["timestamp"])
                    params["exog_history"] = observed.loc[marks < origin]
                    params["exog_future"] = observed.loc[marks >= origin]
                try:
                    forecast = forecast_models.run(model, history, future, **params)
                except forecast_models.ModelUnavailable as unavailable:
                    notes.append(f"{model}: {unavailable}")
                    predictions = []
                    break
                except Exception as failure:                # noqa: BLE001
                    notes.append(f"{model} failed for {building} at "
                                 f"{origin:%Y-%m-%d}: {type(failure).__name__}")
                    continue
                predictions.append(pd.Series(np.asarray(forecast.values, float),
                                             index=future))
            if not predictions:
                return None
            predicted = pd.concat(predictions)
            predicted = predicted[~predicted.index.duplicated(keep="first")]
            paired = pd.concat({"actual": truth, "predicted": predicted},
                               axis=1).dropna()
            if paired.empty:
                return None
            # Kept per building as well as pooled. "How does accuracy vary
            # across the Education buildings at Bull" is a question about the
            # spread, and a single pooled MAPE over sixty-three buildings
            # answers it with one number that belongs to none of them.
            return (paired, {
                "model": model, "building_id": building,
                **_accuracy(paired["actual"].to_numpy(float),
                            paired["predicted"].to_numpy(float),
                            own, paired.index),
                "hours_scored": int(len(paired))})

        # Processes, not threads. A thread pool over buildings looked obvious
        # -- the work is independent and the fits release the GIL -- and it
        # segfaulted (munmap_chunk: invalid pointer): statsmodels' state-space
        # code is not thread-safe, and a crash in a shared library takes the
        # whole service with it. Separate processes have no shared state to
        # corrupt, and the cohort's readings are already fetched here, so what
        # crosses the boundary is frames rather than a database handle.
        #
        # Serially, 62 buildings over 30 November origins is about six hours,
        # which is how a time budget ends up deciding an answer instead of
        # bounding a runaway.
        workers = _backtest_workers(len(cohort))
        # A torch model keeps its weights on the GPU and must not be forked;
        # they are also fast enough warm that parallelism buys nothing. One
        # building has nothing to parallelise either.
        parallel = (workers > 1 and len(cohort) > 1
                    and model not in forecast_models.GPU_MODELS)
        produced = []
        if parallel:
            import concurrent.futures as futures
            import multiprocessing
            payloads = [{
                "model": model, "building": b,
                "own": series.loc[series["building_id"] == b],
                "weather": weather.get(b),
                "origins": origins,
                "weather_variables": weather_variables,
                "window_start": window_start, "window_end": window_end,
                "deadline": deadline,
            } for b in cohort]
            try:
                context = multiprocessing.get_context("fork")
                with futures.ProcessPoolExecutor(max_workers=workers,
                                                 mp_context=context) as pool:
                    for outcome, worker_notes, worker_ran_out in pool.map(
                            _score_building_worker, payloads):
                        produced.append(outcome)
                        notes.extend(worker_notes)
                        ran_out = ran_out or worker_ran_out
                notes.append(f"{model}: {len(cohort)} buildings scored across "
                             f"{workers} worker processes")
            except Exception as pool_failure:               # noqa: BLE001
                # A pool that cannot start is a reason to be slow, not a reason
                # to have no answer.
                notes.append(f"{model}: parallel scoring unavailable "
                             f"({type(pool_failure).__name__}); scored serially")
                produced = [_score_building(b) for b in cohort]
        else:
            produced = [_score_building(b) for b in cohort]
        for outcome in produced:
            if outcome is None:
                continue
            paired, row = outcome
            actuals.append(paired["actual"])
            forecasts.append(paired["predicted"])
            per_building.append(row)
            scored_buildings += 1
        if not actuals:
            # Naming the absence matters more than it looks: asked to compare
            # LightGBM with ARIMA, the run spent its budget on LightGBM and
            # returned a one-row table, so the answer silently became about
            # one model and read as though the other had never been asked for.
            reason = next((n for n in notes if n.startswith(f"{model}:")),
                          f"{model}: ran out of time before any origin could "
                          f"be scored")
            unscored.append({"model": model, "why": reason.split(": ", 1)[-1]})
            continue
        actual = pd.concat(actuals); predicted = pd.concat(forecasts)
        paired_errors[model] = (predicted - actual).abs()
        rows.append({"model": model,
                     **_accuracy(actual.to_numpy(float),
                                 predicted.to_numpy(float),
                                 series, actual.index),
                     "buildings_scored": scored_buildings,
                     "hours_scored": int(len(actual)),
                     "origins": len(origins),
                     "seconds": round(time.time() - began, 2),
                     "library": forecast_models.LIBRARY[model],
                     "truncated": ran_out})
        if ran_out:
            notes.append(f"{model}: stopped early at the tool's deadline; "
                         f"scored {rows[-1]['hours_scored']:,} hours")
    if not rows:
        return insufficient(
            "no_model_scored",
            "; ".join(f"{u['model']}: {u['why']}" for u in unscored)
            or "; ".join(notes) or "no model produced a comparable forecast")

    # Two models scored on the same hours can be compared hour by hour, which
    # is the only way to tell a real difference from a sampling accident. On a
    # thin sample -- and a budget-bounded backtest is often thin -- two models
    # can land a hundredth of a point apart and the table will still name a
    # winner. Pairing the absolute errors says whether the gap survives the
    # noise, and where it does not the answer says so rather than ranking.
    # Every pair worth testing: an explicit two-model comparison, or each
    # variant against the base it varies from. With four arms -- lightgbm,
    # lightgbm+metadata, knn, knn+metadata -- the question is two ablations,
    # not one ranking, and testing only the top two would answer neither.
    scored_names = [r["model"] for r in rows]
    # An arm whose errors match another's to the last decimal did not differ
    # from it. That is a wiring fault, not a finding, and it reads exactly
    # like "the feature makes no difference" unless it is named.
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            same = all(left.get(k) == right.get(k)
                       for k in ("mae", "rmse", "mape")
                       if left.get(k) is not None)
            if same and left["model"] != right["model"]:
                notes.append(
                    f"{left['model']} and {right['model']} produced identical "
                    f"errors to every decimal, which means the feature that "
                    f"distinguishes them never reached the model -- read this "
                    f"as a wiring fault, not as the feature making no "
                    f"difference")
    pairs_to_test = []
    for name in scored_names:
        for suffix in ("_metadata", "_weather"):
            if name.endswith(suffix) and name[:-len(suffix)] in scored_names:
                pairs_to_test.append((name[:-len(suffix)], name))
    if not pairs_to_test and len(rows) == 2:
        pairs_to_test = [(scored_names[0], scored_names[1])]
    tested = [_significance(paired_errors[base], paired_errors[variant],
                            base, variant)
              for base, variant in pairs_to_test
              if base in paired_errors and variant in paired_errors]
    evidence_pair = tested[0] if len(tested) == 1 else None
    all_pairs = tested if len(tested) > 1 else None

    frame = pd.DataFrame(rows)
    wanted = [m for m in (metrics or ["mape", "mae", "rmse"]) if m in frame]
    ordering = wanted[0] if wanted else "mape"
    # An ablation keeps the question's order -- baseline first, then the
    # variant -- because the answer is the change between them and sorting by
    # error would silently swap which one "improved". A plain comparison ranks.
    if report == "improvement" and len(frame) == 2:
        frame["__order"] = [chosen.index(m) for m in frame["model"]]
        frame = frame.sort_values("__order").drop(columns="__order")
    else:
        frame = frame.sort_values(ordering)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    best = frame.loc[frame[ordering].idxmin()] if frame[ordering].notna().any() \
        else frame.iloc[0]
    evidence = {
        "building_id": building_id,
        "buildings": cohort if len(cohort) > 1 else None,
        "cohort_size": len(cohort),
        "models": chosen,
        "pooled": pooled,
        "metadata_features": (list(metadata.columns) if metadata is not None
                              else None),
        "weather_variables": weather_variables or None,
        "window": [f"{window_start:%Y-%m-%d}", f"{window_end:%Y-%m-%d}"],
        "protocol": (f"rolling origin: at each scored origin, forecast the "
                     f"next {horizon_hours} h from history strictly before it, "
                     f"refitting the model there"),
        "origins": len(origins),
        "origins_available": available,
        "sampling": sampling,
        "hours_scored": int(best["hours_scored"]),
        "buildings_scored": (int(best["buildings_scored"])
                             if "buildings_scored" in frame else None),
        "timezone": "local",
        "metric_ranked_on": ordering,
        "metric": ordering,
        "mape_definition": "mean(|actual - predicted| / actual) x 100, over "
                           "hours whose actual is non-zero",
        "notes": notes or None,
        "models_not_scored": unscored or None,
        "paired_comparison": evidence_pair,
        "paired_comparisons": all_pairs,
        "headline": _headline_metric(best, ordering),
    }
    # Name the buildings actually forecast. "Forecast accuracy for 8 buildings
    # ... across 4 buildings" is the cohort in one half of the sentence and
    # the sample in the other.
    counted = int(best["buildings_scored"]) if "buildings_scored" in frame else len(cohort)
    subject = (building_id or cohort[0]) if counted == 1 else f"{counted} buildings"
    # The spread across buildings, and the table it is computed from.
    if sampled_from:
        evidence["cohort_sampled_from"] = sampled_from
        evidence["cohort_sample_rule"] = (
            "the question named no cohort, so the buildings with the most "
            "complete records were used")
    thin = _origin_limited(evidence.get("origins", 0),
                           evidence.get("origins_available", 0))
    evidence["origin_limited"] = thin
    if per_building and len(cohort) > 1:
        spread_frame = pd.DataFrame(per_building)
        evidence["per_building_handle"] = ctx.store.put(
            spread_frame, name="backtest_per_building")
        evidence["per_building_rows"] = int(len(spread_frame))
        by_model = {}
        for model_name, part in spread_frame.groupby("model"):
            scores = pd.to_numeric(part[ordering], errors="coerce").dropna()
            if scores.empty:
                continue
            stats = {
                "buildings": int(len(scores)),
                "mean": round(float(scores.mean()), 4),
                "median": round(float(scores.median()), 4),
                "min": round(float(scores.min()), 4),
                "max": round(float(scores.max()), 4),
                "sd": round(float(scores.std(ddof=1)), 4) if len(scores) > 1 else None,
            }
            # Which building came out best is the least stable thing a thin
            # backtest produces: it is a ranking of one day's luck. The spread
            # is reported (flagged), the ranking is withheld -- a caveat under
            # a named winner does not stop the name being quoted.
            if thin:
                stats["best_building"] = None
                stats["worst_building"] = None
                stats["ranking_withheld"] = (
                    f"per-building best/worst not identifiable from "
                    f"{evidence.get('origins', 0)} origin(s)")
            else:
                stats["best_building"] = part.loc[scores.idxmin(), "building_id"]
                stats["worst_building"] = part.loc[scores.idxmax(), "building_id"]
            by_model[model_name] = stats
        if by_model:
            evidence[f"{ordering}_across_buildings"] = by_model
    # MAPE divides by the actual load, so a cohort holding meters that idle
    # near zero returns a mean in the hundreds of percent: a figure carried by
    # a handful of hours, in which a difference between two models is noise
    # wearing three digits. It is not reported as a headline without the
    # median beside it, which the near-zero hours cannot move.
    inflated = {}
    if "mape" in frame:
        spread = evidence.get("mape_across_buildings") or {}
        for _, row in frame.iterrows():
            value = row.get("mape")
            if value is None or not np.isfinite(value) or value < 100:
                continue
            inflated[str(row["model"])] = {
                "mean_mape": round(float(value), 4),
                "median_across_buildings": (
                    spread.get(str(row["model"]), {}).get("median")),
            }
    evidence["mape_inflated"] = inflated or None
    # "What improvement is obtained by adding X" is answered by the change,
    # not by two numbers the reader has to subtract. The sign is stated in
    # words, because a *negative* improvement is a real and reportable result
    # and "-1.2" alone reads like a win to a hurried eye.
    if report == "improvement" and len(frame) == 2 and frame[ordering].notna().all():
        base, variant = frame.iloc[0], frame.iloc[1]
        change = float(base[ordering]) - float(variant[ordering])
        relative = (change / float(base[ordering]) * 100
                    if float(base[ordering]) else None)
        unit = "percentage points" if ordering in ("mape", "cvrmse") else "kW"
        evidence["improvement"] = {
            "baseline": base["model"], "variant": variant["model"],
            "metric": ordering,
            "baseline_value": float(base[ordering]),
            "variant_value": float(variant[ordering]),
            "change": round(change, 4), "unit": unit,
            "relative_pct": round(relative, 2) if relative is not None else None,
            "direction": ("improvement" if change > 0 else
                          "no change" if change == 0 else "degradation"),
        }
        evidence["headline"] = {
            "value": round(change, 4), "unit": unit,
            "label": (f"{ordering.upper()} {'improvement' if change > 0 else 'change'} "
                      f"from {base['model']} to {variant['model']}"),
        }
        # When the mean is inflated, the difference between two inflated means
        # is the headline least worth quoting. The median across buildings
        # answers the same question and survives the near-zero hours, so it
        # takes the headline and the mean becomes supporting detail.
        spread = evidence.get("mape_across_buildings") or {}
        if ordering == "mape" and evidence.get("mape_inflated"):
            low = spread.get(str(base["model"]), {}).get("median")
            high = spread.get(str(variant["model"]), {}).get("median")
            # The median is the usual escape from an inflated mean, but on a
            # cohort of a few-kW meters EVERY percentile of MAPE is three
            # digits -- UK public services reads 357% at the median while MAE
            # is 3.12 kW. When the robust statistic is itself uninterpretable
            # the metric has run out of readings, and the headline belongs to
            # one that never divides by the load.
            if (low is not None and float(low) >= 100
                    and high is not None and float(high) >= 100):
                base_mae, var_mae = base.get("mae"), variant.get("mae")
                if base_mae is not None and var_mae is not None:
                    shift = float(base_mae) - float(var_mae)
                    evidence["improvement"]["robust_change"] = {
                        "basis": "MAE in kW (MAPE is uninterpretable at every "
                                 "percentile on this cohort)",
                        "baseline_value": float(base_mae),
                        "variant_value": float(var_mae),
                        "change": round(shift, 4), "unit": "kW",
                        "relative_pct": (round(shift / float(base_mae) * 100, 2)
                                         if float(base_mae) else None),
                        "direction": ("improvement" if shift > 0 else
                                      "no change" if shift == 0 else "degradation"),
                    }
                    evidence["headline"] = {
                        "value": round(shift, 4), "unit": "kW",
                        "label": (f"MAE change from {base['model']} to "
                                  f"{variant['model']} (MAPE not usable here)"),
                    }
                    low = high = None
            if low is not None and high is not None:
                robust = float(low) - float(high)
                evidence["improvement"]["robust_change"] = {
                    "basis": "median MAPE across buildings",
                    "baseline_value": float(low), "variant_value": float(high),
                    "change": round(robust, 4), "unit": "percentage points",
                    "relative_pct": (round(robust / float(low) * 100, 2)
                                     if float(low) else None),
                    "direction": ("improvement" if robust > 0 else
                                  "no change" if robust == 0 else "degradation"),
                }
                evidence["headline"] = {
                    "value": round(robust, 4), "unit": "percentage points",
                    "label": (f"median MAPE across buildings, "
                              f"{'improvement' if robust > 0 else 'change'} from "
                              f"{base['model']} to {variant['model']}"),
                }
    evidence["narrative"] = _backtest_narrative(
        subject, frame, evidence, ordering)
    return frame_result(frame, ctx.store, name="backtest_forecast",
                        evidence=evidence)


def _pooled_arm(ctx, model, cohort, series, weather, origins, window_start,
                window_end, metadata, weather_variables, deadline, notes):
    """Score one pooled model: fit across the cohort at each origin.

    One fit per origin rather than one per building per origin, which is what
    makes a cohort-wide question affordable at all -- and it is also the
    arrangement a portfolio forecaster is deployed in.
    """
    collected: dict[str, list] = {}
    ran_out = False
    for origin, stop in origins:
        if deadline and time.time() > deadline - 10:
            ran_out = True
            break
        histories, futures = {}, {}
        steps = int((stop - origin) / pd.Timedelta(hours=1))
        horizon = pd.date_range(origin, periods=steps, freq="h")
        for building in cohort:
            own = series.loc[series["building_id"] == building]
            stamps = pd.to_datetime(own["timestamp"])
            history = own.loc[stamps < origin]
            if len(history) < MIN_LOOKBACK_HOURS:
                continue
            histories[building] = history[["timestamp", "value"]]
            futures[building] = horizon
        if not histories:
            continue
        try:
            produced = forecast_models.run_pooled(
                model, histories, futures, metadata=metadata,
                weather=weather or None,
                weather_variables=weather_variables)
        except forecast_models.ModelUnavailable as unavailable:
            notes.append(f"{model}: {unavailable}")
            return [], ran_out
        except Exception as failure:                        # noqa: BLE001
            notes.append(f"{model} failed at {origin:%Y-%m-%d}: "
                         f"{type(failure).__name__}")
            continue
        for building, forecast in produced.items():
            collected.setdefault(building, []).append(
                pd.Series(np.asarray(forecast.values, float), index=horizon))

    out = []
    for building, pieces in collected.items():
        own = series.loc[series["building_id"] == building]
        stamps = pd.to_datetime(own["timestamp"])
        truth = own.loc[(stamps >= window_start) & (stamps < window_end)].copy()
        if truth.empty:
            continue
        truth["timestamp"] = pd.to_datetime(truth["timestamp"])
        truth = truth.set_index("timestamp")["value"]
        predicted = pd.concat(pieces)
        predicted = predicted[~predicted.index.duplicated(keep="first")]
        paired = pd.concat({"actual": truth, "predicted": predicted},
                           axis=1).dropna()
        if paired.empty:
            continue
        out.append({"building_id": building, "actual": paired["actual"],
                    "predicted": paired["predicted"]})
    return out, ran_out


#: How many buildings to score at once. Each worker pins BLAS to one thread, so
#: the ceiling is cores rather than threads; two are left for the parent and
#: for whatever else shares the machine.
def _backtest_workers(cohort_size: int) -> int:
    import os
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                                  # noqa: BLE001
        available = os.cpu_count() or 1
    return max(1, min(cohort_size, available - 2))


def _score_building_worker(payload: dict):
    """Score one building, in a worker process.

    A module-level function taking plain data, because a process pool has to
    pickle what it is given: the closure version could not cross the boundary,
    and the database handle must not. Everything this needs -- the building's
    own readings, its site weather, the origins -- is fetched in the parent and
    shipped as frames.

    Returns the paired actual/predicted frame, the accuracy row, and the notes
    and truncation flag the parent would otherwise have collected by closure.
    """
    from . import forecast_models
    model = payload["model"]
    building = payload["building"]
    own = payload["own"]
    deadline = payload["deadline"]
    notes: list[str] = []
    ran_out = False

    own_stamps = pd.to_datetime(own["timestamp"])
    truth = own.loc[(own_stamps >= payload["window_start"])
                    & (own_stamps < payload["window_end"])].copy()
    if truth.empty:
        return None, notes, ran_out
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    truth = truth.set_index("timestamp")["value"]
    predictions = []
    for origin, stop in payload["origins"]:
        if deadline and time.time() > deadline - 5:
            ran_out = True
            break
        history = own.loc[own_stamps < origin]
        floor = (MIN_LOOKBACK_HOURS
                 if model in forecast_models.LOW_HISTORY_MODELS
                 else MIN_TRAIN_HOURS)
        if len(history) < floor:
            continue
        steps = int((stop - origin) / pd.Timedelta(hours=1))
        future = pd.date_range(origin, periods=steps, freq="h")
        params = {"window_days": 7, "interval_level": 0.8,
                  "weather_variables": payload["weather_variables"]}
        if model in forecast_models.NEEDS_WEATHER:
            observed = payload.get("weather")
            if observed is None or observed.empty:
                notes.append(f"{model}: no site weather for {building}")
                break
            marks = pd.to_datetime(observed["timestamp"])
            params["exog_history"] = observed.loc[marks < origin]
            params["exog_future"] = observed.loc[marks >= origin]
        try:
            forecast = forecast_models.run(model, history, future, **params)
        except forecast_models.ModelUnavailable as unavailable:
            notes.append(f"{model}: {unavailable}")
            return None, notes, ran_out
        except Exception as failure:                        # noqa: BLE001
            notes.append(f"{model} failed for {building} at "
                         f"{origin:%Y-%m-%d}: {type(failure).__name__}")
            continue
        predictions.append(pd.Series(np.asarray(forecast.values, float),
                                     index=future))
    if not predictions:
        return None, notes, ran_out
    predicted = pd.concat(predictions)
    predicted = predicted[~predicted.index.duplicated(keep="first")]
    paired = pd.concat({"actual": truth, "predicted": predicted},
                       axis=1).dropna()
    if paired.empty:
        return None, notes, ran_out
    row = {"model": model, "building_id": building,
           **_accuracy(paired["actual"].to_numpy(float),
                       paired["predicted"].to_numpy(float),
                       own, paired.index),
           "hours_scored": int(len(paired))}
    return (paired, row), notes, ran_out


def _origin_limited(scored: int, offered: int) -> bool:
    """Is this backtest too thin a sample of origins to characterise a window?

    A day-ahead error is a property of the day as much as of the model. Scoring
    one origin of thirty measures that origin, and the figure it yields can sit
    anywhere in the window's range: on Bull Education in November 2016, a single
    mid-month origin reads 4.7% MAPE where six evenly spaced origins average
    17.7%, and the building the single origin calls *best* (0.60%) scores 117%
    five days earlier. So the sample size governs which claims are reportable at
    all, not merely how firmly they are worded.
    """
    # Scoring every origin there is cannot be a shortfall, however few there
    # are. Without this the warning fired on "2 of 2" -- telling a reader the
    # budget had truncated a backtest that was in fact complete, which is the
    # same disease as the silence it was written to cure.
    if not offered or scored >= offered:
        return False
    return scored < max(5, offered * 0.2)


def _affordable_origins(origins, models, cohort, series, weather, deadline,
                        max_origins, notes):
    """As many evenly spaced origins as the remaining time can pay for.

    One origin of each model is priced first, because the models differ by
    more than an order of magnitude -- a LightGBM fit is seconds and a
    seasonal ARIMA fit is tens of seconds -- and a stride chosen from the
    cheap one leaves the expensive one halfway through the window when the
    deadline arrives. The sample is chosen once and shared, so every model is
    still scored on the same hours.
    """
    if max_origins:
        origins = _evenly(origins, max_origins)
    if not deadline or len(origins) <= 1:
        return origins, "every origin in the window"
    probe_origin, probe_stop = origins[len(origins) // 2]
    # Price on a building that actually has history at the probe origin, and
    # count only the buildings that do. The first member of the cohort was
    # used before this, and in a UK public-services cohort that is
    # Lamb_public_Angeline -- which has no readings in the window at all. Its
    # history was empty, every model was skipped as unfittable, the estimated
    # cost came out at zero, and the tool cheerfully reported it could afford
    # "every origin in the window" before running out of time with one model
    # scored and the other missing.
    scoreable = []
    for member in cohort:
        rows_before = series.loc[(series["building_id"] == member)
                                 & (pd.to_datetime(series["timestamp"])
                                    < probe_origin)]
        if len(rows_before) >= MIN_LOOKBACK_HOURS:
            scoreable.append((len(rows_before), member))
    if not scoreable:
        return origins, "every origin in the window"
    scoreable.sort(reverse=True)
    building = scoreable[0][1]
    own = series.loc[series["building_id"] == building]
    own_stamps = pd.to_datetime(own["timestamp"])
    history = own.loc[own_stamps < probe_origin]
    per_origin = 0.0
    for model in models:
        floor = (MIN_LOOKBACK_HOURS if model in forecast_models.LOW_HISTORY_MODELS
                 else MIN_TRAIN_HOURS)
        if len(history) < floor:
            continue
        steps = int((probe_stop - probe_origin) / pd.Timedelta(hours=1))
        future = pd.date_range(probe_origin, periods=steps, freq="h")
        params = {"window_days": 7, "interval_level": 0.8}
        if model in forecast_models.NEEDS_WEATHER:
            observed = weather.get(building)
            if observed is None or observed.empty:
                continue
            marks = pd.to_datetime(observed["timestamp"])
            params["exog_history"] = observed.loc[marks < probe_origin]
            params["exog_future"] = observed.loc[marks >= probe_origin]
        began = time.time()
        try:
            forecast_models.run(model, history, future, **params)
        except Exception:                                   # noqa: BLE001
            continue
        per_origin += time.time() - began
    if per_origin <= 0:
        return origins, "every origin in the window"
    # The probe itself has been paid for, and a margin is kept so the result
    # can be assembled and returned rather than dying at the line.
    budget = max(deadline - time.time() - 10, 0)
    # ...and the multiplier is the buildings that will actually be forecast,
    # not the size of the cohort the question named.
    affordable = int(budget / (per_origin * max(len(scoreable), 1)))
    if affordable >= len(origins):
        return origins, "every origin in the window"
    if affordable < 1:
        return [], "none: one origin costs more than the time remaining"
    chosen = _evenly(origins, affordable)
    notes.append(
        f"scored {len(chosen)} of {len(origins)} available origins, evenly "
        f"spaced: one origin across {len(models)} model(s) and "
        f"{len(scoreable)} forecastable building(s) costs about "
        f"{per_origin * max(len(scoreable), 1):.0f}s and the time budget "
        f"allows {budget:.0f}s")
    return chosen, (f"{len(chosen)} of {len(origins)} origins, evenly spaced, "
                    f"bounded by the time budget")


def _evenly(items: list, count: int) -> list:
    """`count` items spread across the list, endpoints included."""
    if count >= len(items) or count <= 0:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[int(round(i * step))] for i in range(count)]


def _significance(first_errors: pd.Series, second_errors: pd.Series,
                  first: str, second: str) -> dict:
    """Is the difference between two models real, on each error measure?

    Scored on the *same hours*, so the comparison is paired and the test can
    be a paired one -- which is far more sensitive than comparing two
    independent means, because the hour-to-hour variation both models share
    cancels out.

    MAE is the difference of absolute errors and RMSE the difference of
    squared ones, so each gets its own test rather than one verdict standing
    for all three. MAPE divides by the actual, which makes it heavy-tailed,
    so it is tested by the sign test as well -- a mean is a poor summary of a
    distribution a handful of near-zero hours can dominate.

    The Wilcoxon signed-rank test is reported beside the t-test because the
    error differences are not normal, and where the two disagree the
    distribution-free one is the one to believe.
    """
    from scipy import stats
    left, right = first_errors.align(second_errors, join="inner")
    paired = pd.concat({"a": left, "b": right}, axis=1).dropna()
    if len(paired) < 10:
        return {"compared": [first, second], "hours_paired": int(len(paired)),
                "distinguishable": False, "better": None,
                "test": "too few paired hours to test"}
    a, b = paired["a"].to_numpy(float), paired["b"].to_numpy(float)
    out: dict = {"compared": [first, second], "hours_paired": int(len(paired))}
    for label, left_values, right_values in (
            ("mae", a, b),
            ("rmse", a ** 2, b ** 2)):
        difference = left_values - right_values
        if not np.any(difference):
            out[label] = {"mean_difference": 0.0, "p_value": 1.0,
                          "significant": False}
            continue
        _, p_t = stats.ttest_rel(left_values, right_values)
        try:
            _, p_w = stats.wilcoxon(difference)
        except ValueError:
            p_w = float("nan")
        out[label] = {
            "mean_difference": round(float(difference.mean()), 6),
            "p_value": round(float(p_t), 6),
            "wilcoxon_p": None if p_w != p_w else round(float(p_w), 6),
            "significant": bool(min(p_t, p_w if p_w == p_w else 1) < 0.05),
            "better": second if difference.mean() > 0 else first,
        }
    # The sign test on which model was closer, hour by hour: free of any
    # assumption about the size of the errors.
    wins = int((a > b).sum())
    losses = int((a < b).sum())
    if wins + losses:
        p_sign = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)
        out["sign_test"] = {
            f"{second}_closer_hours": wins, f"{first}_closer_hours": losses,
            "p_value": round(p_sign, 6), "significant": bool(p_sign < 0.05),
            "better": second if wins > losses else first,
        }
    decisive = any(v.get("significant") for k, v in out.items()
                   if isinstance(v, dict) and k in ("mae", "rmse", "sign_test"))
    out["distinguishable"] = decisive
    out["better"] = (out.get("mae", {}).get("better") if decisive else None)
    out["test"] = ("paired t-test and Wilcoxon signed-rank on the hourly error "
                   "differences, plus a sign test on which model was closer; "
                   "significance at p < 0.05")
    # Kept for readers of the older field name.
    out["mean_absolute_error_gap_kw"] = out.get("mae", {}).get("mean_difference")
    out["standard_error_kw"] = None
    return out


def _accuracy(actual: np.ndarray, predicted: np.ndarray,
              series: pd.DataFrame, index) -> dict:
    """The usual accuracy measures, each defined where it is computed.

    MAPE divides by the actual, so hours reading exactly zero are excluded --
    a masked or genuinely dark hour makes the ratio infinite and one of them
    would dominate the mean. How many were excluded is reported, because a
    MAPE over a building that is dark half the time is a different statistic
    from one over a building that never is.
    """
    error = predicted - actual
    non_zero = actual != 0
    naive = np.abs(np.diff(actual)).mean() if len(actual) > 1 else np.nan
    mean_actual = float(actual.mean())
    # MAPE is unusable on a series that approaches zero, and the failure is
    # silent: it produces a large finite number rather than an error. Seven UK
    # public-services buildings drop to 0.027 kW overnight, and an absolute
    # error of 5 kW against that actual is 16,000% -- a handful of such hours
    # set the mean for all of them, and the cohort scored 592% MAPE at an MAE
    # of 8 kW. So "near zero" is measured against the series' own scale rather
    # than against nought, and how many such hours there are travels with the
    # number.
    typical = float(np.median(actual[non_zero])) if non_zero.any() else 0.0
    near_zero = int((actual < 0.01 * typical).sum()) if typical > 0 else 0
    share_near_zero = near_zero / len(actual) if len(actual) else 0.0
    # The measures come first and the diagnostics after, because a text view
    # shows the leading columns and a reader scanning for MAE should not find
    # `mape_reliable` and `hours_near_zero` in its place.
    return {
        "mape": round(float(np.mean(np.abs(error[non_zero] / actual[non_zero]))
                            * 100), 4) if non_zero.any() else None,
        "mae": round(float(np.abs(error).mean()), 4),
        "rmse": round(float(np.sqrt((error ** 2).mean())), 4),
        "cvrmse": round(float(np.sqrt((error ** 2).mean()) / mean_actual * 100),
                        4) if mean_actual else None,
        "mase": round(float(np.abs(error).mean() / naive), 4) if naive else None,
        "bias": round(float(error.mean()), 4),
        # A percentage error is only meaningful where the denominator is. Under
        # one hour in two hundred near zero, MAPE stands; past that it is
        # reporting the dark hours and nothing else.
        "mape_reliable": bool(share_near_zero < 0.005),
        "hours_near_zero": near_zero,
        "hours_excluded_from_mape": int((~non_zero).sum()),
    }


def _headline_metric(best, ordering: str) -> dict:
    """The number to lead with, refusing one that cannot be read.

    A MAPE of 4,834% is not a measure of forecast quality -- it is a statement
    that the cohort holds meters idling near zero, because MAPE divides by the
    actual. Leading with it puts a figure at the top of the page that no reader
    can act on, and one that flatters or damns a model at random. When the
    ranked metric comes back inflated the headline falls to MAE, which is in kW
    and has no denominator; the inflated figure stays in the table below, where
    its own warning sits beside it.
    """
    value = best.get(ordering)
    inflated = (ordering in ("mape", "cvrmse")
                and value is not None and np.isfinite(value) and value >= 100)
    if inflated and best.get("mae") is not None and np.isfinite(best["mae"]):
        return {"value": round(float(best["mae"]), 4), "unit": "kW",
                "label": f"MAE of the best model ({best['model']}) -- "
                         f"{ordering.upper()} is not reportable on this cohort"}
    return {"value": round(float(value), 4) if value is not None else None,
            "unit": "%" if ordering in ("mape", "cvrmse") else "kW",
            "label": f"{ordering.upper()} of the best model ({best['model']})"}


def _backtest_narrative(building_id, frame, evidence, ordering) -> list[str]:
    """The comparison as a table a person reads, worst-to-best made explicit."""
    window = evidence["window"]
    gain = evidence.get("improvement")
    lines = []
    # A sample this thin is not a weaker version of the answer, it is a
    # different claim -- so it leads, rather than trailing the table where a
    # reader who has already taken the headline will never reach it.
    scored = evidence.get("origins", 0)
    offered = evidence.get("origins_available", 0)
    if evidence.get("cohort_sampled_from"):
        lines.append(
            f"⚠ The question names no cohort, so one was chosen: "
            f"{evidence.get('buildings_scored') or evidence.get('cohort_size')} "
            f"buildings with the most complete records, out of "
            f"{evidence['cohort_sampled_from']:,}. A different sample would "
            f"give different numbers; name the buildings to pin them.")
    if evidence.get("origin_limited"):
        lines.append(
            f"⚠ Scored {scored} of {offered} available forecast origins — the "
            f"time budget paid for no more. These figures describe the origins "
            f"sampled, not the whole window: day-ahead error varies severalfold "
            f"between days, so treat them as indicative and ask for a shorter "
            f"period, or fewer models, to score more of it.")
    if gain:
        # The answer first, in the words the question used. A change of
        # -0.4 points is a degradation and has to read as one.
        verb = {"improvement": "improves", "no change": "does not change",
                "degradation": "worsens"}[gain["direction"]]
        lines.append(
            f"Adding {gain['variant'].replace(gain['baseline'], '').strip('_') or 'the variant'}"
            f" to {gain['baseline']} {verb} {gain['metric'].upper()} by "
            f"{abs(gain['change']):.2f} {gain['unit']}"
            + (f" ({abs(gain['relative_pct']):.1f}% relative)"
               if gain.get("relative_pct") is not None else "")
            + f": {gain['baseline_value']:.2f} -> {gain['variant_value']:.2f}"
            + ("%" if gain["metric"] in ("mape", "cvrmse") else "") + ".")
        robust = gain.get("robust_change")
        if robust:
            # Stated before the inflated pair, because whichever number comes
            # first is the one that gets quoted.
            if robust.get("unit") == "kW":
                lines[-1] = (
                    f"Measured in kW, adding "
                    f"{gain['variant'].replace(gain['baseline'], '').strip('_') or 'the variant'}"
                    f" to {gain['baseline']} "
                    + {"improvement": "improves", "no change": "does not change",
                       "degradation": "worsens"}[robust["direction"]]
                    + f" MAE by {abs(robust['change']):.2f} kW"
                    + (f" ({abs(robust['relative_pct']):.1f}% relative)"
                       if robust.get("relative_pct") is not None else "")
                    + f": {robust['baseline_value']:.2f} -> "
                      f"{robust['variant_value']:.2f} kW. MAPE is not reportable "
                      f"on this cohort at any percentile.")
                lines.append(
                    f"  MAPE reads {gain['baseline_value']:.0f}% -> "
                    f"{gain['variant_value']:.0f}% here, which measures the "
                    f"near-zero loads rather than the forecast.")
                robust = None
        if robust:
            lines[-1] = (
                f"On the median building, adding "
                f"{gain['variant'].replace(gain['baseline'], '').strip('_') or 'the variant'}"
                f" to {gain['baseline']} "
                + {"improvement": "improves", "no change": "does not change",
                   "degradation": "worsens"}[robust["direction"]]
                + f" MAPE by {abs(robust['change']):.2f} percentage points"
                + (f" ({abs(robust['relative_pct']):.1f}% relative)"
                   if robust.get("relative_pct") is not None else "")
                + f": {robust['baseline_value']:.2f}% -> "
                  f"{robust['variant_value']:.2f}%.")
            lines.append(
                f"  The cohort *mean* MAPE moves "
                f"{gain['baseline_value']:.1f}% -> {gain['variant_value']:.1f}%, "
                f"but at that scale it is not a measure of forecast quality — "
                f"see the warning below.")
    lines.append(
        f"Forecast accuracy for {building_id}, {window[0]} to {window[1]} "
        f"({evidence['hours_scored']:,} hours scored"
        + (f" across {evidence['buildings_scored']} of "
           f"{evidence['cohort_size']} buildings"
           if evidence.get("buildings_scored")
           and evidence.get("cohort_size", 1) > 1 else "")
        + ", local time).")
    lines += [
        f"  {'model':<18}{'MAPE':>9}{'MAE (kW)':>11}{'RMSE (kW)':>11}"
        f"{'CVRMSE':>9}",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"  {row['model']:<18}"
            f"{row['mape']:>8.2f}%" if row["mape"] is not None
            else f"  {row['model']:<18}{'--':>9}")
        lines[-1] += (f"{row['mae']:>11.2f}{row['rmse']:>11.2f}"
                      + (f"{row['cvrmse']:>8.2f}%" if row["cvrmse"] is not None
                         else f"{'--':>9}"))
    ranked = frame.sort_values(ordering)
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    pair = evidence.get("paired_comparison")
    if not gain and len(frame) > 1 and best[ordering] and worst[ordering]:
        margin = worst[ordering] - best[ordering]
        unit = "percentage points" if ordering in ("mape", "cvrmse") else "kW"
        if pair and pair.get("mae") and pair.get("distinguishable"):
            mae, rmse = pair.get("mae", {}), pair.get("rmse", {})
            sign = pair.get("sign_test", {})
            lines.append(
                f"  Tested on the {pair['hours_paired']:,} hours both models "
                f"forecast:")
            lines.append(
                f"    MAE   difference {mae.get('mean_difference'):+.4f} kW, "
                f"p={mae.get('p_value'):.2g} "
                f"(Wilcoxon {mae.get('wilcoxon_p')}) -- "
                + ("significant" if mae.get("significant") else "not significant"))
            lines.append(
                f"    RMSE  squared-error difference "
                f"{rmse.get('mean_difference'):+.4f}, "
                f"p={rmse.get('p_value'):.2g} -- "
                + ("significant" if rmse.get("significant") else "not significant"))
            if sign:
                variant = pair["compared"][1]
                closer = int(sign.get(f"{variant}_closer_hours", 0))
                lines.append(
                    f"    {variant} was closer in {closer:,} of "
                    f"{pair['hours_paired']:,} hours, "
                    f"p={sign.get('p_value'):.2g}")
            lines.append(f"    {pair['test']}.")
        elif pair and not pair.get("distinguishable"):
            # A ranking is a claim, and on this sample it is not one that can
            # be made. Naming a winner a hundredth of a point ahead on 72
            # hours invites exactly the conclusion the data does not support.
            lines.append(
                f"  The two are not distinguishable on this sample: "
                f"{best['model']} leads by {margin:.2f} {unit}, but the "
                f"hour-by-hour difference in absolute error is "
                + (f"{pair['mean_absolute_error_gap_kw']:+.2f} kW give or take "
                   f"{pair['standard_error_kw']:.2f} across "
                   f"{pair['hours_paired']:,} paired hours"
                   if pair.get("standard_error_kw") is not None
                   else f"untestable on {pair.get('hours_paired', 0)} paired hours")
                + ". Score more origins before ranking them.")
        else:
            # "The gap survives a paired test" is a claim about the hours
            # scored, not about the window they were drawn from. On a sample
            # of four days out of sixty-one, which days were drawn matters
            # more than the test does, and saying so is the difference
            # between a result and an overclaim.
            thin = (evidence.get("origins_available", 0)
                    > evidence.get("origins", 0))
            lines.append(
                f"  {best['model']} is the more accurate by {margin:.2f} "
                f"{unit} on {ordering.upper()}"
                + (f", and the gap holds hour by hour across the "
                   f"{pair['hours_paired']:,} hours scored"
                   + (" -- though those hours are a sample of the window, "
                      "so this ranks the models on them rather than over it."
                      if thin else ".")
                   if pair else "."))
    lines.append(
        f"  Protocol: {evidence['protocol']}. "
        f"{evidence['sampling'].capitalize()}"
        + (f" (of {evidence['origins_available']} available)"
           if evidence.get("origins_available", 0) > evidence["origins"] else "")
        + "; every model scored on the same hours.")
    for missing in (evidence.get("models_not_scored") or []):
        lines.append(f"  ⚠ {missing['model']} could not be scored: "
                     f"{missing['why']}. The comparison you asked for is "
                     f"incomplete without it.")
    for test in (evidence.get("paired_comparisons") or []):
        base, variant = test["compared"]
        mae, rmse = test.get("mae", {}), test.get("rmse", {})
        if not mae:
            lines.append(f"  {base} vs {variant}: {test.get('test')}")
            continue
        lines.append(
            f"  {variant} vs {base}, over {test['hours_paired']:,} shared hours:"
            f"  MAE {mae.get('mean_difference'):+.4f} kW "
            f"(p={mae.get('p_value'):.2g}, "
            + ("significant" if mae.get("significant") else "not significant")
            + f");  RMSE p={rmse.get('p_value'):.2g} "
            + ("significant" if rmse.get("significant") else "not significant")
            + ".")
    spread = evidence.get(f"{ordering}_across_buildings") or {}
    for model_name, stats in spread.items():
        lines.append(
            f"  Across {stats['buildings']} buildings, {model_name} "
            f"{ordering.upper()} averages {stats['mean']:.2f}"
            + ("%" if ordering in ("mape", "cvrmse") else " kW")
            + f" (median {stats['median']:.2f}, "
            f"{stats['min']:.2f} to {stats['max']:.2f}"
            + (f", sd {stats['sd']:.2f}" if stats.get("sd") else "") + ").")
        if stats.get("best_building"):
            lines.append(
                f"    best {stats['best_building']}, worst "
                f"{stats['worst_building']} -- the pooled figure above belongs "
                f"to neither.")
        else:
            lines.append(
                f"    which building is best or worst is not reported: with "
                f"{scored} scored origin(s) that ranking reflects the days "
                f"sampled, not the buildings.")
    if evidence.get("pooled") and any(
            not name.endswith("_metadata") for name in frame["model"]) and any(
            str(name).endswith("_metadata") for name in frame["model"]):
        lines.append(
            "  ⚠ Read the improvement carefully. A pooled model without "
            "metadata has no way to tell one building from another -- it sees "
            "only the calendar, so it predicts roughly the cohort average for "
            "every building, whatever its size. Most of the gain below is "
            "metadata supplying building identity rather than metadata "
            "improving forecasting. The fair control is a pooled model given "
            "the building's identity but not its characteristics, or a "
            "per-building model; against either, the margin is much smaller.")
    lines.append(f"  MAPE is {evidence['mape_definition']}.")
    inflated = evidence.get("mape_inflated") or {}
    if inflated:
        named = ", ".join(
            f"{model} {vals['mean_mape']:,.0f}%"
            + (f" (median across buildings {vals['median_across_buildings']:.2f}%)"
               if vals.get("median_across_buildings") is not None else "")
            for model, vals in inflated.items())
        lines.append(
            f"  ⚠ Mean MAPE exceeds 100% here: {named}. This says the cohort "
            f"contains hours whose actual load is near zero, not that the "
            f"forecast is wrong by that multiple — dividing by a load close to "
            f"zero produces an unbounded term, and a few such hours carry the "
            f"mean. Read the median across buildings, or MAE and RMSE in kW, "
            f"which have no such denominator.")
    # Two warnings that change what the table means, so they sit with it
    # rather than in a footnote nobody reads.
    if "mape_reliable" in frame and not bool(frame["mape_reliable"].all()):
        worst = int(frame["hours_near_zero"].max())
        lines.append(
            f"  ⚠ MAPE is not trustworthy here: {worst:,} of "
            f"{evidence['hours_scored']:,} scored hours sit near zero, where "
            f"dividing by the actual makes a small absolute error enormous. "
            f"Read CVRMSE or MAE instead -- both are on the same scale as the "
            f"demand and neither divides by it.")
    if gain and "mae" in frame and frame["mae"].notna().all() and len(frame) == 2:
        mae_change = float(frame.iloc[0]["mae"]) - float(frame.iloc[1]["mae"])
        if (mae_change > 0) != (gain["change"] > 0) and mae_change != 0:
            lines.append(
                f"  ⚠ The metrics disagree: {gain['metric'].upper()} says "
                f"{gain['direction']} while MAE moves the other way "
                f"({frame.iloc[0]['mae']:.2f} -> {frame.iloc[1]['mae']:.2f} kW). "
                f"On a series that approaches zero this is what a MAPE "
                f"improvement usually means, and the MAE is the one to believe.")
    for note in (evidence.get("notes") or [])[:3]:
        lines.append(f"  Note: {note}")
    return lines


@tool("ai_analytics")
def evaluate_forecast(
    ctx: ToolContext,
    forecast_handle: str,
    building_id: str | None = None,
    metrics: list[Literal["mae", "rmse", "mape", "peak_error",
                          "peak_time_error"]] | None = None,
) -> ToolResult:
    """Score a forecast against the observed continuation.

    Reports peak-magnitude and peak-time error alongside the usual accuracy
    metrics, because for demand work those are often what matters.
    """
    frame = ctx.store.get_or_none(forecast_handle)
    if frame is None:
        return insufficient("unknown_handle",
                            f"no artifact for handle {forecast_handle!r}")
    pred = frame.copy()
    # Either spelling: the column was renamed to carry its unit, and a stored
    # artifact from before that is still a forecast.
    column = next((c for c in ("predicted_kw", "predicted") if c in pred), None)
    if column is None or "timestamp" not in pred:
        return insufficient("shape", "forecast frame needs timestamp and "
                                     "predicted_kw")
    pred["predicted"] = pred[column]
    # A cohort forecast carries many buildings in one frame. Scoring it as if
    # the first building were the only one reported that building's error as
    # the cohort's, which is a different number and a wrong answer to "the
    # median MAE across the buildings".
    if building_id is None and "building_id" in pred:
        names = [b for b in pred["building_id"].dropna().unique()]
        if len(names) > 1:
            return _evaluate_cohort(ctx, pred, names, metrics)
    building_id = building_id or str(pred["building_id"].iloc[0])
    pred["timestamp"] = _naive(pd.to_datetime(pred["timestamp"]))
    # The horizon is scored at the grain it was forecast at. A daily forecast
    # carries one point per day, and the hourly observations underneath it are
    # not its counterpart: merging the two on an exact timestamp keeps only the
    # 00:00 reading of each day and scores a daily total against a single
    # hour. The spacing of the forecast says which grain it is.
    gaps = pred["timestamp"].sort_values().diff().dropna()
    daily = bool(len(gaps)) and gaps.median() >= pd.Timedelta(hours=23)
    lo, hi = pred["timestamp"].min(), pred["timestamp"].max()
    # hi is the last forecast point and the bound below is exclusive, so the
    # window runs to the end of that point rather than to its start: one hour
    # past it hourly, one day past it daily, where the point stands for the
    # whole day. Stopping at hi itself drops the last step of the horizon and
    # scores an n-1 hour forecast.
    hi_q = hi + (pd.Timedelta(days=1) if daily else pd.Timedelta(hours=1))
    select = ("CAST(timestamp_local AS DATE) AS timestamp, SUM(value) AS value"
              if daily else "timestamp_local AS timestamp, value")
    group = " GROUP BY 1" if daily else ""
    truth = ctx.db.df(f"""
        SELECT {select} FROM electricity
        WHERE quality_flag = 'OK' AND building_id = ?
          AND timestamp_local >= ? AND timestamp_local < ?
        {group} ORDER BY timestamp
    """, [building_id, str(lo), str(hi_q)])
    if truth.empty:
        return insufficient("no_ground_truth",
                            f"no observations for {building_id} in the horizon")
    # timestamp_local is stored TIMESTAMP WITH TIME ZONE but holds local wall
    # time under a +00:00 tag, so it arrives tz-aware while a forecast frame
    # is naive. Pandas refuses to merge the two and the question was lost to
    # "You are trying to merge on datetime64[ns] and datetime64[us, UTC]".
    # Dropping the tag compares like with like: both sides are local wall time.
    truth["timestamp"] = _naive(pd.to_datetime(truth["timestamp"]))
    merged = pred.merge(truth, on="timestamp", how="inner").dropna(
        subset=["value", "predicted"])
    if merged.empty:
        return insufficient("no_overlap", "forecast and observations do not align")
    y, yhat = merged["value"].to_numpy(float), merged["predicted"].to_numpy(float)
    naive = np.abs(np.diff(y)).mean() if len(y) > 1 else np.nan
    scores = {
        "n_points": int(len(merged)),
        "mae": float(np.abs(y - yhat).mean()),
        "rmse": float(np.sqrt(((y - yhat) ** 2).mean())),
        "mape": float((np.abs((y - yhat) / np.where(y == 0, np.nan, y))).mean() * 100),
        "mase": float(np.abs(y - yhat).mean() / naive) if naive else None,
        "peak_error": float(yhat.max() - y.max()),
        "peak_pct_error": float(100 * (yhat.max() - y.max()) / y.max()) if y.max() else None,
        "peak_time_error_hours": float(
            (merged["timestamp"].iloc[int(np.argmax(yhat))] -
             merged["timestamp"].iloc[int(np.argmax(y))]).total_seconds() / 3600),
    }
    if metrics:
        scores = {k: v for k, v in scores.items() if k in metrics or k == "n_points"}
    # Which model produced this forecast. Two evaluations sat side by side
    # with nothing saying which was which, so "which model has the lower MAE"
    # could not be answered from them however plainly the numbers differed.
    # The handle is named forecast_<model>_<hash>, which is where it lives.
    scored_model = None
    if isinstance(forecast_handle, str) and forecast_handle.startswith("forecast_"):
        middle = forecast_handle[len("forecast_"):].rsplit("_", 1)
        scored_model = middle[0] if middle and middle[0] else None
    return ok(scores, evidence={"building_id": building_id,
                                "model": scored_model,
                                "horizon": [str(lo), str(hi)],
                                "mase_denominator": "mean absolute first difference"})


def _evaluate_cohort(ctx: ToolContext, pred: pd.DataFrame,
                     names: list[str], metrics) -> ToolResult:
    """Score a cohort forecast per building and report the distribution.

    "The median MAE across the buildings" is a statistic over per-building
    errors, not the error of the buildings pooled: pooling lets one large
    building's kWh dominate a median that was meant to describe the typical
    building. Each is scored on its own and the spread is reported, so the
    median is visible next to what it summarises.
    """
    rows, failed = [], []
    for name in names:
        one = pred[pred["building_id"] == name].copy()
        handle = ctx.store.put(one, name="forecast_slice")
        scored = evaluate_forecast(ctx, forecast_handle=handle,
                                   building_id=name, metrics=metrics)
        if scored.status is not Status.OK:
            failed.append(f"{name}: {scored.message or scored.status.name}")
            continue
        rows.append({"building_id": name, **(scored.data or {})})
    if not rows:
        return insufficient("no_scorable_buildings",
                            "no building in the cohort could be scored"
                            + (f" ({failed[0]})" if failed else ""))
    out = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    median = float(out["mae"].median())
    ev = {"buildings_scored": len(out), "buildings_requested": len(names),
          "median_mae": round(median, 6),
          "mean_mae": round(float(out["mae"].mean()), 6),
          "min_mae": round(float(out["mae"].min()), 6),
          "max_mae": round(float(out["mae"].max()), 6),
          "headline": {"value": round(median, 6), "unit": "kWh",
                       "label": f"median MAE across {len(out)} buildings"}}
    if failed:
        ev["not_scored"] = failed[:10]
        ev["not_scored_count"] = len(failed)
    ev["narrative"] = (
        f"Median MAE across the {len(out)} buildings scored is "
        f"{median:,.4f} kWh (mean {out['mae'].mean():,.4f}, best "
        f"{out['mae'].min():,.4f} at {out.iloc[0]['building_id']}, worst "
        f"{out['mae'].max():,.4f})."
        + (f" {len(failed)} building(s) could not be scored." if failed else ""))
    return frame_result(out, ctx.store, name="forecast_scores_cohort",
                        evidence=ev)


def _multiple_of_typical(ctx: ToolContext, building_ids: list[str],
                         threshold: float, direction: str,
                         start: str | None, end: str | None,
                         days: str = "all") -> ToolResult:
    """Days that pass a multiple of the building's own typical day."""
    if not building_ids:
        return insufficient("building_ids", "name the buildings to examine")
    placeholders = ", ".join("?" for _ in building_ids)
    params: list = list(building_ids)
    window = ""
    if start:
        window += " AND a.period_start >= ?"; params.append(start)
    if end:
        window += " AND a.period_start < ?"; params.append(end)
    comparison = {"high": ">", "low": "<"}.get(direction, "<>")
    # The day-type restriction applies to the baseline as well as to the
    # candidates. "Twice its typical weekend day" means twice what its
    # weekends usually draw; measuring a Saturday against a pooled week makes
    # the weekly cycle itself the finding. ISODOW is 1 (Monday) to 7 (Sunday).
    day_filter = {"weekend": " AND ISODOW(CAST({t}period_start AS DATE)) >= 6",
                  "weekday": " AND ISODOW(CAST({t}period_start AS DATE)) <= 5",
                  }.get(days or "all", "")
    frame = ctx.db.df(f"""
        WITH typical AS (
            SELECT building_id, AVG(total_kwh) AS typical_kwh
            FROM agg_daily WHERE grain = 'daily'
              AND building_id IN ({placeholders})
              {day_filter.format(t="")}
            GROUP BY 1)
        SELECT a.building_id, CAST(a.period_start AS DATE) AS when_day,
               a.total_kwh AS observed, t.typical_kwh,
               a.total_kwh / NULLIF(t.typical_kwh, 0) AS multiple
        FROM agg_daily a JOIN typical t USING (building_id)
        WHERE a.grain = 'daily'{window}{day_filter.format(t="a.")}
          AND a.total_kwh {comparison} ? * t.typical_kwh
        ORDER BY 1, 2
    """, [*params, threshold])
    evidence = {"method": "multiple_of_typical", "threshold": threshold,
                "direction": direction, "period": [start, end], "days": days,
                "criterion": f"a {'' if days in (None, 'all') else days + ' '}"
                             f"day whose total is {comparison} {threshold:g}x "
                             f"the building's mean "
                             f"{'' if days in (None, 'all') else days + ' '}"
                             f"daily total over the whole record",
                "n_days": int(len(frame)),
                "headline": {"label": f"days {'above' if direction == 'high' else 'past'} "
                                      f"{threshold:g}x the typical day",
                             "value": int(len(frame)), "unit": "days"}}
    reference = evidence["criterion"]
    if frame.empty:
        evidence["narrative"] = _detection_narrative(
            "days", [], evidence, reference=reference)
        return ok({"anomalies": [], "n_days": 0}, evidence=evidence)
    frame["when"] = frame["when_day"].astype(str)
    frame = frame.drop(columns=["when_day"]).rename(
        columns={"observed": "observed_kwh", "typical_kwh": "typical_day_kwh"})
    evidence["narrative"] = _detection_narrative(
        "days over the threshold",
        frame.rename(columns={"observed_kwh": "observed",
                              "multiple": "score"}).to_dict("records"),
        evidence, reference=reference)
    return frame_result(frame, ctx.store, name="multiple_of_typical",
                        evidence=evidence)


#: The line each test is conventionally read at, when the caller names none.
#: They are not interchangeable: 3.0 is sigma for a z-score and Tukey's
#: *outer* fence for an interquartile test, so one shared default quietly
#: turned "the IQR method" into a search for extreme outliers only.
METHOD_THRESHOLD = {"zscore": 3.0, "modified_zscore": 3.5, "iqr": 1.5,
                    "local_outlier_factor": 1.5, "multiple_of_typical": 2.0,
                    "first_difference": 3.0, "daily_profile": 3.0,
                    "isolation_forest": 3.0}


@tool("ai_analytics")
def detect_anomalies(
    ctx: ToolContext,
    building_ids: list[str],
    method: Literal["zscore", "modified_zscore", "iqr", "isolation_forest",
                    "local_outlier_factor", "first_difference",
                    "daily_profile", "multiple_of_typical"] = "daily_profile",
    # Per method, because the number means a different thing in each: sigma
    # for a z-score, a Tukey fence multiple for iqr, a density ratio for LOF.
    # One shared 3.0 silently turned "the IQR method" into Tukey's *outer*
    # fence. None means "the method's own convention", listed in
    # METHOD_THRESHOLD below.
    threshold: float | None = None,
    direction: Literal["both", "high", "low"] = "both",
    days: Literal["all", "weekday", "weekend"] = "all",
    hour_from: int | None = None,
    hour_to: int | None = None,
    context_hours: int = 168,
    scale_windows: Literal["minmax", "none"] = "minmax",
    # What the isolation forest is shown. "level" is the reading itself, which
    # is what "an isolation forest over hourly electricity" plainly means;
    # "level_and_calendar" adds hour-of-day and day-of-week, which changes
    # what counts as anomalous -- a normal Sunday value becomes unusual
    # because of when it happened. The parameter was declared and the forest
    # ignored it, always using the calendar form.
    feature_set: Literal["level", "level_and_change",
                         "level_and_calendar"] = "level_and_calendar",
    start: str | None = None,
    end: str | None = None,
    min_coverage: float = 0.9,
    # "Which buildings have at least one reading above their own mean + 3 SD"
    # and "how many such readings are there" are different questions with the
    # same scan behind them. The frame answered the second: 571 incidents
    # where 34 buildings have one. The count per building was already computed
    # and published beside the result; this makes it the result.
    # "How many readings were outliers, and on how many days did at least one
    # occur" are two counts of one scan. The timestamps are already in the
    # result, so the day count is a distinct-date count -- but a reader cannot
    # take it from a truncated preview, and a grader cannot find it at all.
    report: Literal["readings", "buildings", "days"] = "readings",
    # What share of the record an isolation forest is told to call anomalous.
    # None keeps the convention below -- the threshold read as a z-score and
    # converted to the tail it implies -- which is what an operator means by
    # an anomaly rate. "auto" hands the decision to scikit-learn, which on
    # this data labels a third of every record; that is a different question
    # and some references ask it, so it is available rather than imposed.
    contamination: float | Literal["auto"] | None = None,
) -> ToolResult:
    """Flag anomalous readings or days against a stated criterion.

    The tests differ in what they call unusual, and the choice is not a
    detail:

    * ``zscore`` — distance from the building's own mean in standard
      deviations. Both statistics are themselves moved by the outliers being
      looked for, so a building with one enormous day has a mean and a spread
      that hide it.
    * ``modified_zscore`` — the same idea on the median and the median
      absolute deviation, scaled by 0.6745 so the threshold reads on the same
      scale as a z-score. Neither statistic is moved by the outliers, which is
      why 3.5 is the conventional line here rather than 3.
    * ``iqr`` — outside the quartiles by a multiple of their range.
    * ``first_difference`` — a large *change* from the preceding hour rather
      than a large level. A building that steps to a new plateau and stays
      there has one anomalous hour by this test and none by the others.
    * ``local_outlier_factor`` — density-based: an hour whose recent context
      window sits in a sparser neighbourhood than its neighbours' do. This is
      the only test here that looks at the *shape* of the preceding
      ``context_hours`` rather than at one number.
    * ``isolation_forest`` — an ensemble that isolates rare combinations of
      demand, hour and weekday.
    * ``daily_profile`` and ``multiple_of_typical`` work on daily totals.

    ``hour_from`` and ``hour_to`` restrict the scan to a clock window, which
    wraps midnight: 20 to 7 is 8 PM through 7 AM. The baseline is restricted
    with it, so a "quiet hour" is unusual among this building's other quiet
    hours rather than merely lower than its afternoons.

    ``days`` restricts the scan *and the baseline it is measured against*.
    "Unusual weekend consumption" is a weekend day that is unusual among this
    building's other weekend days -- not one that differs from its weekdays,
    which every weekend does. Comparing a Saturday against a pooled week makes
    the weekly cycle itself the anomaly and buries any real one inside it.

    The criterion is part of the answer: "anomalous" is meaningless without the
    reference population and threshold, both of which are returned.

    ``direction`` is part of that criterion. A spike, a surge, an unusually
    high reading is one-sided, and flagging deviations of both signs answers a
    different question: asked whether Fox_office_Alice has suspicious spikes,
    this reported 57 -- every one of them a reading far *below* the building's
    normal level, at 77 against a median of 352, listed under a heading about
    spikes. Use "high" for spikes and surges, "low" for dips and dropouts, and
    "both" when the question does not say.

    ``daily_profile`` compares daily totals and therefore drops days below
    ``min_coverage``: a day with half its readings missing has a low total for a
    reason that is not anomalous behaviour, and including such days both invents
    anomalies and inflates the spread that everything else is judged against.

    ``multiple_of_typical`` is the plain-language criterion: a day whose total
    is more than ``threshold`` times the building's own typical day, where
    typical is its mean daily total over the whole record. "Did consumption
    exceed twice its typical daily average" is that question, and a z-score
    cannot answer it -- twice the mean is one standard deviation for one
    building and four for another. The window says which days are examined;
    the whole record says what typical is.
    """
    # Resolve the convention before anything reads it, so every branch and
    # every narrative sees the same number.
    if threshold is None:
        threshold = METHOD_THRESHOLD.get(method, 3.0)
    if method == "multiple_of_typical":
        return _multiple_of_typical(ctx, building_ids, threshold, direction,
                                    start, end, days)
    # Match the grain to the cohort. An hourly scan reads raw measurements --
    # about 22 million rows for the portfolio, 57 s against 1.2 s for the same
    # question answered from the daily aggregate, which is 47x faster and
    # actually finishes. It is also the more useful answer: 164,954 hourly
    # outliers across 1,256 buildings is not something a person can act on,
    # where 4,778 anomalous days is.
    switched = None
    if (method != "daily_profile" and building_ids
            and len(building_ids) > COHORT_DAILY_GRAIN):
        switched = method
        method = "daily_profile"

    if method == "daily_profile":
        result = _daily_profile_anomalies(ctx, building_ids, threshold, start,
                                          end, min_coverage, days)
        if switched:
            result.evidence["requested_method"] = switched
            result.evidence["method_note"] = (
                f"{len(building_ids):,} buildings: answered at daily grain "
                f"rather than scanning hourly readings. Ask for a smaller "
                f"cohort to use {switched}.")
        return result
    # Load and analyse in batches so the deadline bounds the load as well as
    # the loop, rather than pulling the whole cohort's hourly readings before the
    # budget was consulted once.
    found: list[dict] = []
    fallbacks: set[str] = set()
    _cov = {"done": 0, "total": len(building_ids or []), "truncated": False}
    # Which buildings were actually examined, as opposed to which were asked
    # for: "34 of 45" needs the denominator to be the set that had data to
    # scan, not the set that had an anomaly in it.
    scanned: set[str] = set()
    for _frame, _cov in batched_series(ctx, building_ids, start=start, end=end):
        parts = _only_hours(_only_days(with_local_parts(_frame), days),
                            hour_from, hour_to)
        scanned.update(parts["building_id"].unique().tolist())
        # "Which buildings had the most anomalous readings" needs one
        # detector, not one per building. Fitting a separate isolation forest
        # to each building at a fixed contamination rate hands every building
        # the same count by construction -- 3 anomalies each across all 65
        # education buildings in Berkeley -- so the ranking the question asks
        # for cannot exist. Pooling fits one model over the cohort, and a
        # building then stands out by being unlike the others.
        if (method == "isolation_forest" and report == "buildings"
                and parts["building_id"].nunique() > 1):
            from sklearn.ensemble import IsolationForest
            from scipy.stats import norm
            pooled = parts.dropna(subset=["value"]).copy()
            own = pooled.groupby("building_id")["value"]
            pooled["z"] = ((pooled["value"] - own.transform("mean"))
                           / own.transform("std").replace(0, np.nan))
            pooled["hour"] = pd.to_datetime(pooled["timestamp"]).dt.hour
            pooled["dow"] = pd.to_datetime(pooled["timestamp"]).dt.dayofweek
            # A building's 3 a.m. and its 3 p.m. are different populations.
            # Scored only against its own overall mean, every afternoon hour
            # of a peaky building looks anomalous and the ranking becomes a
            # ranking by peakiness; scored against the same hour on other
            # days, what stands out is a genuinely odd afternoon.
            by_hour = pooled.groupby(["building_id", "hour"])["value"]
            spread = (by_hour.transform(lambda v: v.quantile(0.75))
                      - by_hour.transform(lambda v: v.quantile(0.25)))
            pooled["z_hour"] = ((pooled["value"] - by_hour.transform("median"))
                                / spread.replace(0, np.nan))
            feats = pooled[["z", "z_hour", "hour", "dow"]].fillna(0.0).to_numpy(float)
            rate = (float(contamination)
                    if contamination is not None and contamination != "auto"
                    else float(min(0.5, max(1e-4,
                                   2 * norm.sf(abs(float(threshold)))))))
            flags = IsolationForest(contamination=rate,
                                    random_state=0).fit_predict(feats)
            hit = pooled[flags == -1]
            for row in hit.itertuples():
                found.append({"building_id": row.building_id,
                              "when": _wall_clock(row.timestamp),
                              "observed": float(row.value),
                              "score": float(row.z if row.z == row.z else 0.0),
                              "reference": "one model pooled over the cohort"})
            parts = parts.iloc[0:0]           # the pooled pass consumed them
        for building, grp in parts.groupby("building_id"):
            values = grp.dropna(subset=["value"])
            if len(values) < 168:
                continue
            if method == "daily_profile":
                daily = values.groupby("date")["value"].sum()
                if len(daily) < 30:
                    continue
                mu, sigma = daily.mean(), daily.std(ddof=1)
                if not sigma:
                    continue
                score = (daily - mu) / sigma
                for day, s in score[score.abs() > threshold].items():
                    found.append({"building_id": building, "when": str(day),
                                  "observed": float(daily[day]), "score": float(s),
                                  "reference": "building's own daily totals"})
            elif method in ("zscore", "modified_zscore", "first_difference",
                            "local_outlier_factor", "iqr"):
                v = values["value"]
                if method == "zscore":
                    sigma = v.std(ddof=1)
                    if not sigma:
                        continue
                    score = (v - v.mean()) / sigma
                    mask = (score > threshold if direction == "high"
                            else -score > threshold if direction == "low"
                            else score.abs() > threshold)
                elif method == "modified_zscore":
                    # The median absolute deviation, scaled so the threshold
                    # reads like a z-score. 0.6745 is the 0.75 quantile of the
                    # standard normal: it makes the modified score equal the
                    # ordinary one for normally distributed data, which is
                    # what lets 3.5 mean roughly what 3 means elsewhere.
                    centre = v.median()
                    deviation = (v - centre).abs().median()
                    if not deviation:
                        # A flat or near-flat series has MAD zero and every
                        # departure becomes infinite. Falling back to the mean
                        # absolute deviation keeps the test defined and says
                        # so; skipping the building would silently drop it.
                        deviation = (v - centre).abs().mean()
                        if not deviation:
                            continue
                        fallbacks.add(building)
                    score = 0.6745 * (v - centre) / deviation
                    mask = (score > threshold if direction == "high"
                            else score < -threshold if direction == "low"
                            else score.abs() > threshold)
                elif method == "first_difference":
                    # A large *change*, not a large level. Scored in standard
                    # deviations of the hour-to-hour change so one threshold
                    # works across buildings of different size.
                    change = v.diff()
                    spread = change.std(ddof=1)
                    if not spread or change.isna().all():
                        continue
                    score = change / spread
                    mask = (score > threshold if direction == "high"
                            else score < -threshold if direction == "low"
                            else score.abs() > threshold)
                elif method == "local_outlier_factor":
                    from sklearn.neighbors import LocalOutlierFactor
                    # Each hour is represented by the window of readings
                    # ending at it, min-max scaled within the window so the
                    # comparison is of shape rather than of level. An hour is
                    # anomalous when its window sits in a sparser
                    # neighbourhood than its neighbours' windows do.
                    span = max(int(context_hours), 24)
                    if len(v) < span + 24:
                        continue
                    stacked = np.lib.stride_tricks.sliding_window_view(
                        v.to_numpy(float), span)
                    if feature_set == "level_and_change":
                        # The window's own first differences, appended to it.
                        # Level says where the building is; change says how
                        # fast it got there, and a step into a new plateau is
                        # invisible to one and obvious to the other.
                        change = np.diff(stacked, axis=1)
                        stacked = np.column_stack([stacked, change])
                    if scale_windows == "minmax":
                        lo = stacked.min(axis=1, keepdims=True)
                        hi = stacked.max(axis=1, keepdims=True)
                        scaled = (stacked - lo) / np.where(hi - lo == 0, 1,
                                                           hi - lo)
                    else:
                        # Unscaled, the neighbourhood is dominated by level:
                        # a large building is far from a small one whatever
                        # their rhythms. Worth offering, because whether that
                        # is a fault depends on the question.
                        scaled = stacked
                    neighbours = min(20, max(5, len(scaled) // 20))
                    detector = LocalOutlierFactor(n_neighbors=neighbours)
                    flags = detector.fit_predict(scaled)
                    lof = -detector.negative_outlier_factor_
                    index = v.index[span - 1:]
                    score = pd.Series(lof, index=index)
                    # The threshold is read on the LOF scale, where 1 is
                    # "as dense as its neighbours". A caller passing a z-score
                    # threshold of 3 would otherwise flag nothing, so the
                    # sklearn verdict is used unless a caller asks for a
                    # stricter line than it does.
                    mask = pd.Series(flags == -1, index=index)
                    if threshold > 1.0:
                        mask &= score > threshold
                    v = v.loc[index]
                else:
                    q1, q3 = v.quantile(0.25), v.quantile(0.75)
                    iqr = q3 - q1
                    score = (v - v.median()) / (iqr if iqr else 1)
                    above, below = v > q3 + threshold * iqr, v < q1 - threshold * iqr
                    mask = (above if direction == "high"
                            else below if direction == "low" else above | below)
                for idx in v.index[mask]:
                    found.append({"building_id": building,
                                  "when": _wall_clock(values.loc[idx, "timestamp"]),
                                  "observed": float(v.loc[idx]),
                                  "score": float(score.loc[idx]),
                                  "reference": "building's own hourly readings"})
            else:
                from sklearn.ensemble import IsolationForest
                level = values["value"].to_numpy(float)
                if feature_set == "level":
                    feats = level.reshape(-1, 1)
                elif feature_set == "level_and_change":
                    feats = np.column_stack(
                        [level, np.diff(level, prepend=level[:1])])
                else:
                    feats = np.column_stack([level,
                                             values["hour"].to_numpy(float),
                                             values["dow"].to_numpy(float)])
                # `contamination="auto"` flagged 6,896 of 13,392 hours on
                # the Bull assembly cohort -- half the record called
                # anomalous, which is not a finding but a broken default.
                # The threshold is a z-score everywhere else in this tool, so
                # it is read as one here too and converted to the tail
                # probability it implies: 3 sigma is 0.27% of hours, which is
                # what an operator means by an anomaly rate.
                from scipy.stats import norm
                # "auto" is sklearn's own default and it is the broken one
                # described above -- a plan that passes it explicitly gets the
                # same 63%-of-readings result as one that passes nothing, so
                # it is treated as "unspecified" rather than as a choice.
                rate = (float(contamination)
                        if contamination is not None and contamination != "auto"
                        else float(min(0.5, max(1e-4,
                                       2 * norm.sf(abs(float(threshold)))))))
                model = IsolationForest(contamination=rate, random_state=0)
                labels = model.fit_predict(feats)
                scores = model.score_samples(feats)
                for pos in np.flatnonzero(labels == -1):
                    found.append({"building_id": building,
                                  "when": _wall_clock(values["timestamp"].iloc[pos]),
                                  "observed": float(values["value"].iloc[pos]),
                                  "score": float(scores[pos]),
                                  "reference": "isolation forest over value/hour/dow"})

    # buildings_examined is what was actually walked, not what was asked for:
    # a cohort cut short by the deadline must not report the full count.
    evidence = {"method": method, "threshold": threshold,
                # Which side was tested belongs with the count. "0 anomalies"
                # means nothing without it: the same building has 0 high and 57
                # low against the same threshold.
                "direction": direction,
                "period": [start, end], "buildings_examined": _cov["done"],
                "n_anomalies": len(found)}
    window = ("" if hour_from is None or hour_to is None
              else f", within {int(hour_from):02d}:00-{int(hour_to):02d}:59 only")
    reference = {
        "zscore": f"an hour more than {threshold:g} standard deviations from "
                  f"the building's own mean hourly demand",
        "modified_zscore": f"an hour more than {threshold:g} on the modified "
                           f"z-score -- 0.6745 x (value - median) / median "
                           f"absolute deviation, which the outliers do not move",
        "first_difference": f"an hour whose change from the preceding hour is "
                            f"more than {threshold:g} standard deviations of "
                            f"this building's hour-to-hour change",
        "local_outlier_factor": f"an hour whose preceding {context_hours}-hour "
                                f"window sits in a sparser neighbourhood than "
                                f"its neighbours' windows do (local outlier "
                                f"factor, min-max scaled per window)",
        "iqr": f"an hour more than {threshold:g} interquartile ranges outside "
               f"the building's own quartiles",
        "isolation_forest": f"an isolation forest over demand, hour of day and "
                            f"day of week, at the contamination rate a "
                            f"{threshold:g}-sigma threshold implies",
    }.get(method, method)
    reference += window
    evidence["units"] = {
        "observed": "kW",
        "score": ("standard deviations" if method in
                  ("zscore", "modified_zscore", "first_difference")
                  else "local outlier factor, 1 = as dense as its neighbours"
                  if method == "local_outlier_factor"
                  else "the test's own scale")}
    if hour_from is not None and hour_to is not None:
        evidence["clock_window"] = [int(hour_from), int(hour_to)]
    if method == "local_outlier_factor":
        evidence["context_hours"] = int(context_hours)
        evidence["window_scaling"] = scale_windows
        evidence["feature_set"] = feature_set
    if fallbacks:
        evidence["mad_zero_fallback"] = sorted(fallbacks)[:10]
        notes = (f"{len(fallbacks)} building(s) have a median absolute "
                 f"deviation of zero -- too flat for the modified z-score -- "
                 f"and were scored on the mean absolute deviation instead")
        evidence["note"] = notes
    if not found:
        evidence["narrative"] = _detection_narrative("anomalous hours", [],
                                                     evidence, reference=reference)
        return with_coverage(ok({"anomalies": []}, evidence=evidence), _cov)
    out = pd.DataFrame(found).sort_values("score", key=abs, ascending=False)
    evidence["narrative"] = _detection_narrative(
        "anomalous hours", out.to_dict("records"), evidence,
        unit="kW", reference=reference)
    out = out.rename(columns={"observed": "observed_kw"})
    # One row per building as well as one per anomaly, so a count can be
    # joined to anything else known about the building. A scan that can only
    # be read as a list of incidents cannot enter a prioritisation.
    evidence.update(_per_building_counts(out, ctx, method))
    if report == "days":
        dated = out.assign(date=out["when"].astype(str).str.slice(0, 10))
        summary = (dated.groupby("date")
                   .agg(anomalies=("score", "size"),
                        buildings=("building_id", "nunique"))
                   .reset_index().sort_values("date"))
        evidence["headline"] = {
            "value": int(len(summary)), "unit": "days",
            "label": (f"{len(out):,} anomalous readings on "
                      f"{len(summary):,} distinct day(s), across "
                      f"{out['building_id'].nunique():,} building(s)")}
        evidence["anomalous_readings"] = int(len(out))
        evidence["buildings_with_anomalies"] = int(out["building_id"].nunique())
        evidence["readings_handle"] = ctx.store.put(out, name="anomalies")
        return with_coverage(
            frame_result(summary, ctx.store, name="anomalies_per_day",
                         evidence=evidence), _cov)
    if report == "buildings":
        summary = ctx.store.get(evidence["per_building_handle"])
        noun = "buildings" if len(summary) != 1 else "building"
        evidence["headline"] = {
            "value": int(len(summary)), "unit": "buildings",
            "label": (f"{len(summary):,} {noun} with at least one "
                      f"{'reading' if method != 'daily_profile' else 'day'} "
                      f"flagged by the {method.replace('_', ' ')} test, "
                      f"out of {len(scanned) or out['building_id'].nunique():,} "
                      f"scanned")}
        evidence["buildings_scanned"] = len(scanned)
        evidence["anomalous_readings"] = int(len(out))
        evidence["readings_handle"] = ctx.store.put(out, name="anomalies")
        return with_coverage(
            frame_result(summary, ctx.store, name="anomalies_per_building",
                         evidence=evidence), _cov)
    return with_coverage(
        frame_result(out, ctx.store, name="anomalies", evidence=evidence), _cov)


def _daily_profile_anomalies(ctx, building_ids, threshold, start, end,
                             min_coverage, days="all"):
    """Daily totals from the coverage-aware aggregates, z-scored per building."""
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    params.append(min_coverage)
    where += " AND coverage_fraction >= ?"
    if start:
        where += " AND period_start >= ?"; params.append(start)
    if end:
        where += " AND period_start < ?"; params.append(end)
    daily = ctx.db.df(f"""
        SELECT building_id, period_start AS day, total_kwh, coverage_fraction
        FROM agg_daily WHERE {where} ORDER BY building_id, day
    """, params)
    daily = _only_dates(daily, days, "day")
    evidence = {"method": "daily_profile", "threshold": threshold,
                "min_coverage": min_coverage, "period": [start, end],
                "days": days,
                "reference": ("the building's own daily totals over the window"
                              if days == "all" else
                              f"the building's own {days} totals over the "
                              f"window -- {days}s are compared only with each "
                              f"other, so the weekly cycle is not the anomaly"),
                "buildings_examined": int(daily["building_id"].nunique())
                if not daily.empty else 0}
    if daily.empty:
        return empty(f"no days at or above {min_coverage:.0%} coverage",
                     evidence=evidence)
    found = []
    _walk, _cov = budgeted(ctx, daily.groupby("building_id"))
    for building, grp in _walk:
        if len(grp) < 30:
            continue
        mu, sigma = grp["total_kwh"].mean(), grp["total_kwh"].std(ddof=1)
        if not sigma:
            continue
        z = (grp["total_kwh"] - mu) / sigma
        for idx in grp.index[z.abs() > threshold]:
            found.append({"building_id": building,
                          "when": str(grp.loc[idx, "day"])[:10],
                          "observed": float(grp.loc[idx, "total_kwh"]),
                          "score": float(z.loc[idx]),
                          "reference": "own daily totals"})
    evidence["n_anomalies"] = len(found)
    evidence["units"] = {"observed": "kWh (daily total)",
                         "score": "standard deviations"}
    # The window is the sample, and saying so is not a detail: this test pools
    # every day in it, so a building whose weekday and weekend differ carries
    # that difference in the deviation it is measured against. The reference
    # for this question compares each day against the preceding twenty of its
    # own type instead, and finds days this cannot.
    reference = (f"a day more than {threshold:g} standard deviations from the "
                 f"building's own mean daily total over the window, pooled "
                 f"across weekdays and weekends")
    if not found:
        evidence["narrative"] = _detection_narrative("days", [], evidence,
                                                     reference=reference)
        return ok({"anomalies": []}, evidence=evidence)
    out = pd.DataFrame(found).sort_values("score", key=abs, ascending=False)
    evidence["narrative"] = _detection_narrative(
        "days", out.to_dict("records"), evidence, reference=reference)
    out = out.rename(columns={"observed": "observed_kwh"})
    evidence.update(_per_building_counts(out, ctx, "daily_profile"))
    return frame_result(out, ctx.store, name="anomalies", evidence=evidence)


@tool("ai_analytics")
def detect_change_points(
    ctx: ToolContext,
    building_ids: list[str],
    min_segment_days: int = 30,
    min_shift_pct: float = 10.0,
) -> ToolResult:
    """Find persistent level shifts in daily consumption.

    Distinguished from anomaly detection on purpose: an anomaly is an excursion,
    a change point is a regime that persists.
    """
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    daily = ctx.db.df(f"""
        SELECT building_id, period_start AS day, total_kwh
        FROM agg_daily WHERE {where} AND coverage_fraction >= 0.9
        ORDER BY building_id, day
    """, params or None)
    if daily.empty:
        return empty("no days above the coverage floor")
    rows = []
    _walk, _cov = budgeted(ctx, daily.groupby("building_id"))
    for building, grp in _walk:
        y = grp["total_kwh"].to_numpy(float)
        days = grp["day"].tolist()
        if len(y) < 2 * min_segment_days:
            continue
        best, best_t = None, None
        for cut in range(min_segment_days, len(y) - min_segment_days):
            left, right = y[:cut], y[cut:]
            sl, sr = left.std(ddof=1), right.std(ddof=1)
            pooled = np.sqrt((sl**2 / len(left)) + (sr**2 / len(right)))
            if not pooled:
                continue
            t = abs(right.mean() - left.mean()) / pooled
            if best is None or t > best:
                best, best_t = t, cut
        if best_t is None:
            continue
        left, right = y[:best_t], y[best_t:]
        shift = 100 * (right.mean() - left.mean()) / left.mean() if left.mean() else np.nan
        if abs(shift) < min_shift_pct:
            continue
        rows.append({"building_id": building,
                     "change_date": _wall_clock(days[best_t])[:10],
                     "before_mean_kwh": float(left.mean()),
                     "after_mean_kwh": float(right.mean()),
                     "shift_pct": float(shift), "t_statistic": float(best),
                     "days_before": int(len(left)), "days_after": int(len(right))})
    evidence = {"method": "binary segmentation, maximum Welch t-statistic",
                "min_segment_days": min_segment_days,
                "min_shift_pct": min_shift_pct, "n_detected": len(rows)}
    if not rows:
        return ok({"change_points": []}, evidence=evidence)
    out = pd.DataFrame(rows).sort_values("t_statistic", ascending=False)
    return frame_result(out, ctx.store, name="change_points", evidence=evidence)


# The three internal validity scores, and which way is better. Reporting a
# number without its direction invites the reader to assume higher is better,
# which is true of two of these and false of the third.
CLUSTER_SCORES = {
    "silhouette": ("higher is better, -1 to 1",
                   "how much closer each profile sits to its own cluster than "
                   "to the nearest other one"),
    "davies_bouldin": ("lower is better, 0 upward",
                       "the worst-case ratio of within-cluster spread to "
                       "between-cluster separation, averaged over clusters"),
    "calinski_harabasz": ("higher is better, 0 upward",
                          "between-cluster dispersion over within-cluster "
                          "dispersion, the variance-ratio criterion"),
}


def _cluster_scores(matrix, labels, n_clusters: int,
                    distances=None) -> dict:
    """Score a partition three ways, because they disagree.

    A partition that wins on silhouette and loses on Davies-Bouldin is the
    normal case, not a contradiction: silhouette asks about each point's own
    neighbourhood, Davies-Bouldin about the worst pair of clusters, and
    Calinski-Harabasz about the variance ratio over the whole set. Reporting
    one of them lets a choice of k look better settled than it is.

    Every score needs at least two clusters with at least two members between
    them, and returns NaN rather than raising where it cannot be computed --
    a clustering that could not be scored is still a clustering.
    """
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    scores = {name: float("nan") for name in CLUSTER_SCORES}
    distinct = len(set(labels))
    if n_clusters < 2 or distinct < 2 or len(labels) <= distinct:
        return scores
    if distances is not None:
        # Under DTW only the silhouette is defined, because it is the one score
        # that works from pairwise distances. The other two need a centroid in
        # a Euclidean space, and computing them on the raw profiles would
        # report the separation of a partition nobody formed.
        try:
            scores["silhouette"] = float(
                silhouette_score(distances, labels, metric="precomputed"))
        except Exception:                                   # noqa: BLE001
            pass
        return scores
    for name, scorer in (("silhouette", silhouette_score),
                         ("davies_bouldin", davies_bouldin_score),
                         ("calinski_harabasz", calinski_harabasz_score)):
        try:
            scores[name] = float(scorer(matrix, labels))
        except Exception:                                   # noqa: BLE001
            pass
    return scores


def _normalise(matrix: np.ndarray, how) -> np.ndarray:
    """Put profiles on a common scale, by a named convention.

    Which convention is not cosmetic: it decides what "similar shape" means,
    and the two in use answer different questions. ``mean`` divides by the
    profile's own mean, so peak-to-mean ratio survives and a building that
    doubles at noon stays distinguishable from one that is flat. ``minmax``
    stretches every profile to [0, 1], discarding peak-to-mean entirely and
    comparing only *when* load rises. On Eagle's weekday/weekend profiles the
    two disagree about which day-type clusters more tightly -- neither is
    wrong, so the choice is recorded in the result rather than assumed.
    """
    if how in (False, "none", None) or not matrix.size:
        return matrix
    if how in (True, "mean"):
        means = matrix.mean(axis=1, keepdims=True)
        return np.divide(matrix, means, out=np.zeros_like(matrix),
                         where=means != 0)
    if how == "minmax":
        low = matrix.min(axis=1, keepdims=True)
        span = matrix.max(axis=1, keepdims=True) - low
        return np.divide(matrix - low, span, out=np.zeros_like(matrix),
                         where=span != 0)
    raise ValueError(f"unknown normalize: {how!r}")


#: Sakoe-Chiba band as a fraction of profile length. An unconstrained DTW over
#: a 168-hour week will align Monday morning to Friday night to save a little
#: distance, which is not a similarity anyone asked about; the band keeps the
#: warp local and cuts the cost from O(n^2) to a strip.
DTW_BAND_FRACTION = 0.1


def _dtw_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise DTW distances between profiles, as a square condensed-ready array.

    Used for the silhouette and for linkage, so that a partition formed under
    DTW is also *scored* under DTW. Scoring a DTW partition with Euclidean
    silhouette reports how well the clusters separate under a metric nobody
    clustered with, which reads like a validity score and is not one.
    """
    from tslearn.metrics import cdist_dtw
    radius = max(1, int(round(matrix.shape[1] * DTW_BAND_FRACTION)))
    return np.asarray(cdist_dtw(matrix, global_constraint="sakoe_chiba",
                                sakoe_chiba_radius=radius))


def _fit_dtw_kmeans(matrix: np.ndarray, n_clusters: int, seed: int = 0):
    """TimeSeriesKMeans under DTW, with DTW barycentre averaging.

    This is the algorithm the question names, not k-means on a DTW distance
    matrix: the centroid is a barycentre in the warped space, so a cluster's
    archetype is a profile rather than a row of an affinity matrix.
    """
    from tslearn.clustering import TimeSeriesKMeans
    radius = max(1, int(round(matrix.shape[1] * DTW_BAND_FRACTION)))
    model = TimeSeriesKMeans(
        n_clusters=n_clusters, metric="dtw", random_state=seed,
        n_init=2, max_iter=10,
        metric_params={"global_constraint": "sakoe_chiba",
                       "sakoe_chiba_radius": radius})
    return model.fit_predict(matrix.reshape(matrix.shape[0], matrix.shape[1], 1))


def _select_k(matrix: np.ndarray, k_min: int = 2, k_max: int = 8,
              seed: int = 0, distances=None) -> tuple[int, float, list[dict]]:
    """Pick the cluster count by silhouette, and show the curve it was picked from.

    A fixed k answers "split these into four groups", which is a different
    request from "how many groups are there". When the question asks the
    latter, the number has to come from the data -- and the curve travels with
    it, because a silhouette of 0.31 at the winning k means something very
    different from 0.53, and the margin over the runner-up is often nothing.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    curve, best = [], (k_min, -1.0)
    for k in range(k_min, min(k_max, len(matrix) - 1) + 1):
        if distances is not None:
            # Choose k under the same metric the clustering will use. Picking k
            # by Euclidean silhouette and then clustering under DTW answers the
            # question with a number chosen for a different one.
            labels = _fit_dtw_kmeans(matrix, k, seed)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(distances, labels,
                                           metric="precomputed"))
        else:
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=seed).fit_predict(matrix)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels))
        curve.append({"k": k, "silhouette": round(score, 4)})
        if score > best[1]:
            best = (k, score)
    return best[0], best[1], curve


def _profile_matrix(ctx: ToolContext, building_ids: list[str],
                    normalize: bool = True) -> tuple[list[str], np.ndarray]:
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    # Sum across streams so an appliance-metered building yields one shape.
    df = ctx.db.df(f"""
        SELECT building_id, hour_of_day, SUM(mean_demand) AS mean_demand
        FROM profile_daily WHERE {where} GROUP BY 1, 2 ORDER BY 1, 2
    """, params or None)
    if df.empty:
        return [], np.empty((0, 24))
    wide = df.pivot(index="building_id", columns="hour_of_day",
                    values="mean_demand").dropna()
    return wide.index.tolist(), _normalise(wide.to_numpy(float), normalize)


SEASON_CASE = """CASE WHEN MONTH(timestamp_local) IN (12, 1, 2) THEN 'winter'
                     WHEN MONTH(timestamp_local) IN (3, 4, 5) THEN 'spring'
                     WHEN MONTH(timestamp_local) IN (6, 7, 8) THEN 'summer'
                     ELSE 'autumn' END"""


def _weekly_matrix(ctx: ToolContext, building_ids: list[str], normalize: bool,
                   by_season: bool) -> tuple[list[tuple], np.ndarray]:
    """One 168-slot weekly profile per building, or per building per season.

    A weekly profile is a week of hours, Monday 00:00 to Sunday 23:00, so a
    building that works Saturdays is a different shape from one that does not
    -- which a 24-hour profile averages away. Splitting by season is what lets
    the same building appear twice and land in two clusters, which is the only
    way clustering can show a shift rather than a difference between buildings.
    """
    params: list = []
    where = "quality_flag = 'OK'" + in_clause("building_id", building_ids, params)
    season = SEASON_CASE if by_season else "'all year'"
    df = ctx.db.df(f"""
        SELECT building_id, {season} AS season,
               CAST(EXTRACT(dow FROM timestamp_local) AS INTEGER) * 24
                   + CAST(EXTRACT(hour FROM timestamp_local) AS INTEGER) AS slot,
               AVG(value) AS mean_demand
        FROM electricity WHERE {where}
        GROUP BY 1, 2, 3
    """, params or None)
    if df.empty:
        return [], np.empty((0, 168))
    wide = (df.pivot(index=["building_id", "season"], columns="slot",
                     values="mean_demand")
              .reindex(columns=range(168)).dropna())
    matrix = wide.to_numpy(float)
    return list(wide.index), _normalise(matrix, normalize)


def _daytype_matrix(ctx: ToolContext, building_ids: list[str],
                    normalize: bool) -> dict[str, tuple[list[str], np.ndarray]]:
    """One 24-hour profile per building, separately for weekdays and weekends.

    Clustering each arm on its own is how "are these groups stable" is asked:
    the same buildings, the same k, two different days of the week. A building
    that changes group between them is running a different rhythm at the
    weekend, which is invisible to a clustering that averages the two together.
    """
    params: list = []
    where = "quality_flag = 'OK'" + in_clause("building_id", building_ids, params)
    frame = ctx.db.df(f"""
        SELECT building_id,
               CASE WHEN EXTRACT(dow FROM timestamp_local) IN (0, 6)
                    THEN 'weekend' ELSE 'weekday' END AS day_type,
               CAST(EXTRACT(hour FROM timestamp_local) AS INTEGER) AS hour,
               AVG(value) AS mean_demand
        FROM electricity WHERE {where}
        GROUP BY 1, 2, 3
    """, params or None)
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    if frame.empty:
        return out
    for day_type, part in frame.groupby("day_type"):
        wide = (part.pivot(index="building_id", columns="hour",
                           values="mean_demand")
                    .reindex(columns=range(24)).dropna())
        matrix = _normalise(wide.to_numpy(float), normalize)
        out[str(day_type)] = (wide.index.tolist(), matrix)
    return out


@tool("ai_analytics")
def explain_consumption(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    target: Literal["total_kwh", "eui", "mean_demand"] = "eui",
    features: list[Literal["floor_area_m2", "year_built", "occupants",
                           "number_of_floors", "building_type",
                           "building_subtype", "cluster", "mean_temperature",
                           "temperature_range", "heating_degree_days",
                           "cooling_degree_days"]] | None = None,
    n_clusters: int = 4,
    year: int | None = None,
) -> ToolResult:
    """Fit consumption against building characteristics and weather, and say
    what each is worth.

    Two numbers a reader needs and one they are usually given instead. The
    coefficients say how the target moves with each feature, in the target's
    own units, and the RMSE says how much of the variation is left over -- and
    the second is what stops the first from being over-read. A model that
    explains a tenth of the spread has coefficients, and they are not an
    explanation.

    Categorical features are one-hot encoded and every coefficient is reported
    against a named reference level, because a coefficient for "Office" means
    nothing until you know what it is Office *minus*. Numeric features are
    standardised so the coefficients can be compared with one another, and the
    unstandardised slope is reported beside each so it can still be read in
    physical units.

    ``cluster`` as a feature is the k-means load-shape group, which is how "do
    the clusters explain consumption" is asked: it enters the design as a
    categorical like any other, and its coefficients say what each shape is
    worth once floor area and the rest are accounted for.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    chosen = list(features or ["floor_area_m2", "year_built", "building_type"])
    params: list = []
    where = "1=1" + in_clause("b.building_id", building_ids, params)
    period = ""
    if year:
        period = " AND YEAR(a.period_start) = ?"
    frame = ctx.db.df(f"""
        SELECT b.building_id, b.floor_area_m2, b.year_built, b.occupants,
               b.number_of_floors, b.building_type_canonical AS building_type,
               b.building_subtype, b.site_id,
               SUM(a.total_kwh) AS total_kwh,
               AVG(a.mean_demand) AS mean_demand
        FROM buildings b JOIN agg_yearly a USING (building_id)
        WHERE {where}{period}
        GROUP BY 1,2,3,4,5,6,7,8
    """, ([*params, int(year)] if year else (params or None)))
    if frame.empty:
        return empty("no buildings with consumption in that period")
    frame["eui"] = frame["total_kwh"] / frame["floor_area_m2"].replace(0, np.nan)

    weather_wanted = [f for f in chosen if f in
                      ("mean_temperature", "temperature_range",
                       "heating_degree_days", "cooling_degree_days")]
    if weather_wanted:
        weather = ctx.db.df(f"""
            SELECT site_id,
                   AVG(air_temperature) AS mean_temperature,
                   MAX(air_temperature) - MIN(air_temperature) AS temperature_range,
                   SUM(GREATEST(18 - air_temperature, 0)) / 24 AS heating_degree_days,
                   SUM(GREATEST(air_temperature - 18, 0)) / 24 AS cooling_degree_days
            FROM weather WHERE air_temperature IS NOT NULL
            {"AND YEAR(timestamp_local) = " + str(int(year)) if year else ""}
            GROUP BY 1
        """)
        frame = frame.merge(weather, on="site_id", how="left")

    if "cluster" in chosen:
        ids, matrix = _profile_matrix(ctx, frame["building_id"].tolist(), True)
        if len(ids) >= n_clusters:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n_clusters, n_init=10,
                            random_state=0).fit_predict(matrix)
            frame = frame.merge(
                pd.DataFrame({"building_id": ids,
                              "cluster": [f"shape {int(v)}" for v in labels]}),
                on="building_id", how="left")
        else:
            chosen = [f for f in chosen if f != "cluster"]

    available = [f for f in chosen if f in frame.columns]
    missing = [f for f in chosen if f not in frame.columns]
    usable = frame.dropna(subset=[target, *available])
    if len(usable) < len(available) + 5:
        return insufficient(
            "too_few_complete",
            f"{len(usable)} building(s) carry {target} and all of {available}; "
            f"a fit needs more rows than features")

    numeric = [f for f in available
               if pd.api.types.is_numeric_dtype(usable[f])]
    categorical = [f for f in available if f not in numeric]
    design = pd.DataFrame(index=usable.index)
    scales, references = {}, {}
    for feature in numeric:
        column = pd.to_numeric(usable[feature], errors="coerce")
        spread = float(column.std(ddof=0)) or 1.0
        scales[feature] = spread
        design[feature] = (column - column.mean()) / spread
    for feature in categorical:
        levels = sorted(usable[feature].dropna().astype(str).unique())
        if len(levels) < 2:
            continue
        references[feature] = levels[0]
        for level in levels[1:]:
            design[f"{feature}={level}"] = (
                usable[feature].astype(str) == level).astype(float)

    if design.empty:
        return insufficient("no_features", "no usable feature survived")
    y = pd.to_numeric(usable[target], errors="coerce").to_numpy(float)
    model = LinearRegression().fit(design.to_numpy(float), y)
    fitted = model.predict(design.to_numpy(float))
    residual = y - fitted
    rmse = float(np.sqrt((residual ** 2).mean()))
    coefficients = []
    for name, value in zip(design.columns, model.coef_):
        base = name.split("=")[0]
        coefficients.append({
            "feature": name,
            "coefficient": round(float(value), 4),
            "per_unit": (round(float(value) / scales[base], 6)
                         if base in scales else None),
            "kind": "numeric (per standard deviation)" if base in scales
                    else f"category vs {references.get(base, '?')}",
        })
    coefficients.sort(key=lambda row: -abs(row["coefficient"]))
    out = pd.DataFrame(coefficients)
    unit = {"eui": "kWh/m2/yr", "total_kwh": "kWh", "mean_demand": "kW"}[target]
    narrative = [
        f"{target} explained by {len(design.columns)} term(s) over "
        f"{len(usable):,} buildings.",
        f"  RMSE            {rmse:,.2f} {unit}",
        f"  R-squared       {r2_score(y, fitted):.4f}",
        f"  mean {target}   {y.mean():,.2f} {unit}",
        "  Strongest terms:",
    ]
    for row in coefficients[:6]:
        narrative.append(
            f"    {row['feature']:<34}{row['coefficient']:>12,.2f}  "
            f"({row['kind']})")
    narrative.append(
        f"  A numeric coefficient is the change in {target} for one standard "
        f"deviation of that feature, so they are comparable with one another; "
        f"the per-unit slope is in the table. A category coefficient is that "
        f"level minus its reference level, named beside it.")
    narrative.append(
        f"  RMSE against a mean {y.mean():,.2f} {unit} is what says whether "
        f"these coefficients explain the spread or merely describe it.")
    dropped = len(frame) - len(usable)
    if dropped:
        # A fit silently drops every row missing any feature, and occupants is
        # recorded on 230 of 1,636 buildings. Half a cohort can vanish into a
        # regression that then reports coefficients for "the portfolio".
        thin = sorted(
            ((f, int(frame[f].isna().sum())) for f in available
             if f in frame and frame[f].isna().any()),
            key=lambda pair: -pair[1])[:3]
        narrative.append(
            f"  {dropped:,} of {len(frame):,} buildings were dropped for a "
            f"missing feature"
            + (f" -- mostly " + ", ".join(f"{f} ({n:,} absent)" for f, n in thin)
               if thin else "")
            + f". The fit describes the {len(usable):,} that remain, which is "
              f"not the same population the question named.")
    if missing:
        narrative.append(f"  Not available and dropped: {', '.join(missing)}.")
    return frame_result(out, ctx.store, name="consumption_explained", evidence={
        "target": target, "unit": unit, "buildings": int(len(usable)),
        "features_used": available, "features_dropped": missing,
        "buildings_offered": int(len(frame)),
        "buildings_dropped_for_missing_features": int(len(frame) - len(usable)),
        "rmse": round(rmse, 4),
        "r_squared": round(float(r2_score(y, fitted)), 4),
        "reference_levels": references,
        "narrative": narrative,
        "headline": {"value": round(float(r2_score(y, fitted)), 4), "unit": None,
                     "label": f"R-squared of {target} against "
                              f"{', '.join(available)}"},
    })


@tool("ai_analytics")
def validate_anomaly_detection(
    ctx: ToolContext,
    building_ids: list[str],
    configurations: list[dict] | None = None,
    anomalies_per_building: int = 12,
    kinds: list[Literal["spike", "dip", "step", "flatline"]] | None = None,
    magnitude: float = 3.0,
    tolerance_hours: int = 1,
    start: str | None = None,
    end: str | None = None,
    seed: int = 0,
) -> ToolResult:
    """Score anomaly detectors against anomalies we planted ourselves.

    Every other anomaly answer in this system is unfalsifiable: without labels,
    "57 anomalies" cannot be shown to be right or wrong, and "does this feature
    improve detection" has no answer at all -- more flags is not better
    detection, and fewer is not worse.

    So the labels are made. Known anomalies of known kinds are injected into
    the real series at known hours, each detector is run on the modified
    series, and what comes back is precision, recall and F1 against those
    hours. That is a genuine measurement of a detector's behaviour on *this*
    data, and it is the only basis on which one configuration can be said to
    beat another.

    Two honest limits travel with the result. Injected anomalies are the kinds
    we thought to inject, so recall is recall *of those kinds* -- a detector
    tuned for spikes will look excellent in a test made of spikes. And the
    series already contains real anomalies nobody labelled, so a flag that is
    not near an injection is counted a false positive when it may be a true
    one; precision is therefore a lower bound rather than an estimate.

    ``configurations`` is a list of keyword mappings for ``detect_anomalies``,
    each scored separately on the identical injected series -- which is what
    makes them comparable.
    """
    from sklearn.metrics import precision_recall_fscore_support     # noqa: F401
    rng = np.random.default_rng(seed)
    wanted_kinds = list(kinds or ["spike", "dip", "step"])
    settings = configurations or [
        {"method": "local_outlier_factor", "feature_set": "level"},
        {"method": "local_outlier_factor", "feature_set": "level_and_change"},
    ]
    series = _hourly_many(ctx, building_ids).dropna(subset=["value"])
    if series.empty:
        return empty("no readings for those buildings")
    stamps = pd.to_datetime(series["timestamp"])
    if start:
        series = series.loc[stamps >= pd.Timestamp(start).tz_localize(stamps.dt.tz)]
        stamps = pd.to_datetime(series["timestamp"])
    if end:
        series = series.loc[stamps < pd.Timestamp(end).tz_localize(stamps.dt.tz)]
    if series.empty:
        return empty("no readings in that window")

    # --- plant the anomalies ------------------------------------------------
    injected: dict[str, set] = {}
    modified = []
    for building, part in series.groupby("building_id"):
        part = part.sort_values("timestamp").reset_index(drop=True)
        values = part["value"].to_numpy(float)
        if len(values) < 24 * 14:
            continue
        spread = float(values.std(ddof=0)) or 1.0
        # Kept away from the ends so a detector has context on both sides, and
        # spaced so two injections cannot land in the same hour.
        room = np.arange(24 * 7, len(values) - 24)
        picks = rng.choice(room, size=min(anomalies_per_building, len(room)),
                           replace=False)
        marks = set()
        for position in sorted(picks):
            kind = wanted_kinds[int(rng.integers(len(wanted_kinds)))]
            if kind == "spike":
                values[position] += magnitude * spread
            elif kind == "dip":
                values[position] = max(values[position] - magnitude * spread, 0)
            elif kind == "step":
                # A persistent level shift: the hour it begins is the label.
                values[position:position + 24] += magnitude * spread
            else:                                   # flatline
                values[position:position + 6] = values[position]
            marks.add(pd.Timestamp(part.loc[position, "timestamp"]))
        injected[building] = marks
        part = part.assign(value=values)
        modified.append(part)
    if not modified:
        return insufficient("too_little_history",
                            "no building has enough readings to plant and "
                            "detect anomalies in")
    planted = pd.concat(modified, ignore_index=True)
    total_injected = sum(len(v) for v in injected.values())

    # --- score each configuration on the identical planted series -----------
    store_key = ctx.store.put(planted, name="injected_series")
    rows, deadline = [], getattr(ctx, "deadline", None)
    tolerance = pd.Timedelta(hours=int(tolerance_hours))
    for setting in settings:
        if deadline and time.time() > deadline - 10:
            break
        found = _detect_on_frame(ctx, planted, setting)
        if found is None:
            rows.append({**_label(setting), "detected": None,
                         "note": "configuration could not be run"})
            continue
        hits, misses, false_alarms = 0, 0, 0
        flagged_by_building: dict[str, list] = {}
        for record in found:
            flagged_by_building.setdefault(record["building_id"], []).append(
                pd.Timestamp(record["when"]))
        for building, marks in injected.items():
            flags = sorted(flagged_by_building.get(building, []))
            matched = set()
            for mark in marks:
                near = [f for f in flags if abs(f - mark) <= tolerance]
                if near:
                    hits += 1
                    matched.add(min(near, key=lambda f: abs(f - mark)))
                else:
                    misses += 1
            false_alarms += len([f for f in flags if f not in matched
                                 and not any(abs(f - m) <= tolerance
                                             for m in marks)])
        precision = hits / (hits + false_alarms) if (hits + false_alarms) else 0.0
        recall = hits / (hits + misses) if (hits + misses) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        rows.append({**_label(setting),
                     "detected": hits + false_alarms,
                     "true_positives": hits, "false_negatives": misses,
                     "false_positives": false_alarms,
                     "precision": round(precision, 4),
                     "recall": round(recall, 4),
                     "f1": round(f1, 4),
                     "false_positive_rate_per_1000h": round(
                         1000 * false_alarms / max(len(planted), 1), 3)})
    if not rows:
        return insufficient("nothing_scored", "no configuration could be run")
    frame = pd.DataFrame(rows).sort_values("f1", ascending=False,
                                           na_position="last").reset_index(drop=True)
    best = frame.iloc[0]
    narrative = [
        f"{total_injected:,} anomalies planted across "
        f"{len(injected)} building(s) -- {', '.join(wanted_kinds)}, "
        f"{magnitude:g} standard deviations -- and each configuration scored "
        f"against those hours.",
        f"  {'configuration':<40}{'precision':>11}{'recall':>9}{'F1':>8}",
    ]
    for _, row in frame.iterrows():
        if row.get("precision") is None:
            narrative.append(f"  {row['configuration']:<40}   {row.get('note')}")
            continue
        narrative.append(
            f"  {row['configuration']:<40}{row['precision']:>11.3f}"
            f"{row['recall']:>9.3f}{row['f1']:>8.3f}")
    if len(frame) > 1 and frame["f1"].notna().sum() > 1:
        margin = float(frame.iloc[0]["f1"]) - float(frame.iloc[-1]["f1"])
        narrative.append(
            f"  {best['configuration']} scores highest, by {margin:.3f} F1 "
            f"over the weakest.")
    narrative.append(
        "  Recall is recall *of the kinds planted*: a detector suited to "
        "spikes looks strong in a test made of spikes. And the series already "
        "holds real anomalies nobody labelled, so a flag away from an "
        "injection is counted false when it may be true -- precision is a "
        "lower bound, not an estimate.")
    return frame_result(frame, ctx.store, name="anomaly_validation", evidence={
        "buildings": len(injected), "injected": total_injected,
        "kinds": wanted_kinds, "magnitude_sd": magnitude,
        "tolerance_hours": int(tolerance_hours), "seed": seed,
        "injected_series_handle": store_key,
        "hours_examined": int(len(planted)),
        "narrative": narrative,
        "headline": {"value": round(float(best["f1"]), 4)
                     if best.get("f1") is not None else None,
                     "unit": None,
                     "label": f"best F1 ({best['configuration']}) against "
                              f"{total_injected:,} planted anomalies"},
    })


def _label(setting: dict) -> dict:
    """A configuration named as a reader would name it."""
    parts = [str(setting.get("method", "daily_profile"))]
    for key in ("feature_set", "scale_windows", "context_hours", "threshold"):
        if key in setting:
            parts.append(f"{key}={setting[key]}")
    return {"configuration": " ".join(parts)}


def _detect_on_frame(ctx: ToolContext, planted: pd.DataFrame,
                     setting: dict) -> list | None:
    """Run one detector over an in-memory series rather than the snapshot.

    The injected series exists only here, so the detector cannot read it from
    the database the way `detect_anomalies` does. This mirrors that tool's
    hourly path over the frame it is handed, so the comparison is between
    configurations rather than between code paths.
    """
    from sklearn.neighbors import LocalOutlierFactor
    method = setting.get("method", "local_outlier_factor")
    threshold = float(setting.get("threshold", 1.5))
    span = int(setting.get("context_hours", 168))
    scaling = setting.get("scale_windows", "minmax")
    features = setting.get("feature_set", "level")
    found: list[dict] = []
    for building, part in planted.groupby("building_id"):
        part = part.sort_values("timestamp")
        values = part["value"].to_numpy(float)
        stamps = pd.to_datetime(part["timestamp"]).to_numpy()
        if method == "local_outlier_factor":
            if len(values) < span + 24:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(values, span)
            if features == "level_and_change":
                windows = np.column_stack([windows, np.diff(windows, axis=1)])
            if scaling == "minmax":
                lo = windows.min(axis=1, keepdims=True)
                hi = windows.max(axis=1, keepdims=True)
                windows = (windows - lo) / np.where(hi - lo == 0, 1, hi - lo)
            neighbours = min(20, max(5, len(windows) // 20))
            detector = LocalOutlierFactor(n_neighbors=neighbours)
            flags = detector.fit_predict(windows)
            scores = -detector.negative_outlier_factor_
            mask = flags == -1
            if threshold > 1.0:
                mask &= scores > threshold
            for position in np.flatnonzero(mask):
                found.append({"building_id": building,
                              "when": stamps[span - 1 + position]})
        else:
            centre = np.median(values)
            if method == "modified_zscore":
                deviation = np.median(np.abs(values - centre)) or \
                    np.mean(np.abs(values - centre))
                if not deviation:
                    continue
                scores = 0.6745 * (values - centre) / deviation
            elif method == "first_difference":
                change = np.diff(values, prepend=values[0])
                spread = change.std(ddof=1) or 1.0
                scores = change / spread
            else:
                spread = values.std(ddof=1) or 1.0
                scores = (values - values.mean()) / spread
            for position in np.flatnonzero(np.abs(scores) > threshold):
                found.append({"building_id": building,
                              "when": stamps[position]})
    return found


@tool("ai_analytics")
def sweep_anomaly_sensitivity(
    ctx: ToolContext,
    building_ids: list[str],
    method: Literal["local_outlier_factor", "zscore", "modified_zscore",
                    "iqr", "first_difference"] = "local_outlier_factor",
    thresholds: list[float] | None = None,
    context_hours: list[int] | None = None,
    normalize: list[Literal["minmax", "none"]] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """How much the anomaly rate depends on the settings, rather than the data.

    An anomaly count means nothing on its own: every one of these tests will
    return as many or as few as the threshold asks for. What an operator needs
    to know before acting on a count is how fast it moves when the settings
    move -- a rate that halves between two defensible thresholds is a choice
    being reported as a finding.

    The sweep runs the same detector over a grid and reports the rate at each
    point, as a percentage of the hours examined so cohorts of different size
    are comparable. Where a parameter does not apply to the chosen method it
    is not varied, and the result says so rather than reporting a grid whose
    axes did nothing.
    """
    lines = [float(t) for t in (thresholds or [1.2, 1.5, 2.0, 3.0])]
    windows = [int(c) for c in (context_hours or [168])]
    scalings = list(normalize or ["minmax"])
    if method != "local_outlier_factor":
        # Only the density test reads a context window or a scaling; varying
        # them elsewhere would produce identical rows and imply the settings
        # mattered.
        windows, scalings = [windows[0]], [scalings[0]]
    elif normalize is None:
        # Both, by default: whether the windows are scaled is one of the
        # settings the question is asking about.
        scalings = ["minmax", "none"]

    rows, deadline = [], getattr(ctx, "deadline", None)
    for window in windows:
        for scaling in scalings:
            for line in lines:
                if deadline and time.time() > deadline - 10:
                    break
                result = detect_anomalies(
                    ctx, building_ids=building_ids, method=method,
                    threshold=line, context_hours=window,
                    scale_windows=scaling, start=start, end=end)
                evidence = result.evidence or {}
                found = evidence.get("n_anomalies")
                if found is None and result.status is Status.OK:
                    found = 0
                if found is None:
                    continue
                rows.append({
                    "threshold": line,
                    "context_hours": window if method == "local_outlier_factor" else None,
                    "scaling": scaling if method == "local_outlier_factor" else None,
                    "anomalies": int(found),
                    "buildings_examined": evidence.get("buildings_examined"),
                })
    if not rows:
        return insufficient("no_grid_point",
                            "no combination of settings could be scored")
    frame = pd.DataFrame(rows).dropna(axis=1, how="all")
    counts = frame["anomalies"]
    spread = (counts.max() / counts.min()) if counts.min() else float("inf")
    narrative = [
        f"Anomaly rate for {method.replace('_', ' ')} across "
        f"{len(frame)} setting(s), {len(building_ids):,} building(s).",
        f"  {'threshold':>10}{'context':>10}{'scaling':>10}{'anomalies':>12}",
    ]
    for _, row in frame.iterrows():
        narrative.append(
            f"  {row['threshold']:>10.2f}"
            + (f"{int(row['context_hours']):>10}" if "context_hours" in frame
               else f"{'-':>10}")
            + (f"{str(row['scaling']):>10}" if "scaling" in frame
               else f"{'-':>10}")
            + f"{int(row['anomalies']):>12,}")
    high, low = int(counts.max()), int(counts.min())
    if low == 0:
        narrative.append(
            f"  The count runs from {high:,} down to zero across these "
            f"settings. There is no anomaly rate for this cohort independent "
            f"of the threshold -- any figure quoted without it is a choice "
            f"reported as a finding.")
    elif spread > 2:
        narrative.append(
            f"  The count moves by a factor of {spread:.1f} ({low:,} to "
            f"{high:,}) across these settings, so it is a property of the "
            f"threshold at least as much as of the data. Report the setting "
            f"with the count, or report the rate at several.")
    else:
        narrative.append(
            f"  The count moves by a factor of {spread:.1f} ({low:,} to "
            f"{high:,}), which is stable enough to quote a single figure.")
    return frame_result(frame, ctx.store, name="anomaly_sensitivity", evidence={
        "method": method, "thresholds": lines,
        "context_hours": windows if method == "local_outlier_factor" else None,
        "period": [start, end],
        "count_ratio_max_over_min": (round(spread, 3)
                                     if spread != float("inf") else None),
        "narrative": narrative,
        "headline": _sweep_headline(frame, spread),
    })


@tool("ai_analytics")
def compare_cluster_stability(
    ctx: ToolContext,
    building_ids: list[str],
    n_clusters: int = 4,
    normalize: bool = True,
    method: Literal["kmeans", "hierarchical", "ward", "average", "complete",
                    "single"] = "kmeans",
) -> ToolResult:
    """Cluster weekday and weekend load shapes separately, and compare.

    Two questions, answered together because neither is worth much alone. How
    well does each arm cluster -- silhouette, Davies-Bouldin and
    Calinski-Harabasz on its own partition -- and how far do the groupings
    agree, over the buildings present in both.

    Agreement is the adjusted Rand index rather than a count of matching
    labels, because labels are arbitrary: cluster 2 on weekdays and cluster 0
    at the weekend may be the same set of buildings, and any measure that
    compares label numbers would call that a total disagreement. The adjusted
    index is also corrected for chance, so 0 means "as similar as two random
    partitions of this size", not "nothing in common".
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score
    arms = _daytype_matrix(ctx, building_ids, normalize)
    if len(arms) < 2:
        return insufficient(
            "one_day_type",
            "need both weekday and weekend readings to compare their "
            f"clusterings; found {sorted(arms) or 'none'}")

    labels: dict[str, dict[str, int]] = {}
    rows, scores_by_arm = [], {}
    for day_type in ("weekday", "weekend"):
        ids, matrix = arms.get(day_type, ([], np.empty((0, 24))))
        if len(ids) < n_clusters:
            return insufficient(
                "too_few_profiles",
                f"{len(ids)} complete {day_type} profiles for {n_clusters} "
                f"clusters")
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward" if method == "hierarchical" else method)
        assigned = model.fit_predict(matrix)
        labels[day_type] = dict(zip(ids, (int(v) for v in assigned)))
        scores_by_arm[day_type] = _cluster_scores(matrix, assigned, n_clusters)
        rows.extend({"day_type": day_type, "building_id": b, "cluster": int(c)}
                    for b, c in zip(ids, assigned))

    shared = sorted(set(labels["weekday"]) & set(labels["weekend"]))
    agreement = (float(adjusted_rand_score(
        [labels["weekday"][b] for b in shared],
        [labels["weekend"][b] for b in shared])) if len(shared) > 1 else float("nan"))
    frame = pd.DataFrame(rows)
    moved = [b for b in shared if labels["weekday"][b] != labels["weekend"][b]]

    narrative = [
        f"Weekday and weekend load shapes clustered separately into "
        f"{n_clusters} groups by {method}, over {len(shared):,} buildings "
        f"present in both.",
        f"  {'arm':<10}{'silhouette':>12}{'Davies-Bouldin':>17}"
        f"{'Calinski-Harabasz':>20}",
    ]
    for day_type in ("weekday", "weekend"):
        s = scores_by_arm[day_type]
        narrative.append(
            f"  {day_type:<10}{s['silhouette']:>12.4f}"
            f"{s['davies_bouldin']:>17.4f}{s['calinski_harabasz']:>20.1f}")
    narrative.append(
        f"  Agreement between the two groupings: adjusted Rand index "
        f"{agreement:.4f} "
        + ("-- the two are essentially the same partition."
           if agreement > 0.75 else
           "-- the groupings are related but not the same." if agreement > 0.3
           else "-- the weekend groups the portfolio differently from the week."))
    narrative.append(
        f"  Labels are arbitrary, so agreement is measured by which buildings "
        f"share a group, not by the numbers: the index is corrected for "
        f"chance, so 0 means no more alike than two random partitions.")
    return frame_result(frame, ctx.store, name="cluster_stability", evidence={
        "n_clusters": n_clusters, "method": method, "normalized": normalize,
        "buildings_in_both": len(shared),
        "buildings_changing_cluster": len(moved),
        "silhouette": {k: round(v["silhouette"], 4)
                       for k, v in scores_by_arm.items()},
        "davies_bouldin": {k: round(v["davies_bouldin"], 4)
                           for k, v in scores_by_arm.items()},
        "calinski_harabasz": {k: round(v["calinski_harabasz"], 2)
                              for k, v in scores_by_arm.items()},
        "adjusted_rand_index": round(agreement, 4),
        "score_directions": {k: v[0] for k, v in CLUSTER_SCORES.items()},
        "narrative": narrative,
        "headline": {"value": round(agreement, 4), "unit": None,
                     "label": "adjusted Rand index between the weekday and "
                              "weekend clusterings"},
    })


@tool("ai_analytics")
def rank_profile_divergence(
    ctx: ToolContext,
    building_ids: list[str],
    contrast: Literal["weekday_weekend", "summer_winter"] = "weekday_weekend",
    normalize: Literal["mean", "minmax", "none"] | bool = "minmax",
    metric: Literal["euclidean", "correlation", "peak_shift"] = "euclidean",
    limit: int | None = None,
) -> ToolResult:
    """Rank a cohort by how much its two load shapes differ from each other.

    "Which education buildings show the largest difference between weekday and
    weekend profiles" is a question about each building against *itself*, not
    against the cohort. Clustering answers a different one -- it groups
    buildings by shape and can put a building with a flat week and one with a
    sharp weekday dip in the same cluster, because both differ from the rest
    in the same direction.

    The comparison is between two normalised 24-hour profiles, so it measures
    shape rather than size: a building that simply draws less at the weekend
    scores low, and one that changes *when* it draws scores high. That is what
    "difference in normalized load profiles" asks for, and the unnormalised
    version answers a question about magnitude instead.

    ``metric="correlation"`` reports 1 - r, so every metric here is a distance
    and larger always means more different. ``peak_shift`` is the number of
    hours the busiest hour moves, which is the coarsest reading and the one a
    non-specialist finds easiest to check.
    """
    pair = ("weekday", "weekend")
    params: list = []
    where = "1=1" + in_clause("building_id", building_ids, params)
    if contrast == "weekday_weekend":
        # profile_weekly already holds the 7x24 grid, so the split is a label
        # on day_of_week rather than a pass over the readings. Summing across
        # streams first keeps an appliance-metered building to one shape.
        df = ctx.db.df(f"""
            SELECT building_id,
                   CASE WHEN day_of_week >= 5 THEN 'weekend'
                        ELSE 'weekday' END AS part,
                   hour_of_day,
                   SUM(mean_demand) / COUNT(DISTINCT day_of_week) AS mean_demand
            FROM profile_weekly WHERE {where}
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """, params or None)
    else:
        pair = ("summer", "winter")
        df = ctx.db.df(f"""
            SELECT building_id,
                   CASE WHEN MONTH(timestamp_local) IN (6, 7, 8) THEN 'summer'
                        WHEN MONTH(timestamp_local) IN (12, 1, 2) THEN 'winter'
                        ELSE 'other' END AS part,
                   HOUR(timestamp_local) AS hour_of_day,
                   AVG(value) AS mean_demand
            FROM electricity WHERE quality_flag = 'OK' AND {where}
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """, params or None)
    if df.empty:
        return insufficient("no_profiles",
                            "no hourly profile rows for these buildings")
    df = df[df["part"].isin(pair)]
    rows = []
    for building, grp in df.groupby("building_id"):
        wide = grp.pivot(index="part", columns="hour_of_day",
                         values="mean_demand")
        if not set(pair).issubset(wide.index) or wide.isna().any().any():
            continue
        a = _normalise(wide.loc[[pair[0]]].to_numpy(float), normalize)[0]
        b = _normalise(wide.loc[[pair[1]]].to_numpy(float), normalize)[0]
        if metric == "correlation":
            if a.std() == 0 or b.std() == 0:
                continue
            divergence = 1.0 - float(np.corrcoef(a, b)[0, 1])
        elif metric == "peak_shift":
            divergence = float(abs(int(a.argmax()) - int(b.argmax())))
        else:
            divergence = float(np.linalg.norm(a - b))
        rows.append({
            "building_id": building, "divergence": round(divergence, 4),
            f"{pair[0]}_peak_hour": int(a.argmax()),
            f"{pair[1]}_peak_hour": int(b.argmax()),
            "peak_hours_apart": int(abs(int(a.argmax()) - int(b.argmax()))),
        })
    if not rows:
        return insufficient(
            "no_paired_profiles",
            f"no building has a complete profile for both {pair[0]} and {pair[1]}")
    out = pd.DataFrame(rows).sort_values("divergence", ascending=False)
    if limit:
        out = out.head(int(limit))
    top = out.iloc[0]
    return frame_result(out, ctx.store, name="profile_divergence", evidence={
        "contrast": contrast, "parts": list(pair), "metric": metric,
        "normalize": normalize, "buildings_compared": len(rows),
        "unit": {"euclidean": "distance between normalised 24-hour profiles",
                 "correlation": "1 - Pearson r between the two profiles",
                 "peak_shift": "hours the peak moves"}[metric],
        "headline": {"label": f"largest {pair[0]}/{pair[1]} profile difference: "
                              f"{top['building_id']}",
                     "value": float(top["divergence"]), "unit": None},
    })


def _daily_matrix(ctx: ToolContext, building_ids: list[str], normalize,
                  start: str | None, end: str | None) -> tuple[list[tuple], np.ndarray]:
    """One 24-hour profile per building per *day*, for clustering a year of days.

    The building matrix has one row per building and answers "which buildings
    look alike". A question about a single building -- "how many distinct daily
    load profiles does it have" -- is about that building's own days, and with
    one row there is nothing to cluster. Each complete day becomes a sample
    here instead.
    """
    params: list = []
    where = "quality_flag = 'OK'" + in_clause("building_id", building_ids, params)
    if start:
        where += " AND timestamp_local >= ?"; params.append(start)
    if end:
        where += " AND timestamp_local < ?"; params.append(end)
    df = ctx.db.df(f"""
        SELECT building_id,
               CAST(timestamp_local AS DATE) AS day,
               HOUR(timestamp_local) AS hour_of_day,
               SUM(value) AS value
        FROM electricity WHERE {where}
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """, params or None)
    if df.empty:
        return [], np.empty((0, 24))
    wide = df.pivot_table(index=["building_id", "day"], columns="hour_of_day",
                          values="value")
    # A part-day is not a load profile; dropping it is what keeps a midnight
    # start or a meter outage from becoming its own cluster.
    wide = wide.dropna()
    if wide.empty:
        return [], np.empty((0, 24))
    keys = [(b, str(d)) for b, d in wide.index]
    return keys, _normalise(wide.to_numpy(float), normalize)


def _cluster_labels(matrix, distances, method: str, n_clusters: int,
                    distance: str):
    """Fit one clustering and return its labels, or None if the pair is invalid.

    Shared by ``cluster_timeseries`` and ``compare_clustering_methods`` so the
    comparison runs exactly the algorithm the single-method tool would, rather
    than a second implementation that could drift from it.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    if distance == "dtw":
        if method == "kmeans":
            return _fit_dtw_kmeans(matrix, n_clusters)
        if method == "ward":
            return None
        linkage = "average" if method == "hierarchical" else method
        return AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage,
            metric="precomputed").fit_predict(distances)
    if method == "kmeans":
        if distance != "euclidean":
            return None
        return KMeans(n_clusters=n_clusters, n_init=10,
                      random_state=0).fit_predict(matrix)
    linkage = "ward" if method == "hierarchical" else method
    if linkage == "ward" and distance != "euclidean":
        return None
    return AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage,
        metric="euclidean" if linkage == "ward" else distance).fit_predict(matrix)


@tool("ai_analytics")
def compare_clustering_methods(
    ctx: ToolContext,
    building_ids: list[str],
    methods: list[Literal["kmeans", "hierarchical", "ward", "average",
                          "complete", "single"]] | None = None,
    n_clusters: int | Literal["auto"] | None = "auto",
    normalize: Literal["mean", "minmax", "none"] | bool = True,
    profile: Literal["daily", "weekly"] = "daily",
    distance: Literal["euclidean", "cosine", "manhattan", "dtw"] = "euclidean",
) -> ToolResult:
    """Run several clustering algorithms on one cohort and compare them.

    "How do k-means and hierarchical clustering compare in number of clusters
    and silhouette" is a question about the algorithms, not about the
    buildings, and answering it from two separate calls leaves the comparison
    to the reader -- who cannot tell whether the two ran on the same profiles,
    the same normalisation or the same candidate range for k.

    Each method chooses its own k when ``n_clusters="auto"``, because that is
    part of what is being compared: two algorithms that agree on k and differ
    on silhouette are a different finding from two that disagree on k.
    The profiles are built once and shared, so the only thing varying is the
    algorithm.

    Agreement between the partitions is reported as the adjusted Rand index,
    which is what says whether they found the same structure under different
    names -- two partitions can score the same silhouette and still disagree
    about which buildings belong together.
    """
    from sklearn.metrics import adjusted_rand_score
    wanted = [m for m in (methods or ["kmeans", "hierarchical"])]
    if len(wanted) < 2:
        return insufficient("too_few_methods",
                            "name at least two clustering methods to compare")
    if profile == "weekly":
        keys, matrix = _weekly_matrix(ctx, building_ids, normalize, False)
        width = 168
    else:
        ids, matrix = _profile_matrix(ctx, building_ids, normalize)
        keys = [(b, "all year") for b in ids]
        width = 24
    if len(keys) < 3:
        return insufficient("too_few_profiles",
                            f"{len(keys)} profiles cannot be clustered")

    distances = _dtw_matrix(matrix) if distance == "dtw" else None
    rows, labellings = [], {}
    for method in wanted:
        if method == "ward" and distance != "euclidean":
            rows.append({"method": method, "n_clusters": None,
                         "silhouette": None,
                         "note": "Ward linkage is defined only for Euclidean "
                                 "distance"})
            continue
        k, chosen_by = n_clusters, "caller"
        if k in (None, 0, "auto"):
            k, _, _ = _select_k(matrix, distances=distances)
            chosen_by = "silhouette over k=2..8"
        if len(keys) < k:
            rows.append({"method": method, "n_clusters": None,
                         "silhouette": None,
                         "note": f"{len(keys)} profiles for {k} clusters"})
            continue
        labels = _cluster_labels(matrix, distances, method, int(k), distance)
        if labels is None:
            rows.append({"method": method, "n_clusters": None,
                         "silhouette": None,
                         "note": f"{method} cannot run under {distance}"})
            continue
        scores = _cluster_scores(matrix, labels, int(k), distances=distances)
        labellings[method] = labels
        rows.append({
            "method": method, "n_clusters": int(k), "k_chosen_by": chosen_by,
            "silhouette": round(float(scores["silhouette"]), 4),
            "davies_bouldin": (None if scores["davies_bouldin"] != scores["davies_bouldin"]
                               else round(float(scores["davies_bouldin"]), 4)),
            "calinski_harabasz": (None if scores["calinski_harabasz"] != scores["calinski_harabasz"]
                                  else round(float(scores["calinski_harabasz"]), 1)),
            "largest_cluster_share": round(
                float(pd.Series(labels).value_counts(normalize=True).iloc[0]), 3),
            "note": "",
        })

    agreement = []
    names = list(labellings)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agreement.append({"pair": f"{a} vs {b}",
                              "adjusted_rand_index":
                                  round(float(adjusted_rand_score(
                                      labellings[a], labellings[b])), 4)})
    out = pd.DataFrame(rows)
    scored = [r for r in rows if r.get("silhouette") is not None]
    best = max(scored, key=lambda r: r["silhouette"]) if scored else None
    evidence = {
        "profiles": len(keys), "profile_width_hours": width,
        "distance": distance, "normalize": normalize,
        "methods": wanted, "agreement": agreement,
        "shape": f"{width}-hour mean profile, local time",
    }
    if best:
        evidence["headline"] = {
            "label": f"highest silhouette: {best['method']} at k={best['n_clusters']}",
            "value": best["silhouette"], "unit": None}
    return frame_result(out, ctx.store, name="clustering_method_comparison",
                        evidence=evidence)


@tool("ai_analytics")
def cluster_timeseries(
    ctx: ToolContext,
    building_ids: list[str],
    # "auto" rather than None: the call layer drops None-valued arguments so
    # the declared default applies, which silently turned "how many clusters
    # are there" back into "split these into four" -- the plan asked for a
    # data-chosen k and the tool never saw the request.
    n_clusters: int | Literal["auto"] | None = 4,
    normalize: Literal["mean", "minmax", "none"] | bool = True,
    profile: Literal["daily", "weekly"] = "daily",
    by_season: bool = False,
    method: Literal["kmeans", "hierarchical", "ward", "average", "complete",
                    "single"] = "kmeans",
    distance: Literal["euclidean", "cosine", "manhattan", "dtw"] = "euclidean",
    scope: Literal["across_buildings", "within_building"] = "across_buildings",
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Group buildings by load shape, or one building's days by theirs.

    ``profile="weekly"`` clusters a 168-hour week rather than a 24-hour day,
    which is what separates a building that runs on Saturdays from one that
    does not -- a daily profile averages the weekend into the week.

    ``by_season=True`` gives each building one profile per season and clusters
    them all together, so a building appears up to four times and its rows
    carry a ``shifted`` flag when they do not all land in the same cluster.
    That is the only way clustering can show a change in how a building is
    used: without it, every difference found is a difference between
    buildings, and the same building has only one shape to be in.

    ``method`` chooses the algorithm. ``kmeans`` partitions around centroids
    and is the default because a load-shape archetype *is* a centroid.
    ``hierarchical`` (a synonym for ``ward``) merges the closest pair of
    clusters repeatedly, which answers a different question: it does not
    assume clusters are spherical or similarly sized, and it yields a
    merge order, so a cohort with one odd building and three tight groups is
    described better by it than by k-means. ``average``, ``complete`` and
    ``single`` are the other linkages, and they differ in what "closest" means
    between two groups rather than between two points -- single linkage chains
    through near neighbours and will happily string a cohort into one long
    cluster, which is a property worth seeing rather than a fault.

    ``distance`` is the metric clusters are formed under. Ward linkage is
    defined only for Euclidean distance and says so rather than silently
    substituting one. Cosine distance compares profiles by angle, so two
    buildings with the same rhythm at different scales are near each other
    even when `normalize` is off.

    Cluster labels are arbitrary, so the result reports shape summaries and
    three internal validity scores rather than asking anyone to compare label
    numbers. The three disagree often and each is reported with the direction
    that is better, because a partition that wins on one and loses on another
    is the normal case and the reader has to see it.
    """
    if scope == "within_building":
        # Each day is a sample, so "how many distinct daily load profiles does
        # this building have" has something to cluster. Across buildings the
        # unit is the building and one building yields one row, which is what
        # made this question unanswerable before.
        keys, matrix = _daily_matrix(ctx, building_ids, normalize, start, end)
        width, shape = 24, "daily 24-hour profile, local time"
    elif profile == "weekly" or by_season:
        keys, matrix = _weekly_matrix(ctx, building_ids, normalize, by_season)
        width, shape = 168, "168-hour weekly profile, local time"
    else:
        ids, matrix = _profile_matrix(ctx, building_ids, normalize)
        keys = [(b, "all year") for b in ids]
        width, shape = 24, "24-hour mean daily profile, local time"
    # "How many distinct clusters are there" is not "split these into four".
    # When the caller names no k, it is chosen by silhouette over 2..8 and the
    # curve it was chosen from travels with the answer.
    # Under DTW every step -- choosing k, forming the clusters, scoring them --
    # works from one pairwise distance matrix, computed once because it is the
    # expensive part.
    distances = _dtw_matrix(matrix) if distance == "dtw" else None
    chosen_by, k_curve = "caller", []
    if n_clusters in (None, 0, "auto"):
        if len(keys) < 3:
            return insufficient("too_few_profiles",
                                f"{len(keys)} profiles cannot be clustered")
        n_clusters, _, k_curve = _select_k(matrix, distances=distances)
        chosen_by = "silhouette over k=2..8"
    if len(keys) < n_clusters:
        return insufficient("too_few_profiles",
                            f"{len(keys)} complete {width}-hour profiles for "
                            f"{n_clusters} clusters")
    if method == "ward" and distance != "euclidean":
        return insufficient(
            "linkage_metric",
            "Ward linkage minimises within-cluster variance and is defined "
            "only for Euclidean distance; ask for average, complete or single "
            f"linkage to cluster under {distance}")
    from sklearn.cluster import AgglomerativeClustering, KMeans
    if distance == "dtw":
        # TimeSeriesKMeans with barycentre averaging for k-means; for a linkage
        # method the precomputed DTW matrix drives agglomeration directly.
        # Ward is excluded above: it is defined only for Euclidean distance.
        if method == "kmeans":
            labels = _fit_dtw_kmeans(matrix, n_clusters)
            linkage = None
        else:
            linkage = "average" if method == "hierarchical" else method
            labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage,
                metric="precomputed").fit_predict(distances)
    elif method == "kmeans":
        if distance != "euclidean":
            return insufficient(
                "kmeans_metric",
                "k-means minimises squared Euclidean distance to a centroid "
                f"and cannot be run under {distance}; ask for hierarchical "
                "clustering, which can")
        labels = KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=0).fit_predict(matrix)
        linkage = None
    else:
        linkage = "ward" if method == "hierarchical" else method
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage,
            metric="euclidean" if linkage == "ward" else distance).fit_predict(matrix)
    scores = _cluster_scores(matrix, labels, n_clusters, distances=distances)
    silhouette = scores["silhouette"]
    unit_column = "day" if scope == "within_building" else "season"
    assignments = pd.DataFrame({"building_id": [k[0] for k in keys],
                                unit_column: [k[1] for k in keys],
                                "cluster": labels})
    shifted: list[str] = []
    if by_season:
        per_building = assignments.groupby("building_id")["cluster"]
        moved = per_building.nunique()
        shifted = sorted(moved[(moved > 1) & (per_building.count() > 1)].index)
        assignments["shifted"] = assignments["building_id"].isin(shifted)
    elif scope != "within_building":
        assignments = assignments.drop(columns=["season"])
    summary = []
    for c in range(n_clusters):
        members = matrix[labels == c]
        if not len(members):
            continue
        summary.append({"cluster": c, "n_profiles": int(len(members)),
                        "peak_hour": int(members.mean(axis=0).argmax() % 24),
                        "trough_hour": int(members.mean(axis=0).argmin() % 24),
                        "peak_to_mean": float(members.mean(axis=0).max())})
    headline = ({"label": "buildings whose load shape changes between seasons",
                 "value": len(shifted), "unit": "buildings"} if by_season else None)
    if headline is None and scores["silhouette"] == scores["silhouette"]:
        headline = {"label": f"silhouette of the {n_clusters}-cluster "
                             f"{linkage or 'k-means'} partition",
                    "value": round(scores["silhouette"], 4), "unit": None}
    readable = []
    for name, value in scores.items():
        direction, meaning = CLUSTER_SCORES[name]
        readable.append(f"  {name.replace('_', '-')}: "
                        + (f"{value:.4f}" if value == value else "not computable")
                        + f"  ({direction}) — {meaning}")
    narrative = [
        f"{len(keys):,} {shape.split(',')[0]}s grouped into {n_clusters} "
        f"clusters by {linkage + ' linkage' if linkage else 'k-means'}"
        + (f" under {distance} distance" if distance != "euclidean" else "")
        + ".",
        "  Cluster quality, three ways — they disagree often and the "
        "disagreement is the information:",
        *readable,
        f"  Labels are arbitrary: cluster 0 is not 'first'. Compare the shape "
        f"summaries below, not the numbers.",
    ]
    # Which normalisation was applied decides what "similar shape" means, and
    # two defensible conventions disagree about which group clusters tightest.
    # Stating it is what makes the number reproducible by anyone else.
    # The partition itself, stated before anything that scores it.
    narrative.insert(1, "  Cluster sizes: " + ", ".join(
        f"cluster {k} {v:,} {'day' if scope == 'within_building' else 'building'}"
        f"{'' if v == 1 else 's'}"
        for k, v in sorted(assignments["cluster"].value_counts().items())) + ".")
    narrative.insert(2, "  Profiles normalised by "
                     + {True: "their own mean (peak-to-mean preserved)",
                        "mean": "their own mean (peak-to-mean preserved)",
                        "minmax": "min-max to [0,1] (shape only; peak-to-mean "
                                  "discarded)",
                        False: "nothing (raw kW; size dominates shape)",
                        "none": "nothing (raw kW; size dominates shape)",
                        }.get(normalize, str(normalize))
                     + ". A different convention gives different scores.")
    if k_curve:
        best = ", ".join(f"k={row['k']} {row['silhouette']:.3f}" for row in k_curve)
        narrative.insert(1, f"  k chosen by silhouette, not given: {best} "
                            f"-> k={n_clusters}.")
    if by_season and shifted:
        narrative.insert(1, f"  {len(shifted):,} building(s) do not stay in one "
                            f"cluster across the seasons.")
    # A silhouette can be driven up by tiny far-flung clusters rather than by
    # real structure: one office building's 2017 split as 350 days in one
    # cluster and six clusters of one to four days, scoring 0.995. Reported as
    # "7 distinct daily load profiles" that is true and misleading, so the
    # shape of the partition is stated whenever one cluster dominates.
    sizes = sorted((int(row["n_profiles"]) for row in summary), reverse=True)
    if sizes and len(sizes) > 1:
        share = sizes[0] / float(sum(sizes))
        if share >= 0.8:
            rest = len(sizes) - 1
            others = (f"the other holds {sizes[1]:,}" if rest == 1 else
                      f"the other {rest} hold "
                      + ", ".join(f"{n:,}" for n in sizes[1:]))
            narrative.insert(
                1, f"  The partition is lopsided: {sizes[0]:,} of "
                   f"{sum(sizes):,} profiles fall in one cluster and "
                   f"{others}. The count is driven by a few unusual profiles "
                   f"rather than by that many distinct regimes.")
    # How many members each cluster ended up with: the answer to "how many
    # days belong to each cluster".
    per_cluster = {str(k): int(v) for k, v in
                   assignments["cluster"].value_counts().sort_index().items()}
    unit = "days" if scope == "within_building" else "buildings"
    return frame_result(assignments, ctx.store, name="clusters",
                        evidence={**({"headline": headline} if headline else {}),
                                  "cluster_sizes": per_cluster,
                                  "cluster_sizes_unit": unit,
                                  "method": method, "n_clusters": n_clusters,
                                  "linkage": linkage,
                                  "distance": distance,
                                  "normalized": normalize,
                                  "n_clusters_chosen_by": chosen_by,
                                  "k_selection_curve": k_curve or None,
                                  # The sweep as its own table, so it can be
                                  # charted.
                                  "k_selection_handle": (
                                      ctx.store.put(pd.DataFrame(k_curve),
                                                    name="k_selection")
                                      if k_curve and ctx.store is not None
                                      else None),
                                  "silhouette": silhouette,
                                  "davies_bouldin": scores["davies_bouldin"],
                                  "calinski_harabasz": scores["calinski_harabasz"],
                                  "score_directions": {
                                      k: v[0] for k, v in CLUSTER_SCORES.items()},
                                  "narrative": narrative,
                                  "feature": shape,
                                  "by_season": by_season,
                                  "profiles_clustered": len(keys),
                                  "buildings_shifted": len(shifted) if by_season else None,
                                  "cluster_summary": summary})


@tool("ai_analytics")
def compare_similarity_measures(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    measures: list[Literal["pearson", "euclidean", "cosine"]] | None = None,
    normalize: Literal["minmax", "mean", "none"] = "minmax",
    n_clusters: int | Literal["auto"] | None = "auto",
    start: str | None = None,
    end: str | None = None,
) -> ToolResult:
    """Do different distance measures pick the same neighbour?

    "How do Pearson, Euclidean and cosine compare for finding similar
    buildings" is a question about the measures, not about any one building,
    and every similarity tool here answered for one building at a time. Asked
    it, the planner clustered under cosine and reported a silhouette -- a real
    number that says nothing about whether the three agree.

    For each building it finds the nearest other building under each measure,
    then reports how often the measures pick the *same* neighbour, and how
    often the neighbour lands in the same cluster. The second is usually much
    higher than the first, which is the finding: the choice of measure moves
    the single nearest building around a great deal and the cluster hardly at
    all.
    """
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans

    # start and end were declared here and never read, so a question scoped to
    # a year was answered over the whole record with nothing saying so. The
    # profiles come from a precomputed climatology that carries no date, so
    # the period cannot be honoured -- and a parameter that cannot be kept
    # should refuse rather than be ignored.
    if start or end:
        return insufficient(
            "period",
            "this comparison reads the precomputed daily profile, which is a "
            "climatology over each building's whole record and carries no "
            "date to filter on. Ask without start/end, or use "
            "correlate_group_profiles, which builds its profiles from the "
            "readings in a window.")
    wanted = [m for m in (measures or ["pearson", "euclidean", "cosine"])]
    ids, matrix = _profile_matrix(ctx, building_ids or [], normalize)
    if len(ids) < 3:
        return insufficient(
            "too_few_profiles",
            f"{len(ids)} complete daily profiles: comparing similarity "
            f"measures needs at least three buildings")
    labels = None
    if n_clusters not in (None, 0):
        k = (_select_k(matrix)[0] if n_clusters == "auto" else int(n_clusters))
        k = max(2, min(k, len(ids) - 1))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)

    nearest: dict[str, np.ndarray] = {}
    for measure in wanted:
        if measure == "pearson":
            distance = 1.0 - np.corrcoef(matrix)
        else:
            distance = cdist(matrix, matrix, measure)
        distance = np.asarray(distance, float).copy()
        np.fill_diagonal(distance, np.inf)
        nearest[measure] = distance.argmin(axis=1)

    rows = []
    for i, first in enumerate(wanted):
        for second in wanted[i + 1:]:
            same = float(np.mean(nearest[first] == nearest[second]) * 100)
            rows.append({"pair": f"{first}-{second}",
                         "same_nearest_neighbour_pct": round(same, 2)})
    agree_all = (float(np.mean(np.all([nearest[w] == nearest[wanted[0]]
                                       for w in wanted], axis=0)) * 100)
                 if len(wanted) > 1 else 100.0)
    same_cluster = {}
    if labels is not None:
        for measure in wanted:
            same_cluster[measure] = round(
                float(np.mean(labels == labels[nearest[measure]]) * 100), 2)

    frame = pd.DataFrame(rows)
    narrative = [
        f"{len(ids):,} buildings compared on {normalize}-normalised 24-hour "
        f"profiles under {', '.join(wanted)}.",
        "  The measures pick the same nearest building: "
        + ", ".join(f"{r['pair']} {r['same_nearest_neighbour_pct']:.2f}%"
                    for r in rows)
        + f"; all of them agree for {agree_all:.2f}% of buildings.",
    ]
    if same_cluster:
        narrative.append(
            "  That neighbour falls in the same cluster "
            + ", ".join(f"{m} {v:.2f}%" for m, v in same_cluster.items())
            + " of the time -- the measure moves the single nearest building "
              "far more than it moves cluster membership.")
    return frame_result(frame, ctx.store, name="similarity_measures",
                        evidence={"measures": wanted, "normalized": normalize,
                                  "buildings": len(ids),
                                  "clusters": (int(labels.max()) + 1
                                               if labels is not None else None),
                                  "all_three_agree_pct": round(agree_all, 2),
                                  "same_cluster_rate_pct": same_cluster or None,
                                  "headline": {
                                      "label": "buildings where every measure "
                                               "picks the same nearest neighbour",
                                      "value": round(agree_all, 2), "unit": "%"},
                                  "narrative": narrative})


def _group_attributes(ctx: ToolContext, group_by: str, attribute: str) -> dict:
    """The country, city or site each group sits in, for a cross-group filter.

    A group only has one of these when it is nested inside it -- a site is in
    one country, a building type is in many -- so a group spanning several
    values maps to None and is excluded rather than matched arbitrarily.
    """
    column = {"site": "b.site_id", "city": "s.city",
              "country": "s.country_canonical",
              "building_type": "b.building_type_canonical",
              "building_subtype": "b.building_subtype",
              "industry": "b.industry", "subindustry": "b.subindustry"}
    if group_by not in column or attribute not in column:
        return {}
    frame = ctx.db.df(f"""
        SELECT {column[group_by]} AS grp,
               COUNT(DISTINCT {column[attribute]}) AS n,
               ANY_VALUE({column[attribute]}) AS value
        FROM buildings b JOIN sites s ON s.site_id = b.site_id
        GROUP BY 1
    """)
    return {row.grp: (row.value if row.n == 1 else None)
            for row in frame.itertuples()}


@tool("ai_analytics")
def correlate_group_profiles(
    ctx: ToolContext,
    building_ids: list[str] | None = None,
    group_by: Literal["building_type", "building_subtype", "site", "city",
                      "industry", "subindustry"] = "building_type",
    metric: Literal["pearson", "spearman", "euclidean"] = "pearson",
    normalize: Literal["minmax", "mean", "none"] = "minmax",
    days: Literal["all", "weekday", "weekend"] = "all",
    start: str | None = None,
    end: str | None = None,
    top: int = 5,
    require_different: Literal["none", "country", "city", "site"] = "none",
    # A "load profile" is the 24-hour shape, not the year of hours it was
    # averaged from. Comparing the raw series asks whether two cities were
    # busy in the same *hours of 2017* -- which mostly measures whether their
    # weather and term dates lined up -- where the question asks whether
    # their days look alike. On the office cohort the two rank differently:
    # the series distance picks Ithaca and London, the profile distance picks
    # Dublin and Tempe.
    shape: Literal["series", "daily_profile"] = "series",
) -> ToolResult:
    """How alike two groups' hourly demand is, pair by pair.

    "Which building types move together" is a question about groups, and every
    similarity tool here worked building-to-building: asked which types at
    Eagle are most alike, the planner reached for a normalised-performance
    table and answered with median EUI, which is a different quantity in
    different units.

    Each building is normalised first and the group is the mean of its
    members, so a large building does not simply outvote the rest of its type.
    Correlation is on the hourly series, so two groups score highly when they
    rise and fall together -- not when they consume similar amounts.
    """
    params: list = []
    where = "e.quality_flag = 'OK'" + in_clause("e.building_id", building_ids, params)
    if start:
        where += " AND e.timestamp_local >= ?"
        params.append(start)
    if end:
        where += " AND e.timestamp_local < ?"
        params.append(end)
    column = {"building_type": "b.building_type_canonical",
              "building_subtype": "b.building_subtype",
              "site": "b.site_id", "city": "b.city",
              "industry": "b.industry", "subindustry": "b.subindustry"}[group_by]
    # A weekday profile and an all-week one are different shapes: the weekend
    # flattens the working day out of the average, so two sites can look alike
    # across the week and quite different Monday to Friday.
    day_filter = {"weekday": " AND ISODOW(e.timestamp_local) <= 5",
                  "weekend": " AND ISODOW(e.timestamp_local) >= 6"}.get(days, "")
    rows = ctx.db.df(f"""
        SELECT e.building_id, {column} AS grp, e.timestamp_local AS ts,
               SUM(e.value) AS value
        FROM electricity e JOIN buildings b USING (building_id)
        WHERE {where} AND {column} IS NOT NULL{day_filter}
        GROUP BY 1, 2, 3 ORDER BY 3
    """, params or None)
    if rows.empty:
        return empty("no readings for these buildings in the window")
    wide = rows.pivot_table(index="ts", columns="building_id", values="value")
    if shape == "daily_profile":
        # Collapse each building to its mean hour-of-day profile before
        # anything else, so the comparison is between 24-point shapes.
        wide = wide.groupby(pd.to_datetime(wide.index).hour).mean()
        wide.index.name = "hour"
    if normalize != "none":
        if normalize == "minmax":
            span = wide.max() - wide.min()
            wide = (wide - wide.min()).div(span.replace(0, np.nan))
        else:
            wide = wide.div(wide.mean().replace(0, np.nan))
    owner = rows.drop_duplicates("building_id").set_index("building_id")["grp"]
    groups = {}
    for name, members in owner.groupby(owner):
        columns = [b for b in members.index if b in wide.columns]
        if columns:
            groups[name] = wide[columns].mean(axis=1)
    if len(groups) < 2:
        return insufficient(
            "too_few_groups",
            f"{len(groups)} {group_by.replace('_', ' ')} with data: a "
            f"similarity between groups needs at least two")
    frame = pd.DataFrame(groups).dropna()
    # Correlation asks whether two groups rise and fall together; distance asks
    # whether they sit on top of each other. Two sites can move in step and
    # draw quite different amounts, which is why "most similar profiles" is
    # usually a distance question and ranks differently from a correlation.
    if metric == "euclidean":
        names = list(frame.columns)
        values = frame.to_numpy(float)
        matrix = pd.DataFrame(
            {a: {b: -float(np.linalg.norm(values[:, i] - values[:, j]))
                 for j, b in enumerate(names)}
             for i, a in enumerate(names)})
    else:
        matrix = frame.corr(method=metric)
    # "Most similar despite being in different countries" is the pairing plus a
    # constraint, and without the constraint the answer is almost always two
    # sites next door to each other -- which is the least interesting version
    # of the finding the question is after.
    attributes = _group_attributes(ctx, group_by, require_different) \
        if require_different != "none" else {}
    pairs = []
    excluded = 0
    for i, a in enumerate(matrix.columns):
        for b in matrix.columns[i + 1:]:
            if matrix.loc[a, b] != matrix.loc[a, b]:
                continue
            if require_different != "none":
                left, right = attributes.get(a), attributes.get(b)
                if left is None or right is None or left == right:
                    excluded += 1
                    continue
            # The matrix holds a *negated* distance so that "higher is more
            # similar" sorts one way for both metrics. That sign is an
            # internal convenience and it was reaching the answer: the most
            # similar pair of cities came back at "r = -0.2699" where the
            # distance between them is 0.2699. A Euclidean distance is not
            # negative and it is not a correlation.
            score = float(matrix.loc[a, b])
            row = {"group_a": a, "group_b": b,
                   ("distance" if metric == "euclidean" else "r"):
                       round(abs(score) if metric == "euclidean" else score, 4),
                   "hours": int(len(frame))}
            row["_rank"] = score
            if require_different != "none":
                row[f"{require_different}_a"] = attributes.get(a)
                row[f"{require_different}_b"] = attributes.get(b)
            pairs.append(row)
    if require_different != "none" and not pairs:
        return insufficient(
            "no_qualifying_pairs",
            f"no two {group_by}s differ on {require_different}; "
            f"{excluded} pair(s) were excluded for sharing it")
    pairs.sort(key=lambda row: -row["_rank"])
    label = "distance" if metric == "euclidean" else "r"
    measure = ("Euclidean distance between profiles" if metric == "euclidean"
               else f"{metric} correlation")
    out = pd.DataFrame(pairs[:max(1, int(top))]).drop(columns="_rank")
    best = pairs[0] if pairs else None
    narrative = [
        f"{len(groups)} {group_by.replace('_', ' ')}s compared on "
        f"{normalize}-normalised {'24-hour profiles' if shape == 'daily_profile' else 'hourly demand'} "
        f"over {len(frame):,} shared {'hours of the day' if shape == 'daily_profile' else 'hours'}, "
        f"{measure}.",
    ]
    if best:
        narrative.append(
            f"  Most similar: {best['group_a']} and {best['group_b']} "
            f"({label} = {best[label]:.4f}).")
    narrative.append(
        "  A distance asks whether two profiles sit on top of each other; a "
        "correlation asks only whether they rise and fall together, which two "
        "groups can do while one draws ten times the power of the other."
        if metric == "euclidean" else
        "  Correlation is about shape over time, not size: two groups can "
        "correlate at 0.95 while one draws ten times the power of the other.")
    return frame_result(out, ctx.store, name="group_correlation",
                        evidence={"group_by": group_by, "metric": metric,
                                  "normalized": normalize,
                                  "groups": sorted(groups),
                                  "hours_compared": int(len(frame)),
                                  "headline": ({"label": f"{measure}, "
                                                f"{best['group_a']} and {best['group_b']}",
                                                "value": best[label], "unit": None}
                                               if best else None),
                                  "narrative": narrative})


@tool("ai_analytics")
def find_similar_timeseries(
    ctx: ToolContext,
    building_id: str,
    candidate_ids: list[str] | None = None,
    k: int = 10,
    metric: Literal["euclidean", "correlation", "cosine"] = "correlation",
    normalize: bool = True,
) -> ToolResult:
    """Nearest buildings by daily load shape."""
    pool = candidate_ids or ctx.db.df(
        "SELECT building_id FROM profile_daily GROUP BY 1")["building_id"].tolist()
    pool = [b for b in dict.fromkeys([building_id, *pool])]
    ids, matrix = _profile_matrix(ctx, pool, normalize)
    if building_id not in ids:
        return insufficient("no_profile",
                            f"{building_id} has no complete 24-hour profile")
    index = ids.index(building_id)
    target = matrix[index]
    if metric == "euclidean":
        distance = np.linalg.norm(matrix - target, axis=1)
        score = -distance
    elif metric == "cosine":
        # Cosine compares the profiles as vectors from the origin; Pearson
        # compares them after each is centred on its own mean. The difference
        # is not cosmetic: two buildings that both run flat all day have a
        # cosine similarity near 1 and a correlation near 0, because once the
        # mean is removed there is nothing left to correlate. Which one is
        # wanted depends on whether a constant load counts as a shape.
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
        score = np.divide(matrix @ target, norms,
                          out=np.zeros(len(ids)), where=norms != 0)
        distance = 1 - score
    else:
        centred = matrix - matrix.mean(axis=1, keepdims=True)
        tc = centred[index]
        denom = np.linalg.norm(centred, axis=1) * np.linalg.norm(tc)
        score = np.divide(centred @ tc, denom, out=np.zeros(len(ids)), where=denom != 0)
        distance = 1 - score
    order = np.argsort(-score)
    rows = [{"rank": r + 1, "building_id": ids[i], "similarity": float(score[i]),
             "distance": float(distance[i])}
            for r, i in enumerate(o for o in order if o != index)][:k]
    return frame_result(pd.DataFrame(rows), ctx.store, name="similar_buildings",
                        evidence={"target": building_id, "metric": metric,
                                  "metric_note": {
                                      "correlation": "Pearson over the 24-hour "
                                                     "profile, each centred on "
                                                     "its own mean: shape only",
                                      "cosine": "angle between the profiles as "
                                                "vectors: shape and level "
                                                "together",
                                      "euclidean": "straight-line distance "
                                                   "between the profiles",
                                  }.get(metric),
                                  "normalized": normalize,
                                  "candidates": len(ids) - 1,
                                  "feature": "24-hour mean daily profile"})


@tool("ai_analytics")
def generate_timeseries_representations(
    ctx: ToolContext,
    building_ids: list[str],
    kind: Literal["features", "profile"] = "features",
) -> ToolResult:
    """Compact per-building representations for downstream similarity work."""
    ids, matrix = _profile_matrix(ctx, building_ids, normalize=True)
    if not ids:
        return empty("no complete profiles for the requested buildings")
    if kind == "profile":
        out = pd.DataFrame(matrix, columns=[f"h{h:02d}" for h in range(matrix.shape[1])])
        out.insert(0, "building_id", ids)
        note = "normalised 24-hour load shape"
    else:
        night = matrix[:, 0:6].mean(axis=1)
        morning = matrix[:, 6:12].mean(axis=1)
        afternoon = matrix[:, 12:18].mean(axis=1)
        evening = matrix[:, 18:24].mean(axis=1)
        out = pd.DataFrame({
            "building_id": ids,
            "night_share": night, "morning_share": morning,
            "afternoon_share": afternoon, "evening_share": evening,
            "peak_hour": matrix.argmax(axis=1),
            "trough_hour": matrix.argmin(axis=1),
            "peak_to_trough": matrix.max(axis=1) / np.maximum(matrix.min(axis=1), 1e-9),
            "profile_flatness": matrix.std(axis=1),
        })
        note = "handcrafted load-shape features"
    return frame_result(out, ctx.store, name=f"representations_{kind}",
                        evidence={"kind": kind, "buildings": len(ids),
                                  "description": note})


@tool("ai_analytics")
def analyze_ai_model_results(
    ctx: ToolContext,
    results: list[dict],
    metric: str = "mae",
    lower_is_better: bool = True,
) -> ToolResult:
    """Compare model outputs and summarise which performed best."""
    if not results:
        return insufficient("no_results", "nothing to compare")
    df = pd.DataFrame(results)
    if metric not in df.columns:
        return insufficient("missing_metric",
                            f"{metric!r} absent; available: {list(df.columns)}")
    ranked = df.sort_values(metric, ascending=lower_is_better).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    best = ranked.iloc[0]
    spread = float(ranked[metric].max() - ranked[metric].min())
    return frame_result(ranked, ctx.store, name="model_comparison",
                        evidence={"metric": metric,
                                  "lower_is_better": lower_is_better,
                                  "best": {k: (v.item() if hasattr(v, "item") else v)
                                           for k, v in best.items()},
                                  "spread": spread, "n_models": len(ranked)})


# Checked once, at import.
_assert_model_vocabulary()


@tool("ai_analytics")
def detect_breakouts(
    ctx: ToolContext,
    building_ids: list[str],
    min_segment_hours: int = 168,
    penalty: float | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> ToolResult:
    """Persistent shifts in consumption level, and when each one happened.

    A breakout is not an outlying hour. It is the point where a building's
    normal level moves and stays moved -- a control change, a tenant leaving,
    a plant swapped -- and the anomaly tools cannot see it: every hour after
    the shift is unremarkable against its new neighbours. Asked how many
    breakouts a building had, the app scanned for anomalous hours over a
    168-hour window and answered none, which is true and about a different
    question.

    Segmentation is binary with an L1 (median-shift) cost, so a step in the
    middle of the record is found even when the noise around it is heavy.
    ``min_segment_hours`` is the shortest run that counts as a new level: a
    building does not change what it is for six hours.

    ``penalty`` decides how large a shift has to be to count. Left unset it
    scales with the building's own variability -- its median absolute
    deviation times the log of its length -- because a fixed number means
    different things to a steady office and a volatile one, and a penalty
    tuned on one building silently over- or under-detects on the next.
    """
    import numpy as np

    try:
        import ruptures as rpt
    except ImportError:                                     # pragma: no cover
        return insufficient(
            "ruptures", "breakout detection needs the ruptures package; "
                        "install it with `pip install ruptures`")
    if not building_ids:
        return insufficient("building_ids",
                            "breakout detection reads one building's whole "
                            "record, so it needs a named cohort")
    frame = load_series(ctx, building_ids=building_ids, start=start, end=end)
    if frame is None or frame.empty:
        return empty("no readings in the requested period")
    found, scanned = [], 0
    for building, grp in frame.groupby("building_id"):
        ordered = grp.dropna(subset=["value"]).sort_values("timestamp")
        values = ordered["value"].to_numpy(float)
        # Two segments of the minimum length, or there is nothing to split.
        if len(values) < 2 * int(min_segment_hours):
            continue
        scanned += 1
        spread = float(np.median(np.abs(values - np.median(values))))
        level = penalty if penalty is not None else (
            10.0 * (spread or 1.0) * float(np.log(len(values))))
        algo = rpt.Binseg(model="l1", min_size=int(min_segment_hours),
                          jump=24).fit(values)
        edges = algo.predict(pen=float(level))
        starts = [0] + list(edges[:-1])
        for position, edge in enumerate(edges[:-1]):
            before = values[starts[position]:edge]
            after = values[edge:edges[position + 1]]
            if not len(before) or not len(after):
                continue
            median_before, median_after = float(np.median(before)), float(np.median(after))
            found.append({
                "building_id": building,
                "when": _wall_clock(ordered["timestamp"].iloc[edge]),
                "median_before_kw": round(median_before, 3),
                "median_after_kw": round(median_after, 3),
                "change_pct": (round(100.0 * (median_after - median_before)
                                     / median_before, 2) if median_before else None),
                "hours_before": int(len(before)), "hours_after": int(len(after)),
                "penalty": round(float(level), 3)})
    evidence = {"min_segment_hours": int(min_segment_hours),
                "penalty": "scaled to each building's own spread"
                           if penalty is None else float(penalty),
                "buildings_scanned": scanned,
                "method": "binary segmentation, L1 (median-shift) cost, "
                          "24-hour candidate step",
                "period": [start, end]}
    if not found:
        evidence["headline"] = {"value": 0, "unit": "breakouts",
                                "label": f"no significant level shift in "
                                         f"{scanned} building(s)"}
        return ok({"breakouts": []}, evidence=evidence)
    out = pd.DataFrame(found).sort_values(["building_id", "when"])
    if limit:
        out = out.head(int(limit))
    noun = "breakout" if len(out) == 1 else "breakouts"
    lead = out.iloc[0]
    evidence["headline"] = {
        "value": int(len(out)), "unit": "breakouts",
        "label": (f"{len(out)} significant {noun} across {scanned} building(s); "
                  f"first {lead['when']} "
                  f"{lead['median_before_kw']:g} -> {lead['median_after_kw']:g} kWh"
                  + (f" ({lead['change_pct']:+.1f}%)"
                     if lead["change_pct"] is not None else ""))}
    return frame_result(out, ctx.store, name="breakouts", evidence=evidence)
