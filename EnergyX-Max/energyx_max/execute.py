"""Run a Coding Agent program on slot values, in worker processes that hold the data in memory.

Each call compiles the program, calls answer(data, **slots) under a time limit, and checks that the
result is a usable answer: JSON-serializable, not empty, and with no NaN or infinite number.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import signal
import traceback

import numpy as np
import pandas as pd

from . import config

TIME_LIMIT = config.setting("pipeline", "program_time_limit_seconds")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout(f"took longer than {TIME_LIMIT} s")


def _init() -> None:
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import sys
    from . import bdg2
    bdg2.electricity_wide(), bdg2.metadata(), bdg2.weather_long()
    sys.modules["data"] = bdg2  # programs sometimes write `from data import ...`; point it at the same module
    signal.signal(signal.SIGALRM, _alarm)


def _plain(value):
    """numpy and pandas values as plain Python, recursively."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        items = [_plain(v) for v in (value.tolist() if hasattr(value, "tolist") else value)]
        return sorted(items, key=str) if isinstance(value, set) else items
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _plain(value.to_dict(orient="index"))
    return value


def _problem(answer) -> str | None:
    if answer is None:
        return "answer is None"
    if isinstance(answer, (list, dict, str)) and len(answer) == 0:
        return "answer is empty"
    text = json.dumps(answer)
    if "NaN" in text or "Infinity" in text:
        return "answer contains NaN or infinity"
    if len(text) > 20000:
        return "answer is too long"
    def numbers(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from numbers(x)
        elif isinstance(v, list):
            for x in v:
                yield from numbers(x)
        elif isinstance(v, float):
            yield v
    if any(not math.isfinite(x) for x in numbers(answer)):
        return "answer contains a non-finite number"
    return None


def run(code: str, slots: dict) -> dict:
    """Run one program on one set of slot values. Called inside a worker."""
    from . import bdg2
    signal.alarm(TIME_LIMIT)
    try:
        namespace: dict = {}
        exec(compile(code, "<program>", "exec"), namespace)
        result = namespace["answer"](bdg2, **slots)
        if not isinstance(result, dict) or "answer" not in result:
            return {"ok": False, "error": "answer() must return a dict with an 'answer' key"}
        answer = _plain(result["answer"])
        problem = _problem(answer)
        if problem:
            return {"ok": False, "error": problem}
        unit = result.get("unit")
        return {"ok": True, "answer": answer, "unit": None if unit is None else str(unit)}
    except _Timeout as err:
        return {"ok": False, "error": str(err)}
    except Exception:  # the program is untrusted model output; any failure is a rejected run
        return {"ok": False, "error": traceback.format_exc(limit=3)[-1500:]}
    finally:
        signal.alarm(0)


def pool(workers: int | None = None):
    workers = workers or config.setting("pipeline", "workers")
    return mp.get_context("spawn").Pool(workers, initializer=_init, maxtasksperchild=200)


class Runner:
    """Runs programs in worker processes with a hard time limit.

    The in-process alarm can't interrupt code stuck inside a compiled library, so a result that
    doesn't arrive in time counts as a timeout and the whole pool is replaced.
    """

    def __init__(self, workers: int | None = None):
        self.workers = workers
        self.pool = pool(workers)

    def run_many(self, code: str, slot_sets: list[dict]) -> list[dict]:
        jobs = [self.pool.apply_async(run, (code, slots)) for slots in slot_sets]
        results, stuck = [], False
        for job in jobs:
            try:
                results.append(job.get(timeout=TIME_LIMIT + 30))
            except mp.TimeoutError:
                results.append({"ok": False, "error": f"took longer than {TIME_LIMIT} s (worker stopped)"})
                stuck = True
        if stuck:
            self.pool.terminate()
            self.pool = pool(self.workers)
        return results

    def close(self) -> None:
        self.pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
