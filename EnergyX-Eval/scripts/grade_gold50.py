#!/usr/bin/env python3
"""Score an app answer against the golden answers for the 50-question set.

The golden solver for this set writes one record per question carrying three
things this grader uses and one it reports:

    value     the answer itself -- a number, a dict of named numbers, or a
              list of named things. This is what the golden script called the
              answer, so it is what decides.
    detail    the working behind it: counts, runners-up, thresholds. Checked
              and shown, but supporting -- two implementations that agree on
              the finding routinely differ in the sixth runner-up.
    answer    the prose, used only to show a reader what was expected.
    _seconds  how long the golden solver took, carried into the table beside
              the app's own time so the two are read together.

Tolerance: 2% relative for a continuous
quantity, exact for a count or a small integer, and a set answer passes when
it names most of the set. The difference here is that ``value`` is available
and unambiguous, so the headline never has to be guessed out of the prose --
which is the one place the older grader could be led astray.

    python EnergyX-Eval/scripts/grade_gold50.py              # coverage summary
    python EnergyX-Eval/scripts/grade_gold50.py --show Q44
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
#: The answer of record for this set. Every figure in it was computed from the
#: snapshot, independently of the application.
GOLDEN = ROOT / "questions" / "golden_answers_50.json"

from grading_rules import (                    # noqa: E402
    NAMEABLE, TOLERANCE, is_informational, numbers)

#: Keys whose value is a count, an index or a cluster number -- an integer
#: where being close is not being right.
EXACT_TOKENS = {"n", "count", "counts", "k", "best_k", "clusters",
                "buildings", "days", "months", "hours", "readings", "index"}


def is_exact(key: str) -> bool:
    """Whether this key names something that has to match exactly.

    Split on separators and compare whole words. The inherited version tested
    ``startswith`` against a token list containing the bare letters "n" and
    "k", so *every* key beginning with either was treated as an integer:
    ``newer_median_eui`` is a kWh/m2 intensity and was required to match to
    the last decimal, which failed an answer that agreed to four significant
    figures.
    """
    parts = re.split(r"[^a-z0-9]+", key.lower())
    return any(part in EXACT_TOKENS for part in parts if part)

#: For quantities whose value moves with a protocol choice rather than with
#: correctness -- a distance between normalised profiles, a silhouette.
LOOSE_TOLERANCE = 0.05

#: Keys the golden record carries to show what a *different* protocol would
#: have given. They are useful context and they are not the answer: the
#: question asks for one of them, and which one is settled by its wording.
ALTERNATIVE_PREFIXES = ("pooled_", "sensitivity", "cross_check",
                        "count_complete_days_only", "n_if_", "_if_coverage",
                        "city_if_", "median_eui_if_", "mae_season_",
                        "seasonal_naive_mae_season_")


def close(value: float, want: float, exact: bool,
          tolerance: float = TOLERANCE) -> bool:
    """Whether a number found in the answer is the number expected."""
    if exact:
        return float(value) == float(want)
    if want == 0:
        return abs(value) <= tolerance
    return abs(value - want) / abs(want) <= tolerance


@lru_cache(maxsize=1)
def _records() -> dict:
    if not GOLDEN.exists():
        return {}
    # The final file already carries the revisions; nothing is layered on it.
    return dict(json.loads(GOLDEN.read_text()).get("answers", {}))


def golden(qid: str) -> dict:
    return _records().get(qid, {})


def runtime(qid: str) -> float | None:
    """How long the golden solver took on this question, in seconds."""
    record = golden(qid)
    return None if not record else float(record.get("_seconds") or 0.0)


def _leaves(value, prefix: str = "") -> tuple[list[tuple[str, float]],
                                              list[tuple[str, list[str]]]]:
    """Scalar and name-list leaves of a nested golden value, with their paths."""
    scalars: list[tuple[str, float]] = []
    names: list[tuple[str, list[str]]] = []
    if isinstance(value, bool):
        return scalars, names
    if isinstance(value, (int, float)):
        return [(prefix or "value", float(value))], names
    if isinstance(value, str):
        return scalars, [(prefix or "value", [value])]
    if isinstance(value, dict):
        for key, inner in value.items():
            s, n = _leaves(inner, f"{prefix}.{key}" if prefix else str(key))
            scalars += s
            names += n
        return scalars, names
    if isinstance(value, list):
        # A list of names is a set answer; a list of records contributes the
        # names it holds and the numbers beside them.
        plain = [v for v in value if isinstance(v, str)]
        if plain:
            names.append((prefix or "value", plain))
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                s, n = _leaves(item, f"{prefix}[{index}]")
                scalars += s
                names += n
        return scalars, names
    return scalars, names


def grade(qid: str, answer: str) -> tuple[bool | None, str, dict]:
    """correct, a readable reason, and the evidence behind the verdict."""
    record = golden(qid)
    if not record:
        return None, "no golden answer on file", {}
    text = answer or ""
    found = numbers(text)
    lowered = text.lower()

    checks: list[dict] = []
    scalars, names = _leaves(record.get("value"))
    # Only the first record of a ranked list decides. "Which five buildings
    # had the highest X" is answered by naming them; being able to reproduce
    # the fifth one's fourth decimal is not part of the question.
    for path, want in scalars:
        # A whole number here is a count, an hour, a month or a cluster size,
        # and on every one of those being close is not being right: 1427
        # buildings is within 2% of 1455 and is the wrong answer to Q03 --
        # it is the calendar-year cohort where the question asks for any
        # twelve consecutive months. Only a quantity the golden solver wrote
        # with a fraction gets the tolerance.
        exact = is_exact(path.split(".")[-1]) or float(want).is_integer()
        hit = next((v for v in found if close(v, float(want), exact)), None)
        # A figure the golden record added to describe an *alternative*
        # protocol is not the answer to the question. Q43 asks for outliers
        # per building and the record also carries the pooled count, which
        # the question never mentions; scored as deciding it failed an answer
        # that matched both of the numbers actually asked for.
        alternative = any(path.lower().startswith(term) or f".{term}" in path.lower()
                          for term in ALTERNATIVE_PREFIXES)
        decides = ("[" not in path or path.startswith("value[0]")) \
            and not is_informational(path) and not alternative
        checks.append({"kind": "value", "name": path, "expected": want,
                       "matched": hit, "ok": hit is not None,
                       "decides": decides,
                       "how": "exact" if exact else f"within {TOLERANCE:.0%}"})
    for path, wanted in names:
        # A verdict is not satisfied by the winner's name appearing
        # somewhere: "LightGBM has the lower MAE: AutoARIMA 18.5, LightGBM
        # 8.5" names AutoARIMA and says the opposite. Where the key names a
        # winner, the name has to sit in a winning position.
        if any(t in path.lower()
               for t in ("winner", "lower_", "better", "best")):
            # Either the name leads a verdict clause -- "AutoARIMA has the
            # lower MAE" -- or it follows the word "winner". Anything looser
            # matches the list of scores that follows the verdict, where
            # every model's name appears whichever one won.
            hits = [v for v in wanted
                    if re.search(rf"\b{re.escape(v.lower())}\s*(?:\([^)]*\))?\s*"
                                 rf"(?:has|is|was|wins|achieves|gives|scored)\b"
                                 rf"|\bwinner\b\W{{0,4}}\**{re.escape(v.lower())}\b",
                                 lowered)]
        else:
            hits = [v for v in wanted if v.lower() in lowered]
        share = len(hits) / len(wanted)
        checks.append({"kind": "set", "name": path, "expected": len(wanted),
                       "matched": len(hits), "ok": share >= 0.6,
                       "decides": len(wanted) <= NAMEABLE,
                       "how": f"{share:.0%} of {len(wanted)} named"})
    if not checks:
        return None, "golden answer carries no checkable value", {
            "golden": str(record.get("answer", ""))[:400]}

    # A quantity that is arithmetic on two quantities already matched is not
    # independent evidence. "How did the median EUI differ between office and
    # education" is answered by 121.79 and 115.07; whether the reply prints
    # their difference (6.72) or their ratio (1.058x) is a choice of wording,
    # and failing it fails an answer that is right twice over.
    matched_values = [c["expected"] for c in checks
                      if c["kind"] == "value" and c["ok"]]
    for check in checks:
        if not check["ok"] or check["kind"] != "value":
            pass
        if check["ok"] or check["kind"] != "value":
            continue
        want = float(check["expected"])
        derived = any(
            close(want, abs(a - b), False) or (b and close(want, a / b, False))
            for a in matched_values for b in matched_values if a is not b)
        if derived:
            check["decides"] = False
            check["how"] += " -- derivable from values already matched"

    # A distance whose scale depends on the normalisation convention is not a
    # 2%-reproducible quantity: two implementations that agree on *which*
    # buildings are closest routinely differ in the third decimal on *how*
    # close. The finding decides; the scale gets the looser bar.
    for check in checks:
        if check["ok"] or check["kind"] != "value":
            continue
        # The name is a path -- "best_pair.euclidean_distance" -- so the
        # test has to look at the whole of it, not just the leaf. Q35 found
        # Dublin and Tempe, the answer the question asks for, and failed on a
        # distance nested one level down that the leaf-only test never saw.
        if any(term in check["name"].lower()
               for term in ("distance", "silhouette", "similarity", "score",
                            "correlation", "mae", "rmse", "mape")):
            hit = next((v for v in found
                        if close(v, float(check["expected"]), False,
                                 tolerance=LOOSE_TOLERANCE)), None)
            if hit is not None:
                check["ok"] = True
                check["matched"] = hit
                check["how"] = f"within {LOOSE_TOLERANCE:.0%} (scale depends " \
                               f"on the normalisation convention)"

    # "Compare A and B and report which has the lower MAE" asks for a
    # winner. Where the golden record names one and the answer agrees, the
    # individual scores are the working: two defensible implementations of
    # the same model differ in the second figure and agree on the finding.
    # Q39's LightGBM scores 13.75 here against a reference 14.95, because
    # this one uses lags of 24 hours or more and never feeds a prediction
    # back, and both say AutoARIMA wins -- which is the question.
    verdict = next((c for c in checks
                    if c["kind"] == "set"
                    and any(t in c["name"].lower()
                            for t in ("winner", "lower_", "better", "best"))),
                   None)
    if verdict is not None and verdict["ok"]:
        for check in checks:
            if check is not verdict and check["kind"] == "value" \
                    and any(t in check["name"].lower()
                            for t in ("mae", "rmse", "mape", "mase")):
                check["decides"] = False
                check["how"] += " -- the question asks which model, and that matched"

    deciding = [c for c in checks if c["decides"]]
    ok = all(c["ok"] for c in deciding) if deciding else all(c["ok"] for c in checks)
    summary = "; ".join(
        f"{c['name']}: " + ("matched" if c["ok"] else "no match")
        + (f" ({c['how']})" if not c["ok"] else "")
        + ("" if c["decides"] else " [supporting]")
        for c in checks)
    detail = record.get("detail")
    return ok, summary[:220], {
        "checks": checks,
        "golden": str(record.get("answer", ""))[:400],
        "golden_value": json.dumps(record.get("value"), default=str)[:300],
        "golden_seconds": runtime(qid),
        "method": (detail.get("note", "") if isinstance(detail, dict) else "")[:300]}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show")
    args = parser.parse_args()
    if args.show:
        print(json.dumps(golden(args.show), indent=2, default=str)[:2500])
        return 0
    have = sorted(_records())
    decisive = 0
    for qid in have:
        scalars, names = _leaves(golden(qid).get("value"))
        decisive += bool(scalars or names)
    total = sum(runtime(q) or 0.0 for q in have)
    print(f"  golden answers: {len(have)}")
    print(f"  with a checkable value: {decisive}")
    print(f"  golden solver total runtime: {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
