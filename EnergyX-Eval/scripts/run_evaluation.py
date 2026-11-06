#!/usr/bin/env python3
"""Ask the application all 50 questions and score each answer.

This is the whole evaluation in one command. It reads the question set, puts
each question to a running EnergyX service, grades the answer against the
golden answer with the same grader that produced the reported figures, and
writes both a per-question CSV and a summary.

    # against a service already running on this machine
    python EnergyX-Eval/scripts/run_evaluation.py --service http://127.0.0.1:8321

    # a subset, while checking the setup works
    python EnergyX-Eval/scripts/run_evaluation.py --ids Q01 Q02 Q03

    # score answers collected elsewhere, without asking the app again
    python EnergyX-Eval/scripts/run_evaluation.py --answers my_answers.csv

The answers file for the last form is a CSV with an ``id`` column and an
``answer`` column; anything else in it is ignored. That is the route to take
when the application runs somewhere this script cannot reach.

Exit status is 0 when every question was asked and scored, 1 otherwise. A
question the app fails to answer is recorded as incorrect rather than skipped,
because a run that quietly drops its failures reports a success rate for a
different set of questions than the one it claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from grade_gold50 import golden, grade, runtime as golden_seconds   # noqa: E402

QUESTIONS = ROOT / "questions" / "final50_questions.csv"
RESULTS = ROOT / "results"


def load_questions(ids: list[str] | None) -> list[dict]:
    rows = list(csv.DictReader(QUESTIONS.open()))
    if ids:
        wanted = {q.upper() for q in ids}
        rows = [r for r in rows if r["id"].upper() in wanted]
        missing = wanted - {r["id"].upper() for r in rows}
        if missing:
            raise SystemExit(f"no such question: {', '.join(sorted(missing))}")
    return rows


def ask(service: str, question: str, timeout: float) -> tuple[str, float, str]:
    """Put one question to the service. Returns (answer, seconds, error)."""
    body = json.dumps({"question": question}).encode()
    request = urllib.request.Request(
        service.rstrip("/") + "/ask", data=body,
        headers={"Content-Type": "application/json",
                 # the tunnel serves an interstitial to browsers without it
                 "ngrok-skip-browser-warning": "1"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return "", time.time() - started, f"HTTP {exc.code}"
    except Exception as exc:                                   # noqa: BLE001
        return "", time.time() - started, type(exc).__name__
    text = payload.get("answer") or payload.get("text") or ""
    return str(text), time.time() - started, ""


def load_answers(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    if not rows or "id" not in rows[0] or "answer" not in rows[0]:
        raise SystemExit(f"{path} needs an 'id' column and an 'answer' column")
    return {r["id"].upper(): r["answer"] or "" for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="http://127.0.0.1:8321",
                        help="base URL of a running EnergyX service")
    parser.add_argument("--answers", type=Path,
                        help="score a CSV of answers instead of asking the app")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="seconds to allow one question (default 300)")
    parser.add_argument("--out", default="evaluation",
                        help="basename for the files written to results/")
    args = parser.parse_args()

    questions = load_questions(args.ids)
    collected = load_answers(args.answers) if args.answers else None
    if collected is None:
        print(f"asking {len(questions)} question(s) of {args.service}")
    else:
        print(f"scoring {len(questions)} question(s) from {args.answers}")

    rows, failures = [], 0
    for n, q in enumerate(questions, 1):
        qid = q["id"]
        if collected is None:
            answer, seconds, error = ask(args.service, q["question"], args.timeout)
        else:
            answer, seconds, error = collected.get(qid.upper(), ""), 0.0, ""
            if qid.upper() not in collected:
                error = "no answer in file"

        correct, detail, _evidence = grade(qid, answer)
        if correct is None:                 # no golden answer -- not scoreable
            correct = False
            detail = detail or "no golden answer"
        if error:
            correct, detail = False, error
            failures += 1

        mark = "ok  " if correct else "FAIL"
        print(f"  [{n:2d}/{len(questions)}] {qid} {mark} {seconds:6.1f}s  {detail[:60]}")
        rows.append({
            "id": qid,
            "difficulty": q.get("difficulty", ""),
            "category": q.get("category", ""),
            "question": q["question"],
            "correct": int(bool(correct)),
            "detail": detail,
            "seconds": round(seconds, 2),
            "golden_seconds": golden_seconds(qid) or "",
            "golden_answer": json.dumps(golden(qid).get("value"), default=str)[:500],
            "app_answer": answer[:2000],
        })

    RESULTS.mkdir(exist_ok=True)
    per_question = RESULTS / f"{args.out}_per_question.csv"
    with per_question.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(r["correct"] for r in rows)
    by_level: dict[str, list[int]] = {}
    for r in rows:
        by_level.setdefault(r["difficulty"] or "--", []).append(r["correct"])

    lines = [f"# EnergyX evaluation — {len(rows)} questions", "",
             f"**{correct}/{len(rows)} correct ({100 * correct / len(rows):.0f}%)**",
             f"Total wall clock: {sum(r['seconds'] for r in rows):.0f}s", "",
             "| difficulty | correct | of | success |", "|---|---:|---:|---:|"]
    for level in sorted(by_level):
        got, n = sum(by_level[level]), len(by_level[level])
        lines.append(f"| {level} | {got} | {n} | {100 * got / n:.0f}% |")
    if failures:
        lines += ["", f"{failures} question(s) could not be asked; each is "
                      "counted as incorrect."]
    summary = RESULTS / f"{args.out}_summary.md"
    summary.write_text("\n".join(lines) + "\n")

    print(f"\n{correct}/{len(rows)} correct ({100 * correct / len(rows):.0f}%)")
    print(f"  {per_question}")
    print(f"  {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
