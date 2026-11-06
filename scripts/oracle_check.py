"""Grade the agent against SQL, on questions SQL can answer unambiguously.

The point is coverage without a human in the loop. Every question here is
generated from the catalog together with the answer, computed directly in
DuckDB, so a wrong answer is provable rather than a matter of taste. This is
deliberately the *factual* slice -- counts, listings, memberships, extremes --
which is where silent filters and wrong-entity routing show up, and exactly the
class of defect that manual spot-checks are worst at catching.

No analytical judgement is tested here -- only facts SQL can settle.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyx.access.db import connect                                # noqa: E402

BASE = "http://127.0.0.1:8321"


def build_cases(con, per_kind: int) -> list[dict]:
    """Questions paired with the SQL that answers them."""
    cases: list[dict] = []
    sites = con.execute("""SELECT site_id, site_name, city, country_canonical AS country,
                                  n_buildings FROM sites ORDER BY n_buildings DESC""").df()
    types = con.execute("""SELECT building_type_canonical AS t, COUNT(*) AS n
                           FROM buildings WHERE building_type_canonical IS NOT NULL
                           GROUP BY 1 ORDER BY 2 DESC""").df()
    countries = con.execute("""SELECT s.country_canonical AS c, COUNT(*) AS n
                               FROM buildings b JOIN sites s ON s.site_id=b.site_id
                               WHERE s.country_canonical IS NOT NULL
                               GROUP BY 1 ORDER BY 2 DESC""").df()
    cities = con.execute("""SELECT s.city AS city, COUNT(*) AS n
                            FROM buildings b JOIN sites s ON s.site_id=b.site_id
                            WHERE s.city IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""").df()

    # --- scalar counts over the catalog ---------------------------------
    cases.append({"kind": "count", "q": "How many buildings are in this dataset?",
                  "expect": int(con.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]),
                  "entity": "buildings"})
    cases.append({"kind": "count", "q": "How many sites are in this dataset?",
                  "expect": int(len(sites)), "entity": "sites"})
    cases.append({"kind": "count", "q": "How many countries are covered?",
                  "expect": int(len(countries)), "entity": "countries"})
    cases.append({"kind": "count", "q": "How many cities are covered?",
                  "expect": int(len(cities)), "entity": "cities"})

    for _, r in sites.head(per_kind).iterrows():
        cases.append({"kind": "count",
                      "q": f"How many buildings are at site {r.site_id}?",
                      "expect": int(r.n_buildings), "entity": "buildings"})
    for _, r in types.head(per_kind).iterrows():
        cases.append({"kind": "count",
                      "q": f"How many {r.t} buildings are there?",
                      "expect": int(r.n), "entity": "buildings"})
    for _, r in countries.head(per_kind).iterrows():
        cases.append({"kind": "count",
                      "q": f"How many buildings are in {r.c}?",
                      "expect": int(r.n), "entity": "buildings"})
    for _, r in cities.head(per_kind).iterrows():
        cases.append({"kind": "count",
                      "q": f"How many buildings are in {r.city}?",
                      "expect": int(r.n), "entity": "buildings"})

    # --- set membership --------------------------------------------------
    cases.append({"kind": "set", "q": "List all sites in this dataset.",
                  "expect": sorted(sites.site_id.tolist()), "column": "site_id"})
    cases.append({"kind": "set", "q": "Which countries does this dataset cover?",
                  "expect": sorted(countries.c.tolist()), "column": "country"})
    cases.append({"kind": "set", "q": "What building types are in this dataset?",
                  "expect": sorted(types.t.tolist()), "column": "building_type"})
    return cases


def ask(question: str, mode: str, timeout: int) -> dict:
    body = json.dumps({"question": question, "mode": mode}).encode()
    request = urllib.request.Request(f"{BASE}/ask", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as exc:
        return {"status": "REQUEST_FAILED", "answer": str(exc), "plan": [],
                "metrics": {}, "table": None, "evidence": []}


def numbers_in(text: str) -> set[int]:
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text or "")}


def grade(case: dict, answer: dict) -> tuple[str, str]:
    """PASS / FAIL / PARTIAL, with the reason."""
    table = answer.get("table") or {}
    rows, evidence = table.get("rows") or [], table.get("evidence") or {}

    if case["kind"] == "count":
        want = case["expect"]
        candidates = {table.get("total_rows"), evidence.get("matched_filter"),
                      evidence.get("buildings_in_catalog"),
                      evidence.get("buildings_metered")}
        candidates |= numbers_in(answer.get("answer", ""))
        if want in candidates:
            if table.get("total_rows") == want or want in numbers_in(answer.get("answer", "")):
                return "PASS", f"{want}"
            return "PARTIAL", f"{want} present only in evidence"
        return "FAIL", (f"expected {want}; answer carried "
                        f"{sorted(c for c in candidates if isinstance(c, int))[:6]}")

    want = set(case["expect"])
    column = case["column"]
    got = {str(r.get(column)) for r in rows if r.get(column) is not None}
    if not got:                                   # column may be named differently
        for alt in (column, column.replace("_id", ""), "name"):
            got = {str(r.get(alt)) for r in rows if r.get(alt) is not None}
            if got:
                break
    if got == want:
        return "PASS", f"{len(want)} exact"
    if got and got <= want:
        return "PARTIAL", f"{len(got)}/{len(want)} returned, all correct"
    if not got:
        return "FAIL", f"no {column} column in the answer (got {table.get('tool')})"
    return "FAIL", (f"{len(got & want)}/{len(want)} correct, "
                    f"{len(got - want)} not in the catalog")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single", "multi", "both"])
    parser.add_argument("--per-kind", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out", default="runs/oracle_check.json")
    args = parser.parse_args()

    cases = build_cases(connect(), args.per_kind)
    modes = ["single", "multi"] if args.mode == "both" else [args.mode]
    records, tally = [], {}

    for mode in modes:
        print(f"\n{'='*86}\n{mode} mode — {len(cases)} questions with SQL ground truth\n")
        for i, case in enumerate(cases, 1):
            answer = ask(case["q"], mode, args.timeout)
            verdict, detail = grade(case, answer)
            tally[(mode, verdict)] = tally.get((mode, verdict), 0) + 1
            records.append({**{k: v for k, v in case.items() if k != "expect"},
                            "expect": case["expect"], "mode": mode,
                            "verdict": verdict, "detail": detail,
                            "planner": answer.get("metrics", {}).get("planner"),
                            "tools": [s.get("tool") for s in answer.get("plan", [])]})
            mark = {"PASS": "  ok ", "PARTIAL": " ~~~ ", "FAIL": " FAIL"}[verdict]
            print(f"{mark} [{i:2d}/{len(cases)}] {case['q'][:52]:<52} {detail[:46]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\n{'='*86}")
    for mode in modes:
        total = sum(v for (m, _), v in tally.items() if m == mode)
        line = "   ".join(f"{k[1]} {v}" for k, v in sorted(tally.items()) if k[0] == mode)
        passed = tally.get((mode, "PASS"), 0)
        print(f"  {mode:7s} {passed}/{total} exact   ({line})")
    print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
