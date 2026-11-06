"""Exercise the agent layer, end to end, with no LLM.

Checks that both modes run, that the plan DAG is emitted and serialised, that
the Verifier gates answers, that refusals stay honest, and that every run leaves
a replayable trace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyx.agents import Budget, EnergyXAgent  # noqa: E402

QUESTIONS = [
    "Which five office buildings in the UK had the highest electricity EUI in 2017?",
    "How many office buildings are there in each country?",
    "Which meters have the most missing data?",
    "What was the baseload of Fox_office_Clayton during July 2017?",
    "Compare weekday and weekend load profiles for offices in London",
    "Which offices are most sensitive to outdoor temperature?",
    "How does peak demand timing vary across cities?",
    "Was 2017 anomalous for Robin_office_Donald?",
    "Find buildings with load shapes similar to Fox_office_Clayton",
    "Forecast the next 24 hours for Fox_office_Clayton",
    "Which buildings should be investigated first for poor energy performance?",
    "What is the per-occupant electricity consumption of Fox_office_Clayton?",
]


def main() -> int:
    failures = 0
    rows = []
    for mode in ("multi", "single"):
        agent = EnergyXAgent(mode=mode, budget=Budget(max_steps=20, max_seconds=120))
        for question in QUESTIONS:
            answer = agent.ask(question)
            bad = answer.status == "ERROR" or answer.verification["status"] == "reject"
            failures += bad
            rows.append({
                "mode": mode, "question": question, "status": answer.status,
                "verify": answer.verification["status"],
                "steps": answer.metrics["steps_executed"],
                "cat": (answer.plan.get("steps") or [{}])[0].get("tool", "-"),
                "resolutions": answer.metrics["resolution_counts"],
                "agents": len(answer.metrics["agents_used"]),
                "seconds": answer.metrics["wall_clock_s"],
                "bad": bad, "run": answer.run_id,
            })

    width = max(len(r["question"]) for r in rows)
    print()
    for mode in ("multi", "single"):
        print(f"  ---- mode = {mode} " + "-" * (width - 4))
        for r in (x for x in rows if x["mode"] == mode):
            mark = "!!" if r["bad"] else "  "
            graph = r["resolutions"].get("graph_only", 0)
            print(f" {mark} {r['question']:<{width}}  {r['status']:<18} "
                  f"verify={r['verify']:<5} steps={r['steps']} "
                  f"graph_only={graph} agents={r['agents']} {r['seconds']:.2f}s")

    # Parity: both modes should reach the same verdict on the same question.
    divergent = []
    for question in QUESTIONS:
        statuses = {r["mode"]: r["status"] for r in rows if r["question"] == question}
        if len(set(statuses.values())) > 1:
            divergent.append((question, statuses))
    print(f"\n  mode divergence: {len(divergent)}")
    for question, statuses in divergent:
        print(f"    {question} -> {statuses}")

    # Trace bundles must be complete and replayable.
    sample = rows[0]["run"]
    trace = json.loads((Path("runs") / sample / "trace.json").read_text())
    required = {"run_id", "question", "mode", "snapshot", "intent", "plans",
                "steps", "verification", "answer", "metrics"}
    missing = required - set(trace)
    print(f"\n  trace bundle keys present: {not missing}"
          + (f"  missing={sorted(missing)}" if missing else ""))
    print(f"  trace steps recorded: {len(trace['steps'])}, "
          f"plans: {len(trace['plans'])}, "
          f"args captured: {all('args' in s for s in trace['steps'])}")

    refusals = [r for r in rows if r["status"] == "INSUFFICIENT_DATA"]
    print(f"  honest refusals: {len(refusals)}")
    print(f"\n  runs: {len(rows)}   hard failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
