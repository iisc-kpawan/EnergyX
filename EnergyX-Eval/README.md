# EnergyX-Eval

Everything needed to reproduce the reported evaluation: the question set, the
answer of record for each question, the grader that decides right from wrong,
and the results we obtained.

You can reproduce the scoring without a GPU and without the dataset — that path
is `--answers` below. Reproducing the *answers* needs a running application,
which `../README.md` explains how to build.

The set is balanced by construction: **10 categories × 5 questions**, and
**5 difficulty levels × 10 questions**, so neither a category nor a level can
dominate the headline figure.

## What is here

```
questions/
  final50_questions.csv      the 50 questions: id, difficulty, category, text
  final50_questions.xlsx     the same, for reading
  golden_answers_50.json     the answer of record for each question
scripts/
  run_evaluation.py          ask the app all 50 and score them
  grade_gold50.py            the grader: one answer against its golden answer
  grading_rules.py           the tolerance policy, and why each rule exists
results/
  five_model_summary.md      what five planner models scored
  per_question_matrix.csv    which model got which question right
```

## Reproducing the evaluation

With a service running (see the root README §3):

```bash
python EnergyX-Eval/scripts/run_evaluation.py --service http://127.0.0.1:8321
```

It asks all 50 questions in order, scores each answer as it arrives, and writes
`results/evaluation_per_question.csv` and `results/evaluation_summary.md`.
Expect roughly 8–20 minutes depending on the planner model; the first question
is slower while the model weights load.

Check the setup on three questions first:

```bash
python EnergyX-Eval/scripts/run_evaluation.py --ids Q01 Q02 Q03
```

If the application runs somewhere this script cannot reach, collect the answers
into a CSV with an `id` column and an `answer` column and score them offline:

```bash
python EnergyX-Eval/scripts/run_evaluation.py --answers my_answers.csv
```

Inspect a single golden answer:

```bash
python EnergyX-Eval/scripts/grade_gold50.py --show Q44
```

## How an answer is judged

The golden record for each question carries a `value` — the answer itself, as a
number, a set of named numbers, or a list of named things. The grader compares
the application's prose against that value. It never has to guess which figure
in the prose is the headline, because `value` says.

Four rules decide the comparison, and each exists because a stricter reading
failed an answer that was right. They are stated once, in
`scripts/grading_rules.py`:

| Rule | Value | Why |
|---|---|---|
| Continuous quantities | within **2%** | a mean or a fitted slope moves with implementation detail; the finding does not |
| Counts, indices, cluster numbers | **exact** | being close to a count is not being right |
| Set answers | names **most** of the set, up to **25** | beyond 25 the question is answered by the count and a table; requiring recitation measured summary length, not correctness |
| Method-quality figures (e.g. silhouette) | **reported, not decisive** | a silhouette moves with the normalisation convention, so it would fail an answer that agrees on the finding and differs on a protocol choice |

Clock times are stripped before figures are extracted: `nighttime
(20:00–06:59)` otherwise parses as 20, 0, −6 and 59, and each became a number
the answer was expected to recite.

A question the application fails to answer is scored **incorrect**, not
skipped. A run that drops its failures reports a success rate for a different
set of questions than the one it claims.

## Reading the results

`evaluation_per_question.csv` has one row per question:

| column | meaning |
|---|---|
| `correct` | 1 or 0, the grader's verdict |
| `detail` | why — `value: matched`, `value: no match (exact)`, or the error |
| `seconds` | wall clock for that question |
| `golden_seconds` | what the reference implementation took on the same question |
| `golden_answer` | the value the answer was compared against |
| `app_answer` | what the application said |

`evaluation_summary.md` gives the headline: overall success, and a breakdown by
difficulty level L1–L5. Success rate is the count of `correct` over the number
of questions asked — there is no partial credit, and no question is excluded.

The figure to compare against is in `results/five_model_summary.md`: five
planner models on this same set, scoring between 25/50 and 38/50.

## Two reference implementations

`../EnergyX-Gold/` holds a second, independent implementation of the same 50
questions, with its own pinned definitions. The two agree everywhere except
where the pinned convention itself differs, which is worth knowing before
comparing a score against either:

| | This set | EnergyX-Gold | Why |
|---|---|---|---|
| Q24 | 139 | 134 | the question here evaluates **every** use category; EnergyX-Gold drops categories with fewer than 10 buildings |
| Q36 | 0.1417 | 0.225 | the question here pins a **24-hour** season for the seasonal-naive forecast; EnergyX-Gold pins 168 |

Neither is an error. Each answer is correct for the question as that set states
it, which is the point: a protocol named without its settings is not one
answer but a family of them. The figures for both conventions are carried in
the `detail` of each question here.

Q45 is a third case and a weaker one. It asks for Isolation Forest results, and
the two sets disagree on which buildings are most anomalous. Re-running the
stated method reproduces neither list exactly, because the result moves with
feature scaling and library version. Treat that question's answer as
implementation-bound rather than settled.

## Where the golden answers came from

Each was computed independently of the application, in pandas and SQL directly
against the snapshot, and reviewed before being fixed as the answer of record.
Where a question could be read more than one way — which buildings a comparison
includes, whether a threshold is inclusive — the convention is stated in the
question text itself, so the question is answerable only one way.

`golden_answers_50.json` carries, per question: `value`, the answer the grader
compares against; `detail`, the working behind it; `answer`, the prose form;
and `_seconds`, how long the reference implementation took.

Reference runtime for the whole set is 108.9 s, which is the number the
`golden_seconds` column compares against.
