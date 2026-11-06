# EnergyX 50 — five-planner comparison

One application, five planners. Identical app code, question set, golden answers, grader and harness across every run; the planner model is the only variable.

Snapshot: BDG2, 1,578 buildings. Strict LLM planning (`ENERGYX_REQUIRE_LLM=1`) throughout — no rule-based fallback, so a planning failure is a lost question rather than a silent downgrade.

## 1. Headline

|                           | qwen3.5-9b    | qwen3.5-27b   | deepseek-r1-14b   | gemini-3.8-flash   | gemini-3.5-flash-lite   |
|:--------------------------|:--------------|:--------------|:------------------|:-------------------|:------------------------|
| task success              | 38/50 = 76.0% | 38/50 = 76.0% | 25/50 = 50.0%     | 38/50 = 76.0%      | 35/50 = 70.0%           |
| correct                   | 38            | 38            | 25                | 38                 | 35                      |
| prompt tokens             | 745680        | 787569        | 1286918           | 796683             | 827669                  |
| completion tokens         | 32237         | 35419         | 96104             | 272811             | 26480                   |
| total tokens              | 777917        | 822988        | 1383022           | 1069494            | 854149                  |
| total time (seconds)      | 450           | 1204          | 1766              | 861                | 222                     |
| planning time (seconds)   | 365           | 1106          | 1607              | 779                | 144                     |
| tool time (seconds)       | 75            | 66            | 75                | 74                 | 72                      |
| model calls               | 57            | 58            | 95                | 57                 | 59                      |
| tool steps                | 150           | 144           | 217               | 119                | 141                     |
| mean steps / question     | 3.0           | 2.88          | 4.34              | 2.38               | 2.82                    |
| tokens per correct answer | 20472         | 21658         | 55321             | 28145              | 24404                   |

## 2. By difficulty

|    | qwen3.5-9b   | qwen3.5-27b   | deepseek-r1-14b   | gemini-3.8-flash   | gemini-3.5-flash-lite   |
|:---|:-------------|:--------------|:------------------|:-------------------|:------------------------|
| L1 | 10/10        | 9/10          | 5/10              | 10/10              | 10/10                   |
| L2 | 6/10         | 8/10          | 5/10              | 8/10               | 7/10                    |
| L3 | 9/10         | 8/10          | 5/10              | 7/10               | 6/10                    |
| L4 | 7/10         | 6/10          | 6/10              | 7/10               | 7/10                    |
| L5 | 6/10         | 7/10          | 4/10              | 6/10               | 5/10                    |

## 3. By category

|                                      | qwen3.5-9b   | qwen3.5-27b   | deepseek-r1-14b   | gemini-3.8-flash   | gemini-3.5-flash-lite   |
|:-------------------------------------|:-------------|:--------------|:------------------|:-------------------|:------------------------|
| Anomaly & Change Detection           | 5/5          | 5/5           | 2/5               | 5/5                | 5/5                     |
| Building Performance & Normalization | 4/5          | 4/5           | 3/5               | 3/5                | 3/5                     |
| Clustering & Similarity Analytics    | 3/5          | 3/5           | 1/5               | 3/5                | 2/5                     |
| Consumption & Demand Statistics      | 2/5          | 3/5           | 3/5               | 2/5                | 2/5                     |
| Data Discovery & Availability        | 4/5          | 4/5           | 5/5               | 5/5                | 5/5                     |
| Data Quality & Preprocessing         | 5/5          | 5/5           | 3/5               | 5/5                | 5/5                     |
| Forecasting & Predictive Analytics   | 3/5          | 3/5           | 2/5               | 4/5                | 3/5                     |
| Spatial & Spatio-Temporal Analytics  | 4/5          | 3/5           | 1/5               | 2/5                | 1/5                     |
| Temporal & Load-Profile Analytics    | 4/5          | 3/5           | 2/5               | 4/5                | 4/5                     |
| Weather & Climate Analytics          | 4/5          | 5/5           | 3/5               | 5/5                | 5/5                     |

## 4. Agreement

- solved by all 5: **18** — Q01, Q11, Q21, Q36, Q41, Q02, Q17, Q03, Q08, Q18, Q28, Q09, Q24, Q29, Q39, Q44, Q10, Q50

- solved by none: **8** — Q22, Q47, Q38, Q14, Q34, Q49, Q15, Q20

A question no planner solves is an application finding, not a planner one: the reading, the routing or the tool is what failed, and changing the model will not move it.
