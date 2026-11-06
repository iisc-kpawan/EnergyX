# Review: Q31 to Q35 and Q41 to Q45

Modules are in `energyx_gold/tasks/category_07_spatial/` and `energyx_gold/tasks/category_09_anomaly/`. Each has an independent pandas check in `tests/test_qNN.py`, and all ten checks pass. The raw metadata's labels are mapped to the site metadata as follows, and each test checks the mapping against the snapshot's `sites` table:
- Country: the US state names and "US" map to US, and "Anonymous" maps to no country.
- City: "Ottawa, Ontario" maps to Ottawa, and "Anonymous" maps to no city.

## Q31: buildings per country
- Answer: Canada 20, Ireland 36, UK 215, US 965, no country recorded 400 (1,636 in total).
- The answer rests on counting every building in the metadata. If only buildings with at least one valid reading count, the numbers are Canada 18, Ireland 36, UK 214, US 923 and no country 381.

## Q32: site with the most education buildings
- Answer: Rat, with 111 education buildings. Lamb is second with 83.
- Counting only buildings with readings gives the same counts at the top, so the answer is stable.

## Q33: city with the highest median office EUI in 2017
- Answer: Washington DC, with a median of 181.26 kWh/m2 over 22 offices. Orlando is second at 174.42 over 24 offices.
- Nine cities qualify and none has fewer than 5 offices. All London offices that qualify are at Robin.
- The answer rests on the 90% coverage rule. Without it (any 2017 reading), Orlando wins at 174.42 and Washington DC falls to 171.95 over 24 offices, so the answer flips.
- Changing the city minimum to 3 does not change the answer.

## Q34: weekday office profiles by country, 2017
- Answer:
  - Ireland: peak at 12:00, night-to-day ratio 0.1428, 9 offices.
  - UK: peak at 13:00, ratio 0.1170, 27 offices.
  - US: peak at 13:00, ratio 0.2623, 112 offices.
- Canada has no office buildings; Crow and Moose are all education. Six of the 262 qualifying office profiles are flat and become all zeros. That count includes the offices at anonymous sites.
- Without the 90% rule, the UK peak moves to 14:00 and its ratio to 0.1414 (41 offices). The US ratio becomes 0.2654.
- Over 2016 to 2017, as the older EXG-39 computed it, the ratios are Ireland 0.1304, UK 0.1362 and US 0.2616. The peak hours are the same.
- The UK peak hour depends on the coverage rule, and the grading tolerance on it is 0.

## Q35: most similar cross-country city pair
- Answer: `Cardiff|Washington DC`, with r = 0.989543. The runner-up is `Dublin|Princeton` at 0.986980.
- Grouping by city merges the three London sites into one city. Only Robin contributes offices.
- A minimum of 5 offices per city instead of 3 gives the same result.
- Grouping by site gives the same pair as `Lamb|Rat`.
- Without the 90% rule, `Dublin|Princeton` wins at 0.985890. The gap at the top is 0.0026 under the pinned reading.

## Q41: Fox_office_Alice above mean + 3 sd
- Answer: 0 readings. The threshold is 477.80 kWh and the highest reading is 467.96 kWh, 9.8 kWh below it. There are 8,758 valid hours.
- Using ddof 0 also gives 0. A two-sided test gives 23.
- An answer of 0 is easy to guess, so the question has little power to tell good answers from bad ones.

## Q42: Fox_office_Alice above mean + 2 sd
- Answer: 209 readings, above a threshold of 438.835 kWh.
- The nearest reading is 0.005 kWh from the threshold, and the snapshot and the raw CSV agree on it.
- Using ddof 0 also gives 209. A two-sided test gives 241.

## Q43: IQR outliers at London offices in 2017
- Answer: 2,268 outlier readings (635 low and 1,633 high) on 201 days.
- The cohort is 25 offices, all at Robin. Robin_office_Erma has no 2017 reading and is left out.
- The answer rests on per-building fences. With one set of fences pooled over all the buildings, there are 0 outliers.
- Counting only high outliers gives 1,633 readings on 183 days.
- On BDG2's uncleaned `electricity.csv`, there are 9,161 outliers on 237 days, because the zeros are still in that file.

## Q44: day-over-day increases above 20% at Cockatoo_office_Alton
- Answer: 54 days out of 648 compared pairs of complete days. Most of them are Mondays after a weekend.
- The nearest ratio to the threshold is 1.199397, so using ">=" instead of ">" also gives 54.
- Other readings of which days count change the answer:

  | Reading | Days |
  |---|---|
  | Pinned: 24 valid readings, compared with the previous calendar day | 54 |
  | Any day with a valid reading | 65 |
  | Days with at least 20 valid readings | 57 |
  | Compared with the previous complete day, skipping gaps | 58 |

## Q45: Isolation Forest at Berkeley education buildings, May 2016
- Answer: Bear_education_Chun, Darrell, Shelley, Yuri and Yvette. There are 65 buildings with readings; Benjamin and Carlo have none.
- Fifth place is fragile: Yvette has 584 anomalous hours and Leonel has 583.
- The contamination is "auto", which labels 62.7% of all hours as anomalous, so the ranking separates buildings by small differences in a large count.
- The random seed changes the set:
  - Seed 1 gives Shelley, Yuri, Darrell, Leonel and Owen.
  - Seed 2 gives Shelley, Darrell, Yuri, Augusta and Leonel.
- The answer is deterministic at random_state 0, and scikit-learn's version is recorded in the evidence. I suggest `gold_candidate`.

## Definitions worth changing
- **Q45:** with contamination "auto", most hours are labelled anomalous, and the top five change with the seed. Two options:
  - Fix the contamination, for example at 0.01, and rank by the share of anomalous hours.
  - Ask for the single building with the most anomalies, if its lead holds across seeds. Shelley leads under seeds 0 to 2.
- **Q41:** the answer is 0. A two-sided test (23), or Q42 alone, would be a stronger question.
- **Q33 and Q34:** both answers depend on the 90% coverage rule, which the question text doesn't state. That is acceptable because the rule is a shared convention, but a reviewer should know that the answer flips without it.
