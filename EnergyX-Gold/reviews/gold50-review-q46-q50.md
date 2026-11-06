# Review notes: Q46 to Q50 (clustering and similarity)

Modules: `energyx_gold/tasks/category_10_clustering/Q46.py` to `Q50.py`, shared code in `_helpers.py`. Independent checks: `tests/test_q46.py` to `tests/test_q50.py`, all passing. Library versions: scikit-learn 1.7.2, tslearn 0.9.0, numpy 2.5.3.

## Q46: K-Means day clusters for Eagle_office_Lamont, 2017

Answer: cluster sizes 131, 104, 95 (330 complete days of 365).

Choices the answer rests on:
- A day is complete when all 24 local hours have a valid reading. Masked zeros count as missing.
- Days are clustered in raw kWh, as pinned. No scaling.

Alternatives:
- Each day min-max scaled to 0-1 first: 227, 54, 49. The raw-kWh clusters mostly split days by load level; the scaled ones split by shape. This is the main reading an agent might take, since "load profile" often implies shape. Both are in the evidence.
- random_state 1, 2 or 3, or n_init 1: 131, 104, 95 each time. The answer is stable to the seed.

## Q47: five office buildings most similar to Eagle_office_Lamont, 2017

Answer: Eagle_office_Mandi, Eagle_office_Marisela, Eagle_office_Nereida, Eagle_office_Phyllis, Eagle_office_Remedios.

Choices the answer rests on:
- Candidates: 261 of the 296 office buildings with readings have at least 7,884 valid 2017 hours. The 90% rule is not applied to Eagle_office_Lamont itself (it has 8,268, so this does not matter).
- Euclidean distance between min-max scaled 24-hour profiles.

Alternatives:
- The 5th and 6th distances are 0.507899 (Eagle_office_Phyllis) and 0.511101 (Hog_office_Lavon), a gap of 0.0032. Small changes to the profile would swap them.
- Pearson ranking top five: Remedios, Marisela, Nereida, Panther_office_Hannah, Hog_office_Lavon. Three of five overlap with the Euclidean answer.
- Without the 90% rule (293 candidates with a reading in every hour): Remedios, Amanda, Nereida, Marisela, Flossie. Two buildings change.

## Q48: best Ward k for London office profiles

Answer: 2 clusters (silhouette 0.366312; sizes 15 and 11; 26 buildings).

Choices the answer rests on:
- Cohort: London (Mouse, Robin, Shrew) office buildings with any valid reading, 26 buildings, all at Robin and Shrew.
- Profile: mean valid reading per local hour over 2016-2017, min-max scaled.

Alternatives:
- Runner-up k 3 scores 0.350352, a margin of 0.016. k 9 and 10 score 0.223 and 0.203, so a range of 2 to 10 gives the same answer.
- A 2017-only profile cannot be built as written: Robin_office_Erma has an hour of day with no valid 2017 reading.

## Q49: K-Means against Ward for London office profiles

Answer: K-Means k 2, silhouette 0.365162; Hierarchical k 2, silhouette 0.366312; better method Hierarchical.

Choices the answer rests on:
- Both methods scored by the scikit-learn Euclidean silhouette on the same 26 profiles.
- The better method is decided by a difference of 0.00115 (0.3%).

Alternatives:
- K-Means with random_state 1 to 5 gives k 2 and 0.365162 each time, so the K-Means score is stable. The verdict holds under these settings but rests on a very small gap.
- The two k 2 partitions differ by one building (16/10 against 15/11; adjusted Rand index 0.846).
- The graded silhouette scores have a 2% tolerance, which is wider than the gap between them. An agent reporting both scores within tolerance can still name the wrong method. The `better_method` field decides the question.

## Q50: DTW TimeSeriesKMeans clusters of London education profiles

Answer (sizes 9, 7, 6; 22 buildings, all at Robin):
- Billi, Della, Derick, Derrick, Karyl, Kristopher, Leslie, Madeline, Terrance
- Jasper, Julius, Kiera, Lashandra, So, Takako, Zenia
- Audrea, Cecilia, Lizbeth, Margarito, Megan, Mercedes

(all `Robin_education_*`). DTW silhouette of this partition: 0.122309.

Choices the answer rests on:
- Cohort and profile as in Q48, for Education.
- tslearn 0.9.0 with the pinned settings. Clusters ordered by size, then by first member.

Alternatives:
- The partition is sensitive to the seed. With random_state 1 to 5 the sizes are 9/8/5, 11/10/1, 9/7/6, 11/7/4 and 11/6/5, with adjusted Rand index 0.29 to 0.72 against the answer. A different tslearn version could also change it.
- Euclidean K-Means with k 3: sizes 10/6/6, adjusted Rand index 0.21 against the answer.
- The low silhouette (0.12) says the profiles do not form three clear groups, which explains the instability.

## Definitions that may need changing

- Q50: exact membership with tolerance 0 is only reproducible with the same library version and seed. Consider grading cluster agreement (for example adjusted Rand index at least 0.8 against the gold partition) instead of exact lists, or mark the question `gold_candidate` with this reason.
- Q49: the verdict rests on a 0.00115 silhouette gap. Consider asking for the scores only, or state that the method comparison is at the scores' full precision.
- Q46: the question says "load profiles" and the definition clusters raw kWh. If shape is meant, the normalized sizes (227, 54, 49) should be the answer.
