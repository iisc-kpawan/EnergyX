# Review notes: Q01 to Q10

Categories 1 (data discovery and availability) and 2 (data quality and preprocessing). Every module matches its independent pandas check on BDG2's own CSV files, and two runs give the same output hash. Times are site-local standard time; a valid reading is a non-null value, so ZERO_MASKED rows are missing. Coverage bars are compared in integers (100 x valid hours >= bar x base hours), so a building exactly on a bar is never lost to float rounding.

## Q01: buildings in London

- Answer 68: Mouse 7, Robin 52, Shrew 9. BDG2's City column gives the same three sites and the same counts.
- Reading "London" as the UK gives 215 (Cardiff's 147 Lamb buildings added).

## Q02: building-use labels in the United States

- Answer 16 labels, from Education to Warehouse/storage, over 965 buildings at US sites. No US building lacks a label.
- The US sites are those whose BDG2 Country is a US state or "US" (Rat, Washington DC). Adding the five anonymous sites adds no new label, so the answer is the same either way.

## Q03: 12 consecutive months at 95% coverage

- Answer 1,455 buildings. A window's hours follow the local clock: the windows starting January and February 2016 have 8,784 hours, the rest 8,760. 1,117 members pass on the first window (calendar 2016).
- The stricter reading, every month at least 95% valid, gives 1,195 (260 fewer; it adds none). The EXG-03 reading, 8,760 consecutive valid hours with no gap at all, gives 285.

## Q04: office buildings eligible for weather normalization in 2016

- Answer 211 offices. "Those hours" is read as the building's valid electricity hours, so the temperature bar is 95% of them. 289 offices have a floor area and a 2016 reading; 211 pass the electricity bar.
- The temperature bar never binds: every member has a valid air temperature for at least 99.63% of its valid hours. Measuring temperature against the year's 8,784 hours instead gives the same 211.
- Every BDG2 building has a floor area above 0, so the floor-area filter removes none.

## Q05: buildings eligible for weather-normalized annual analysis

- Answer 1,520 buildings of 1,572 with a floor area and a reading. 1,344 qualify in 2016, 1,450 in 2017, 1,274 in both. Both bars must hold in the same year.
- Requiring 80% of the year's hours to have both a reading and a temperature (a joint bar) gives the same 1,520.

## Q06: duplicate timestamps for Bear_education_Alfredo

- Answer 0. The building has 17,490 rows, 17,490 distinct local timestamps and 17,490 distinct UTC timestamps, all valid, in one stream. The check reads the timestamp strings straight from BDG2's electricity.csv and finds no repeat.
- The answer can't be anything else: BDG2 is a wide file with one row per hour, so a duplicate can only come from a broken ingest. The question tests whether an agent checks rather than assumes.

## Q07: missing-hour percentage for Peacock_office_Annie

- Answer 0.0513% (9 missing hours of 17,544). The first valid reading is 2016-01-01 00:00 and the last is 2017-12-31 23:00, so the observed period is the whole snapshot and that reading choice doesn't matter. The building has no ZERO_MASKED rows, so the raw and cleaned files agree.
- The result is stored rounded to 6 decimals (0.0513, exact value 0.051300); the check compares it with an absolute tolerance of 5e-7, since a relative 1e-6 is tighter than the rounding for a number this small.

## Q08: education buildings with gaps longer than seven days

- Answer 167 of 617 education buildings (604 with a valid reading). Longest gaps run from 177 to 14,503 hours.
- Counting runs before the first or after the last reading adds 39 (206). Reading "longer than seven days" as at least 168 hours adds 2 (169). The EXG-08 version over all building types gives 488.

## Q09: more than 72 consecutive zero hours

- Answer 343 buildings of 1,006 with a zero reading. A run is consecutive clock hours; a missing hour ends it. Runs among members range from 73 to 17,544 hours (a building zero for the whole snapshot).
- Skipping missing hours inside a run (the EXG-09 rule) gives 448, a superset (105 added). The check confirms that every ZERO_MASKED hour is a raw zero and every raw zero is masked.

## Q10: hours imputed for Peacock_office_Annie

- Answer 2. The series has 9 missing hours in 3 gaps: two gaps of 1 hour are filled, and one gap of 7 hours is left unfilled.
- Filling the first six hours of every gap gives 8. With so few gaps the question separates the two readings but tests little else; a building with more gaps would make it more telling.
