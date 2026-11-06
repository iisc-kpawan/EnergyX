# Review notes: Q11 to Q20

Categories 3 (consumption and demand statistics) and 4 (temporal and load-profile analytics). Every module matches its independent pandas check on BDG2's cleaned CSV, and two runs give the same output hash. Times are site-local standard time; a valid reading is a non-null value.

## Q11: Rat_office_Adele, July 2016 total

- Answer 557,631.77 kWh from 744 valid hours out of 744. No hour is missing or masked.
- BDG2's uncleaned file gives the same total, so the cleaning choice doesn't matter here.

## Q12: Fox_office_Alice, peak day of 2017

- Answer 2017-06-22 with 9,705.87 kWh over 24 valid hours. The runner-up is 2017-08-22 with 9,610.29 kWh, 1.0% lower.
- The uncleaned file and the whole 2016 to 2017 snapshot give the same day. 363 of 365 days have 24 valid hours; a partial day can only lose rank, and neither partial day is near the top.

## Q13: US offices, highest 2017 peak-to-average ratio

- Answer: Bull_office_Efren, Bull_office_Ella, Bull_office_Marco, Bull_office_Sally, Bull_office_Trevor. 124 US offices, 112 meet the 90% rule. The fifth ratio is 13.48 and the sixth (Panther_office_Christin) 6.72.
- Dropping the coverage rule gives the same five.
- All five are Bull buildings whose maximum is one spike far above normal use (Bull_office_Sally: 4,336 kWh against a mean of 88.6 kWh). BDG2's cleaning left these spikes in. With the 99.9th percentile as the peak, only Bull_office_Sally stays in the top five; with the 99th percentile, the top five are Panther_office_Christin, Fox_office_Molly, Fox_office_Essie, Fox_office_Juana and Fox_office_Brandy. The answer measures meter spikes more than demand peaks.

## Q14: London building types with a month above 1.2 times the annual daily average

- Answer: Lodging/residential only (January 1.386, February 1.349, November 1.255, December 1.232). The next highest type is Office at 1.101 (February). 51 of 68 London buildings meet the 90% rule; no London building lacks a type.
- The 90% rule decides the answer. Without it, Healthcare (1.396), Office (1.733) and Technology/science (11.774) also qualify, because buildings with partial years (several Shrew offices have 576 to 600 valid hours, Mouse_science_Micheal 251) make some months look far above the year. Five Mouse buildings sit just under the rule at 6,950 to 7,104 valid hours (79% to 81%).

## Q15: largest 10% against smallest 50% by floor area, 2017

- Answer: largest 10% total 682,090,709.73 kWh (155 buildings), smallest 50% total 262,409,163.57 kWh (774 buildings), ratio 2.599. The cohort is 1,547 buildings with at least one valid 2017 reading; every one has a floor area. Cuts are 19,050.22 m2 and 5,238.6 m2 by linear interpolation.
- Taking the top 155 and bottom 774 buildings by rank gives the same groups.
- Applying the 90% rule, as other yearly measures do, keeps 1,350 buildings and gives a ratio of 2.404, 7.5% lower and outside the 2% tolerance. The definition pins no coverage rule, so the gold has none.

## Q16: Cockatoo_office_Alton, peak hour of 2017

- Answer 11 (123.55 kWh). The runner-up is 13 (122.29 kWh), 1.0% lower, and 12 is 122.20 kWh. The margin is small but the cleaned file, the uncleaned file and the whole 2016 to 2017 snapshot all give 11.
- Hour 11 means the reading that starts at 11:00.

## Q17: Bear_education_Alfredo, peak month of 2017

- Answer January (195.94 kWh from 737 of 744 hours). February is next at 175.43 kWh from 670 of 672 hours.
- The mean per valid hour and the uncleaned file both give January too.

## Q18: Cockatoo_office_Alton, weekday against weekend profile, 2017

- Answer: weekday mean 104.94 kWh, weekend mean 88.85 kWh, ratio 1.181, weekday peak hour 11, weekend peak hour 6, largest gap +42.30 kWh at hour 11 (weekday above weekend).
- The largest gap is reported with its sign, since the definition says the gap is weekday minus weekend. The older EXG-18 module reported its absolute value; here they are the same because weekday is higher at every hour.
- The weekend peak is close: hour 6 is 96.96 kWh and hour 5 is 95.78 kWh (1.2% lower). The weekday peak is 135.78 kWh at 11 against 133.72 at 12.
- Averaging over readings instead of over the 24 profile values gives 104.93 and 88.83 kWh, under 0.03% different. Using the whole 2016 to 2017 snapshot (the EXG-18 reading) gives 102.51 and 86.15 kWh, about 2.3% lower, with the same peak and gap hours.

## Q19: Berkeley (Bear), summer afternoon mean, 2017

- Answer: Bear_education_Bulah, Bear_education_Chad, Bear_education_Rebecca, Bear_education_Wilton, Bear_education_Zandra. 90 of 92 buildings meet the rule of 332 valid hours out of 368. The fifth mean is 762.08 kWh and the sixth (Bear_education_Patricia) 713.48 kWh.
- The two left out (Bear_assembly_Roxy with 328 hours, Bear_education_Deborah with 270) average 19.7 and 1.3 kWh, so dropping the rule changes nothing. Including the hour starting 18:00 also gives the same five.

## Q20: London education, weekday against weekend normalized profile gap, 2017

- Answer: Robin_education_Billi, Robin_education_Della, Robin_education_Derick, Robin_education_Jasper, Robin_education_Kristopher. All 22 London education buildings meet the 90% rule. The fifth gap is 0.1555 and the sixth (Robin_education_Zenia) 0.1514, a 2.7% margin.
- Without normalizing, the mean absolute kWh gap picks mostly different buildings (Derick, Derrick, Julius, Leslie, So). Over the whole 2016 to 2017 snapshot, Robin_education_Kiera replaces Robin_education_Billi.

## Definitions worth another look

- **Q13:** the ratio uses the single highest reading, so it ranks meter spikes. A high percentile (99th or 99.9th) as the peak, or a rule removing readings above some multiple of the median, would measure demand peaks. As written, the answer is correct but says little about the buildings.
- **Q15:** consider stating whether the 90% coverage rule applies. The shared conventions apply it to "yearly measures", and a 2017 total is one; applying it moves the ratio from 2.599 to 2.404.
