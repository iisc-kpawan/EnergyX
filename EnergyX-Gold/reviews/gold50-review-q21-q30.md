# Review notes: Q21 to Q30

Categories 5 (building performance and normalization) and 6 (weather and climate analytics). Every module agrees with its independent pandas check from BDG2's own CSVs (20 tests pass). Shared rules: 2017 is site-local standard time, a valid reading has a non-null value, the coverage bar is 8,760 x 0.90 = 7,884 valid hours, and valid floor area means `floor_area_m2 > 0`.

## Q21: Fox_office_Alice 2017 EUI

- Answer: 168.071710 kWh/m2 (3,160,891.03 kWh / 18,806.8 m2), from 8,758 of 8,760 valid hours.
- No coverage bar applies. BDG2's uncleaned file gives the same sum (168.071710), so the cleaning doesn't matter. Using `sqft` x 0.0929 instead of `sqm` gives 168.071469, a difference of 0.0001%.

## Q22: top five office EUIs

- Answer: Fox_office_Clayton, Hog_office_Darline, Hog_office_Gustavo, Hog_office_Lavon, Panther_office_Lois. The cohort has 262 offices. The fifth EUI is 1005.64 and the sixth is 746.81, so the cut is clear.
- The 90% coverage bar removes 32 of 294 offices. Without it the top five stay the same. Every office has a valid floor area.

## Q23: median EUI, office vs education

- Answer: office 121.7917 (262 buildings), education 115.0691 (501), difference 6.7226, ratio 1.0584.
- Without the coverage bar (294 and 589 buildings) the medians are 110.34 and 101.63, and the ratio is 1.086. Means reverse the sign: office 168.36, education 171.45.

## Q24: above the label's 90th percentile

- Answer: 134 buildings from 11 labels. Five labels are left out for having fewer than 10 buildings: Food sales and service (4), Religious worship (3), Services (9), Technology/science (5), Utility (3). Manufacturing/industrial has exactly 10 buildings and is kept. 15 cohort buildings have no label.
- Using `>=` instead of `>` gives 136 buildings. Dropping the size rule gives 139.

## Q25: newer vs older US education buildings

- Answer: newer (built 2007 or later) median 168.1956, 30 buildings; older median 123.9659, 204 buildings; difference 44.2298; ratio 1.3568; newer is higher. 124 buildings with no year built are left out.
- "US" is the site's country. In raw BDG2 this means the `Country` values US, Arizona, California, Florida, New Jersey, New York and Texas. One building was built in exactly 2007. Reading "10 years old or newer" as built after 2007 moves it to the older group: the newer median becomes 174.1623 (29 buildings) and the older 123.1802 (205). The newer group is still higher.

## Q26: London July 2017 mean temperature

- Answer: 18.899462 °C over 2,232 valid site-hours (744 per site).
- The Mouse, Robin and Shrew sites carry identical weather series in BDG2 (same mean, minimum and maximum), so the pooled mean and the mean of the three site means are the same. July uses UTC+0 standard time. Using British Summer Time clock hours (UTC+1) gives 18.900538.

## Q27: Pearson, Fox_office_Alice vs air temperature

- Answer: 0.455673 over 8,758 paired hours. Spearman on the same hours gives 0.469968.

## Q28: offices with |Pearson r| > 0.75 against air temperature

- Answer: 4 of 262 offices: Fox_office_Sheila, Hog_office_Terry, Panther_office_Lauretta, Rat_office_Annis. All four correlations are positive. The highest rejected |r| is 0.7368 (Hog_office_Elnora).
- The set depends on the threshold: 11 offices clear 0.70 and none clears 0.80. Using `>=` changes nothing. Spearman instead of Pearson gives 6 offices.

## Q29: Spearman, Fox_office_Alice vs dew point

- Answer: 0.045533 over 8,758 paired hours, with average ranks for ties (scipy `spearmanr`; the test ranks with pandas and applies Pearson). Pearson gives 0.043306.
- The value is close to zero, so the 2% tolerance is ±0.0009. That is tight, but the definition pins the statistic, so an agent that follows it should land inside.

## Q30: offices with |Spearman rho| > 0.75 against dew point

- Answer: 1 office, Hog_office_Terry (rho 0.8273). The next is Hog_office_Merilyn at 0.7415.
- The answer is a single building, and 0.70 would give 6 offices. Pearson on dew point gives an empty set at 0.75.
