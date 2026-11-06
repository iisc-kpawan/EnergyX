# Writing and verifying a gold task

The procedure for turning a question into a verified gold answer. The worked examples are the EnergyX-GOLD draft modules in the sibling energyx-gold repository (`energyx_gold/tasks/category_01_discovery/EXG_01.py` and `tests/test_exg_01.py`); the `reuse` key in the settings file names the closest one.

## Sources for each question

- **The manifest line:** `get_task("Q01")` gives the question, its answer type, the tolerance, the answer format and the pinned `definition`.
- **The settings file:** `data/gold50_questions.yaml` holds the same definition with the shared conventions in its header (the default year, the EUI formula, the 90% coverage rule, night hours, and so on).
- **The model settings:** `docs/ml-protocol.md` for forecasting, anomaly and clustering questions.

The question list gives only wording, level and category, so every definition was chosen here. Anything a reviewer should confirm goes in `reviews/gold50-review.md`.

## The data

`GoldContext` views, all in DuckDB:

| View | Columns worth knowing |
|---|---|
| `buildings` | building_id, site_id, building_type_canonical, building_type_raw (BDG2 `primaryspaceusage`), floor_area_m2, year_built, occupants, industry, latitude, longitude, timezone, city, state_region, country_canonical |
| `sites` | site_id, site_name, city, state_region, country_canonical, timezone, utc_offset_hours |
| `building_sites` | buildings plus site_name, country_code, utc_offset_hours, weather_source_id |
| `building_attributes` | building_id, attribute_name (UID, date_opened, energystarscore, eui, leed_level, rating, site_eui, source_eui, sqft), attribute_value |
| `electricity` | timestamp_local, site_id, building_id, stream_id, value, quality_flag |
| `weather` | timestamp_local, site_id, air_temperature, dew_temperature, sea_level_pressure, wind_direction, wind_speed, cloud_coverage, precip_depth_1hr, precip_depth_6hr |
| `electricity_streams`, `data_quality`, `capabilities`, `weather_sources` | EnergyX's derived coverage, gap, quality and capability columns |
| `agg_daily`, `agg_monthly`, `agg_yearly`, `profile_daily`, `profile_weekly` | EnergyX's precomputed aggregates. For cross-checks only |

**Rules:**
- **Time** is site-local standard time (`timestamp_local`). It equals the timestamp in BDG2's CSV files, so "2017" means local 2017.
- **Readings:** an hour with no row in `electricity` is missing in BDG2's cleaned file. A `quality_flag = 'ZERO_MASKED'` row has a null value: the raw reading was 0 and BDG2's cleaning removed it. Both count as missing unless the question is about zeros.
- **Derived columns:** compute measures from readings and metadata. EnergyX's derived tables are for cross-checks only.
- **Knowledge graph:** `context.query_kg` runs SPARQL on EnergyX's graph, for questions that need relationships the tables lack.

## Steps

1. **Pin the reading.** Implement the definition in the manifest and the YAML. If it can't be implemented as written, make the smallest reasonable change and record it in the YAML and the review page.
2. **Write the module** at `energyx_gold/tasks/category_NN_<name>/QNN.py`.
   - Copy the shape of the EXG module named by `reuse`: a docstring with the question and the pinned definition, named constants, and `main(context=None) -> dict`.
   - End with the `if __name__ == "__main__"` block.
   - Put code shared inside a category in `_helpers.py` in the same folder.
   - Don't edit `core/`, `data_access/`, `analytics/`, `tests/conftest.py` or the manifest.
3. **Write an independent check** in `tests/test_qNN.py`.
   - Compute the answer from BDG2's own files with pandas, through the `raw` fixture: `raw.electricity_cleaned`, `raw.electricity`, `raw.weather` and `raw.metadata`. Don't use the snapshot, DuckDB or `energyx_gold.analytics` for it.
   - For a model question, call LightGBM, statsmodels, scikit-learn, tslearn, ruptures or Tiny Time Mixers directly, restating the same settings.
   - Compare with `gold("QNN")`: sets exactly, numbers to about 1e-6 relative, since results are rounded to 6 decimals.
   - A difference means the two computations disagree about the definition. Find out why, and fix the one that is wrong.
4. **Run:**
   ```sh
   python -m energyx_gold.tasks.category_01_discovery.Q01          # prints the output
   pytest tests/test_q01.py -q
   python -m energyx_gold.cli.validate_gold Q01                # shape, determinism, data check
   ```
   When several people are working at once, run `run_task`, `run_all` and `run_gold.py` one at a time. Each rewrites `outputs/gold_results.jsonl`.
5. **Record what the answer rests on** in `reviews/gold50-review.md`: the choice made, and how the answer changes under the alternative, with numbers.

## Result shapes

| answer_type | result | notes |
|---|---|---|
| numeric | `{"answer": 557631.77, "unit": "kWh"}` | Counts as int. Round other numbers to 6 decimals; `answer_format` says what an agent should report |
| set | `{"answer": [sorted, unique str], "count": n}` | Members are the `key` in the answer format (building_id, site_id, and so on) |
| scalar_text | `{"answer": "Healthcare"}` | The exact label from the data |
| record | `{"answer": {"mae": 1.2, ...}, "units": {"mae": "kWh"}}` | Several named values, compared one by one |
| refusal | `{"refused": true, "predicate": "..."}` | `evidence.absence_check = {"query": ..., "rows": 0}` |
| method | `{"answer": null, "method": "..."}` | Scored against a rubric |

**Evidence:** always include the settings used, the supporting numbers a reviewer would want (rankings, per-building values, silhouette scores), and `queries: list(context.queries_run)` last.

## Status

- **`gold_verified`:** the module and the independent check agree, and the question has one clear reading.
- **`gold_candidate`:** anything else. Put a one-line reason in `status_note` and the detail in the review page.

## Writing style

For review notes and docstrings:
- plain words and short sentences;
- no em dashes;
- no filler and no hedging;
- sentence-case headings;
- numbers rather than adjectives;
- no personal names.
