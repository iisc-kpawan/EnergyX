# EnergyX-Gold release files

EnergyX-Gold has 50 building-energy analytics tasks over BDG2: 10 analytical categories with 5 tasks each, one at each difficulty level L1 to L5. Every task has:
- a pinned definition;
- a deterministic reference implementation;
- an independent check of that implementation, recomputed from BDG2's own CSV files;
- the reference answer.

## Files

| File | Contents |
|---|---|
| `outputs/EnergyX-Gold.jsonl` | One object per task, with every field below and the reference evidence (rankings, per-building values, settings). |
| `outputs/EnergyX-Gold.csv` | The same without the evidence. List and dictionary fields are JSON text. |
| `outputs/EnergyX-Gold.xlsx` | The CSV as a spreadsheet. Two set answers are too long for an Excel cell and show only their count; the full lists are in the JSONL. |

## Fields

**Task metadata.** These follow the EnergyX-Seed survey form.

| Field | Meaning |
|---|---|
| `question_id`, `question` | Q01 to Q50, and the question as posed to a system |
| `category`, `secondary_categories`, `sub_task` | The primary category, related categories, and the specific analytical task |
| `difficulty_level` | L1 is a single lookup or calculation; L2 basic aggregation, filtering or ranking; L3 combines several operations or data sources; L4 spans several buildings, data sources or tools; L5 is multi-stage analysis, such as comparing models or methods across a population |
| `intended_user_roles`, `expected_output_format`, `required_data`, `temporal_scope` | Who asks it, what form the answer takes, which data it needs, and which period it covers |
| `knowledge_source`, `source_reference` | The kind of source that motivates the task, and the reference |
| `seed_origin`, `closest_seed_id`, `closest_seed_question` | Whether the task was derived from an EnergyX-Seed question (similarity 0.6 or more) or newly written, and its closest seed |

**Answer and grading.**

| Field | Meaning |
|---|---|
| `definition` | The pinned reading of the question: population, period, formula, method settings and tie-breaking. The shared conventions are listed below. |
| `answer_instructions`, `answer_type`, `output_type`, `unit` | What a system must report, and in what form |
| `grading_rule`, `relative_tolerance` | Continuous values are accepted within a 2% relative tolerance. Counts, labels, dates and set members must match exactly. |
| `reference_answer`, `reference_evidence` | The reference output and the numbers behind it |
| `reference_implementation`, `independent_check` | Paths of the module that computes the answer and of its independent test, both relative to the repository root |
| `execution_stack`, `data_version` | The libraries used, and the data snapshot (EnergyX-BDG2-v1) |

## Shared conventions

The full list is at the top of `data/gold50_questions.yaml`.

- **Time.** Times are site-local standard time, and "2017" means the calendar year 2017.
- **Valid readings.** A reading is valid when it isn't null. Zero readings that BDG2's cleaning removed count as missing.
- **Building types.** An office building has BDG2 `primaryspaceusage` "Office", and an education building has "Education".
- **Locations.** "United States" means the sites whose country is US. The five anonymous sites have no city or country. London is the Mouse, Robin and Shrew sites.
- **EUI.** EUI is the year's valid kWh divided by the floor area in m2.
- **Yearly measures.** A building qualifies for a yearly measure when at least 90% of the year's hours are valid.
- **Percentiles.** Percentiles use linear interpolation, and "top five" rankings break ties by building_id.

## Reproducing the answers

See the repository README. In short, run:

```sh
uv sync
uv run python run_gold.py
uv run pytest -q
uv run python scripts/export_release.py
```

Running the reference needs the EnergyX-BDG2-v1 snapshot built from BDG2. The independent checks need BDG2's CSV files and the site-annotated metadata.

## Review notes

`reviews/` records, for every task, the choices its answer rests on and how the answer changes under the main alternative reading.
