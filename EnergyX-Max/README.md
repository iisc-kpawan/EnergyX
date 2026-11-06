# EnergyX-Max

EnergyX-Max is a large generated extension of EnergyX-Bench. It expands the EnergyX-Seed questions into several thousand questions over BDG2, and every question has an answer computed by a program.

The questions are not manually validated. Their answers have passed only the automatic checks described below. EnergyX-Max is released for future benchmarking and is not used in the evaluation reported in the paper.

`outputs/summary.md` gives the counts for this release by category, level and pipeline stage.

## Files

| File | Contents |
|---|---|
| `outputs/EnergyX-Max.jsonl`, `outputs/EnergyX-Max.csv` | One row per question (fields below) |
| `outputs/templates.jsonl` | Every template the Generator Agent wrote, with the Judge Agent's verdict and reason, and the status of its program |
| `outputs/summary.md` | Counts by category and level, and by pipeline stage |
| `outputs/validation_report.json` | The checks in `energyx_max/validate.py`: structure, exclusion, and re-running programs |
| `outputs/stages/` | Each stage's own records: seeds, generated and judged templates, program results, instances |
| `programs/` | The Coding Agent's program for each template with a working program |
| `config/max.yaml` | Models, targets, thresholds and paths |
| `data/` | The seed sheets, the EnergyX-Gold questions (for exclusion) and the site-annotated BDG2 metadata |
| `energyx_max/` | The code that produced all of the above |

## Fields

Each question keeps the attributes of the EnergyX-Seed questions:
- `category`, `difficulty_level`, `sub_task`;
- `intended_user_role`, `expected_output_format`, `required_data`;
- `knowledge_source`, `source_reference`, `assumptions`.

It also records how it was made:

| Field | Meaning |
|---|---|
| `answer`, `unit`, `answer_type` | The answer computed by the template's program |
| `template_id`, `template`, `slots` | The template and the slot values that produced this question |
| `seed_id`, `seed_source` | The seed question the template came from |
| `program` | Path of the program that computed the answer |
| `generator_model`, `judge_model`, `coder_model` | The model behind each agent |
| `validation` | Always "automatic checks only; not manually validated" |

## How it is made

The pipeline has five stages. Each stage writes its records to `outputs/stages/` and can be resumed. The models, targets and thresholds are set in `config/max.yaml`.

1. **Seeds** (`energyx_max/seeds.py`). The seeds come from three sources:
   - the EnergyX-Seed survey sheets;
   - the team sheet of 122 questions;
   - the advanced-category sheet of 24 questions.

   The example questions from the benchmark's design conversation that BDG2 can answer as written are added. Seeds are mapped to the 10 analytical categories, and near-duplicates (similarity of 0.92 or more) are dropped.
2. **Generator Agent** (`energyx_max/generate.py`). Each seed becomes up to two question templates:
   - one for the seed's own task;
   - one for a related task at another difficulty level, chosen to even out the levels within the category.

   A template varies only through named slots from a fixed vocabulary (`energyx_max/slots.py`), such as `{building_id}`, `{city}`, `{building_type}`, `{year}`, `{month}` and `{k}`.
3. **Judge Agent** (`energyx_max/judge.py`). A different model checks each template against a filled-in example. It asks five things:
   - Is it answerable from the data?
   - Does it have exactly one computable answer?
   - Does it use only allowed slots?
   - Is it unambiguous?
   - Do its category and level fit?

   The Judge accepts the template, fixes its wording, or rejects it.
4. **Coding Agent** (`energyx_max/coder.py`). For each accepted template, the Coding Agent writes `answer(data, **slots)` against a small data interface (`energyx_max/bdg2.py`). The program runs on three sampled slot combinations. If none gives a usable answer, the agent sees the error once and fixes the program. A template whose program still fails is dropped.
5. **Instantiation** (`energyx_max/instantiate.py`, `energyx_max/build.py`). Slots are filled with valid combinations drawn from the data. For example:
   - a building comes with a year in which at least 90% of its hours are valid;
   - a building type comes with a place that has at least three such buildings.

   The target is 120 questions per category and level, 6,000 in all. A question is kept only when all of these hold:
   - its program returns a usable answer, which must be serializable, not empty, and free of NaN or infinite values;
   - its text appears only once;
   - it is neither a copy nor a near-copy (similarity of 0.9 or more) of any EnergyX-Seed or EnergyX-Gold question.

   So EnergyX-Max contains no Seed or Gold question.

## Models

The three agents run as separate calls, each in its own context, on local models served by Ollama:

| Agent | Model |
|---|---|
| Generator | qwen3.5:9b |
| Judge | qwen2.5:14b |
| Coding | qwen3.5:9b |

## Data and conventions

The programs read BDG2's cleaned hourly electricity file, the site weather file, and the site-annotated building metadata. They follow the EnergyX-Gold conventions:
- times are site-local;
- a reading is valid when it isn't NaN;
- a yearly measure needs at least 90% of the year's hours valid, unless the question says otherwise;
- percentiles use linear interpolation;
- models use `random_state=0`.

## Known limits

- **No manual validation.** No answer has been checked by hand. An answer can be wrong in two ways even when its program runs:
  - the program may compute something other than what the question asks;
  - a template may admit more than one reasonable reading.
- **Uneven cells.** Some category and level cells have fewer templates than others, so their questions come from fewer distinct tasks.
- **Recommended check before use.** Before using EnergyX-Max for comparative benchmarking, manually verify a random sample of at least 20% of the questions and answers.

## Running

Requirements:
- Python 3.12 and [uv](https://docs.astral.sh/uv/);
- [Ollama](https://ollama.com) with the three models pulled (`ollama pull qwen3.5:9b`, `ollama pull qwen2.5:14b`);
- BDG2's CSV files.

The data goes under `data/bdg2/`:

```sh
mkdir -p data/bdg2
base=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
curl -L -o data/bdg2/electricity_cleaned.csv $base/meters/cleaned/electricity_cleaned.csv
curl -L -o data/bdg2/weather.csv             $base/weather/weather.csv
cp data/site_metadata/BDG2_metadata.csv data/bdg2/
shasum -a 256 -c data/SHA256SUMS
```

Then run:

```sh
uv sync
uv run python -m energyx_max.run_all            # every stage; rerun it to resume
uv run python -m energyx_max.validate --sample 0 # re-run every program and check the release files
uv run pytest -q
```

Each stage can also be run alone, as `uv run python -m energyx_max.<stage>`, where the stage is `seeds`, `generate`, `judge`, `coder`, `instantiate`, `build` or `validate`.

On a 16 GB Apple M4 laptop a full run takes about 15 hours of machine time. The agents' calls are the slow part, so keep the machine awake, for example with `caffeinate -i`.

## Data licence

BDG2 is licensed CC BY-SA 4.0. The questions and answers here are derived from it and shared under the same terms. Please cite: Miller et al. (2020), *The Building Data Genome Project 2*, Scientific Data 7, 368.
