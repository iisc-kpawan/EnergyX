# EnergyX-Gold

EnergyX-Gold has 50 building-energy analytics tasks (Q01 to Q50) over BDG2. This repository holds their metadata, pinned definitions, reference implementations, independent checks and reference answers. The release files are in `outputs/`; `outputs/README.md` describes every field.

Each task has:
- **A pinned definition** in `data/gold50_questions.yaml`. It fixes the population, period, formula, method settings and tie-breaking. The shared conventions are at the top of the file.
- **A reference module** at `energyx_gold/tasks/category_NN_<name>/QNN.py`. It computes the answer from the EnergyX-BDG2-v1 snapshot through DuckDB.
- **An independent check** in `tests/test_qNN.py`. It recomputes the answer from BDG2's own CSV files with pandas, without the snapshot, DuckDB or this package's helpers. Model questions call scikit-learn, statsforecast, LightGBM, tslearn or Tiny Time Mixers directly with the same settings.
- **A review note** in `reviews/`. It records the choices the answer rests on, and how the answer changes under the main alternative reading.

All 50 modules run and are deterministic, and all 115 tests pass.

## Requirements

- **Software.** Python 3.12 and [uv](https://docs.astral.sh/uv/).
- **Disk.** About 700 MB for the data.
- **Network.** Internet access the first time Q40 runs, to download Tiny Time Mixers (`ibm-granite/granite-timeseries-ttm-r2`) from Hugging Face. After that, `HF_HUB_OFFLINE=1` works.

## Quick start

```sh
uv sync
shasum -a 256 -c data/SHA256SUMS          # confirms the BDG2 files, once they are in place
uv run python run_gold.py                 # computes all 50 answers, about 2 min
uv run python -m energyx_gold.cli.validate_gold   # shape, determinism and data checks
uv run pytest -q                          # the independent checks, about 6 min
uv run python scripts/export_release.py   # writes outputs/EnergyX-Gold.{jsonl,csv,xlsx}
```

## Data

Put the files under `data/`, as copies or symlinks:

| Path | What it is | Where to get it |
|---|---|---|
| `data/bdg2/electricity_cleaned.csv` | Hourly kWh per building, after BDG2's cleaning | [BDG2](https://github.com/buds-lab/building-data-genome-project-2), `data/meters/cleaned/` |
| `data/bdg2/electricity.csv` | The same meters before cleaning | BDG2, `data/meters/raw/` |
| `data/bdg2/weather.csv` | Hourly weather per site | BDG2, `data/weather/` |
| `data/bdg2/BDG2_metadata.csv` | BDG2's `metadata.csv` with four site columns added: Country, City, Site name and UID | Included here as `data/site_metadata/BDG2_metadata.csv` |
| `data/snapshot/EnergyX-BDG2-v1/` | The EnergyX snapshot built from the four files above | Built by EnergyX's ingestion; see "The snapshot" |

BDG2's CSV files are Git LFS files. Fetch them from GitHub's media host:

```sh
mkdir -p data/bdg2
base=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
curl -L -o data/bdg2/electricity_cleaned.csv $base/meters/cleaned/electricity_cleaned.csv
curl -L -o data/bdg2/electricity.csv         $base/meters/raw/electricity.csv
curl -L -o data/bdg2/weather.csv             $base/weather/weather.csv
cp data/site_metadata/BDG2_metadata.csv data/bdg2/
```

## The snapshot

The snapshot stores timestamps in UTC with a constant offset per site. The reference modules convert them back to site-local standard time, which matches the timestamps in BDG2's CSV files.

The snapshot also marks the zero readings that BDG2's cleaning removed (`quality_flag` ZERO_MASKED).

These answers were computed on the snapshot with hash `a67993c5575995e11732984194159fbbe9cb0172bf67817779a4a57e79945eab`, which every result records under `provenance`. A rebuild with other library versions can write different Parquet bytes and so give a different hash. If yours differs, run `validate_gold` and `pytest` to see whether the answers still agree.

## Commands

| Command | Does | Writes |
|---|---|---|
| `python run_gold.py` | Runs every task | `outputs/gold_results.jsonl` |
| `python -m energyx_gold.cli.run_task Q01 [...]` | Runs the named tasks | the tasks' lines in `outputs/gold_results.jsonl` |
| `python -m energyx_gold.tasks.category_01_discovery.Q01` | Runs one task and prints its full output | nothing |
| `python -m energyx_gold.cli.validate_gold [Q01 ...]` | Checks each output's shape, runs it twice to check determinism, and checks the data views it uses | `outputs/validation_report.json` |
| `python scripts/import_gold50.py` | Rebuilds the manifest from the question list and `data/gold50_questions.yaml` | `data/task_manifest.jsonl` |
| `python scripts/export_release.py` | Writes the release files | `outputs/EnergyX-Gold.*` |

## Layout

```
config/paths.yaml                     where the data is
data/gold50_questions.yaml            the pinned definition and metadata of each task
data/task_manifest.jsonl              the manifest built from it
data/questions/                       the question list
data/seeds/                           the EnergyX-Seed sheets, used to record each task's closest seed
energyx_gold/core/                    manifest, schemas, runner, provenance
energyx_gold/data_access/             DuckDB views over the snapshot
energyx_gold/analytics/               shared series, forecasting and clustering helpers
energyx_gold/tasks/category_NN_*/     one module per task
tests/                                the independent checks
reviews/                              what each answer rests on
docs/                                 how a task is written and verified; the ML protocol
```

## Determinism

- **Threads.** DuckDB, LightGBM, BLAS, OpenMP and PyTorch each run on one thread.
- **Randomness.** Every model uses `random_state=0`.
- **Determinism hash.** The hash leaves out `provenance` and rounds floats.

## Data licence

BDG2 is licensed [CC BY-SA 4.0](https://github.com/buds-lab/building-data-genome-project-2/blob/master/LICENSE). The snapshot, the metadata file and the reference answers are derived from it. Share them under the same terms, and cite the BDG2 data descriptor (Miller et al., Scientific Data 7, 368, 2020).
