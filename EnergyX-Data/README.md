# EnergyX-Data

What the system reads, where to get it, and how to turn it into the snapshot
the application queries.

The dataset is **not committed to this repository**. It is third-party, it is
large, and it is already published by its authors — redistributing it here
would add a second copy of record that could drift from theirs.

## The source dataset

**Building Data Genome Project 2 (BDG2)**

| | |
|---|---|
| Source | https://github.com/buds-lab/building-data-genome-project-2 |
| Citation | Miller et al., *Scientific Data*, 2020 |
| Licence | see the upstream repository — it governs the data, not this code |
| Content | non-residential building meters, 2016–2017, hourly |
| Meter channel read | **electricity only** |
| Also ingested | site weather, building metadata |

Electricity is the only *meter* channel read. The source carries chilled water,
steam, hot water, gas, water, irrigation and solar as well, and EnergyX
deliberately does not read them: about 180 MB of cleaned series sits unread in
`BDG2/`. Site weather is a different matter — it is ingested in full, because
weather sensitivity and weather normalisation are part of the analytical scope.

The refusal path depends on the distinction: a question about gas, or about
indoor conditions, is answered with a refusal naming the missing predicate,
while a question about outdoor temperature is answered from the weather table.

## Obtaining it

Clone the upstream repository, or download its release, and place these four
files in a directory named `BDG2/` at the root of this repository:

```
BDG2/
├── electricity_cleaned.csv     the series the system reads
├── electricity.csv             the raw series, for the reconciliation check
├── BDG2_metadata.csv           building metadata
└── weather.csv                 site weather
```

The paths are declared in `datasets/bdg2/energyx.yaml`, which is the schema
registry entry for this dataset. Nothing is downloaded at build time or at run
time; if `BDG2/` is absent the build stops and says so.

## Building the snapshot

```bash
# measurements, catalog tables and MANIFEST.json
./.venv/bin/python -c "from energyx.ingest.bdg2 import build_snapshot; print(build_snapshot)"

# daily, monthly and yearly rollups, and typical load profiles
./.venv/bin/python -c "from energyx.ingest.aggregates import build_aggregates; print(build_aggregates)"

# project the catalog into the RDF store
./.venv/bin/python scripts/build_graph.py

# confirm counts, coverage and provenance reconcile
./.venv/bin/python scripts/verify_snapshot.py
```

`verify_snapshot.py` is the one to run after any rebuild: it fails loudly
rather than reporting a partial snapshot as a whole one.

This produces `data/EnergyX-BDG2-v1/`, which is what the application queries.
It is derived data — deleting it costs only the time to rebuild.

## What the snapshot contains

| | |
|---|---|
| Buildings with metadata | 1,636 |
| Buildings with an electricity meter | 1,578 |
| Sites | 19 |
| Countries | 4 |
| Measurements | 26,372,337 hourly readings |
| Period | 2016-01-01 → 2017-12-31, local standard time |
| Layout | `dataset → site_id → year`, 37 partitions |

Two counts, both correct: 1,636 buildings have metadata and 1,578 have a meter.
The 58-building gap is BDG2's own. Answers state which one they used.

Six streams are flagged `electricity = Yes` but carry no data at all. They are
excluded by explicit predicate, never silently, and named in
`data/EnergyX-BDG2-v1/MANIFEST.json`.

`MANIFEST.json` is written by the build and records the source directory, the
scope, every count above, and whether the row reconciliation was exact. It is
the provenance record for a snapshot: if two runs disagree, compare manifests
first.

## The evaluation dataset

The 50-question evaluation set and its answer key are **not** here — they are
in [`../EnergyX-Eval/`](../EnergyX-Eval/), alongside the grader that scores
against them.

## Timestamps, and why it matters

BDG2 timestamps are naive and expressed in each site's local standard time, at
a constant offset with no daylight saving. The build resolves each site's
offset and stores everything as UTC, partitioning by local year.

The DuckDB session must run with `SET TimeZone='UTC'`. Without it the host's own
timezone shifts every hour-of-day extraction and silently corrupts every
aggregate. `scripts/verify_snapshot.py` catches this by asserting the yearly
table's row count — one of the reasons to run it after a build.

Full treatment of the data model is in [`../docs/DATA_MODEL.md`](../docs/DATA_MODEL.md).
