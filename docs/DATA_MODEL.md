# EnergyX Canonical Data Model (ECDM) & Snapshot Specification

> **Keep the canonical core small and stable; map heterogeneous source schemas into
> it; preserve source-specific attributes through an extensible layer.**

---

## 1. The canonical model

```
Dataset
  └── Site ──────────────► WeatherSource
       └── Building
            ├── BuildingAttributes   (extensible key/value)
            └── ElectricityStream
                 └── Measurements
```

Five entities, deliberately. Measurements stay minimal — everything else arrives by
join, so metadata is never duplicated per timestamp.

### Entity tables

**`datasets`**
```
dataset_id*        dataset_name*      dataset_version    description
source_url         license            citation           building_sector
```

**`sites`**
```
site_id*           dataset_id*        site_name          city
state_region       country_canonical  country_raw        country_code
latitude           longitude          timezone*          climate_zone
weather_source_id
```

**`buildings`**
```
building_id*       site_id*           dataset_id*
building_type_canonical   building_type_raw    building_subtype
floor_area_m2      year_built         number_of_floors   occupants
heating_type       industry           subindustry
latitude           longitude          timezone
```

**`building_attributes`** — extensible property layer
```
building_id*   attribute_name*   attribute_value   attribute_unit   source
```
New datasets bring attributes nobody anticipated (`bedrooms`, `tariff`,
`retrofit_year`, `ownership`). They land here rather than widening the core schema.
BDG2's `energystarscore`, `leed_level`, `rating`, `eui`, `site_eui`, `source_eui`
and `date_opened` are carried as attributes, not core fields.

**`electricity_streams`**
```
stream_id*         building_id*       dataset_id*
stream_level*      end_use            unit*              resolution_minutes
start_timestamp    end_timestamp      n_observations     expected_observations
coverage_fraction  source_stream_name parquet_path
```

`stream_level ∈ {whole_building, submeter, circuit, appliance}` and `end_use` are
kept **separate**, so `(appliance, refrigerator)` and `(whole_building, null)`
coexist cleanly and commercial and residential datasets share one table.

**`weather_sources`**
```
weather_source_id*  site_id*   source_name   resolution_minutes
start_timestamp     end_timestamp            coverage_fraction
<variable>_available  (one boolean per weather variable)
parquet_path
```

**`data_quality`** — one row per stream
```
stream_id*         coverage_fraction   missing_fraction   zero_masked_fraction
duplicate_count    outlier_fraction    longest_gap_hours
first_valid_timestamp   last_valid_timestamp   quality_status
dst_duplicate_hours     dst_missing_hours
```

**`capabilities`** — derived, one row per building
```
building_id*  can_consumption_analysis  can_eui          can_weather_analysis
can_occupancy_normalisation             can_peer_benchmark
can_forecasting                         can_spatial_analysis
```
Computed from availability, not asserted:
```
can_eui = electricity_available AND floor_area_m2 IS NOT NULL AND coverage >= threshold
can_spatial_analysis = country_canonical IS NOT NULL AND city IS NOT NULL
```

### Measurements

```
electricity:  timestamp_utc*   stream_id*   value   quality_flag
weather:      timestamp_utc*   weather_source_id*   <variables...>
```

That is all. No country, city, building type, floor area or unit repeated per row.

---

## 2. Conventions

### Timestamps — UTC stored, local analysed

BDG2 timestamps are naive and expressed in each site's **local standard time**, on
a fixed UTC offset with **no daylight-saving adjustment** — established from the
data itself. Ingestion applies each site's constant offset and stores
**UTC**; the offset is resolved once and kept in `sites.utc_offset_hours` so
nothing downstream re-derives it. Fetch and analysis tools default to **local
building time**.

Storage is unambiguous and joinable; load profiles and peak timing stay physically
meaningful across countries — essential for Category 8, where comparing peak timing
in UTC across Tempe and London would be meaningless.

Because the series carries no DST, `dst_duplicate_hours` and `dst_missing_hours`
are 0 throughout this snapshot. The fields remain in `data_quality` for BYOD
datasets that do carry DST.

Partition `year` is the **local** year, so each site has exactly one partition per
calendar year of its own clock rather than spilling into a third UTC year.

### Quality flags

| Flag | Meaning |
|---|---|
| `OK` | present in both raw and cleaned |
| `ZERO_MASKED` | raw reading was exactly `0`, masked by BDG2 cleaning |
| *(row absent)* | missing in the source itself — a true gap |

Analytics default to cleaned values. Data-quality tools see through to what was
masked, which is what keeps Category 2's "extended zero runs" questions answerable.

### Canonical vs raw

Any normalised field keeps its source value:
`country_canonical` / `country_raw`, `building_type_canonical` /
`building_type_raw`. Provenance is preserved and the mapping is reversible —
essential for BYOD, where a user must be able to see how their vocabulary was
interpreted.

### Identifiers

Stable and dereferenceable, minted from primary keys, and identical to the IRIs the
knowledge graph projects (`KNOWLEDGE_GRAPH.md` §2). A fact asserted in one run must
still address the same entity in the next.

---

## 3. Physical layout

```
data/EnergyX-BDG2-v1/
├── electricity/
│   └── dataset=bdg2/site_id=<site>/year=<yyyy>/part-*.parquet
├── weather/
│   └── dataset=bdg2/site_id=<site>/year=<yyyy>/part-*.parquet
├── aggregates/
│   ├── hourly/  daily/  monthly/  yearly/
│   └── profiles/          typical daily and weekly load shapes
├── catalog/
│   ├── datasets.parquet          sites.parquet
│   ├── buildings.parquet         building_attributes.parquet
│   ├── electricity_streams.parquet
│   ├── weather_sources.parquet   data_quality.parquet
│   └── capabilities.parquet
├── graph/
│   └── ekg.ttl / oxigraph store
└── MANIFEST.json                    provenance, counts, checksums
```

Partitioning is `dataset → site_id → year`: 19 sites over 2 years, **37
partitions** — Swan carries 2017 only.
`building_id` stays a **column**, and files are sorted by `(building_id,
timestamp_utc)` so Parquet row-group statistics prune by building without creating
1,578 tiny files.

Deliberately avoided: country/city/building/day partitions — thousands of small
files and deep directory trees hurt far more than they help at this scale.

---

## 4. EnergyX-BDG2-v1 — the frozen snapshot

Source: the complete official BDG2 release in `BDG2/`. Nothing downloaded.

### Inclusion rules

- Electricity is the only meter channel. Chilled water, steam, hot water, gas,
  water, irrigation and solar are present in the source and deliberately unread.
  Site weather and building metadata are ingested alongside it as context.
- All 1,578 buildings flagged `electricity == "Yes"`.
- Full period 2016-01-01 00:00 → 2017-12-31 23:00 local, hourly.
- `stream_level = whole_building` for every BDG2 stream.

### Verified properties

| Property | Value |
|---|---|
| Buildings in metadata | 1,636 |
| Buildings with electricity | **1,578** |
| Reconciliation | electricity columns ≡ `electricity == "Yes"` rows, **exactly**, no orphans either way |
| Sites | 19 |
| Countries (canonical) | 4 — US, UK, Canada, Ireland |
| Hourly timestamps | 17,544 (731 days × 24; 2016 is a leap year) |
| Measurement cells | 27.7M |
| `sqm` present | 1,636 / 1,636 (**100%**) — EUI computable everywhere |
| `yearbuilt` present | 817 (50%) |
| `occupants` present | **230 (14%)** |

### Cleaned vs raw — measured

```
raw     NaN: 1,312,095  (4.74%)
cleaned NaN: 2,471,853  (8.93%)

newly masked (raw → NaN): 1,159,758
values modified:                  0
zeros: raw 1,159,758  →  cleaned 0
```

Cleaning in this release is **exactly and only zero-masking**. The newly-masked
count equals the raw zero count precisely, and not one surviving value was altered
— nothing smoothed, imputed or rescaled. This is why the `ZERO_MASKED` flag is
sufficient to reconstruct the raw series.

### Coverage

```
per-building electricity coverage:  mean 91.1%   median 98.9%
  ≥ 90%:  1,256 buildings
  ≥ 95%:  1,019
  ≥ 99%:    758
  zero data at all: 6 buildings  (flagged electricity=Yes, entirely empty)
```

Those 6 are excluded by explicit predicate, never silently.

### Geography

All 19 sites carry weather. The source `Country` column mixes countries with US
states and a placeholder, and is normalised on ingest:

| Source value | `country_canonical` | `state_region` |
|---|---|---|
| `US` | US | *(null)* |
| `Arizona`, `Florida`, `Texas`, `New York`, `California`, `New Jersey` | US | that state |
| `UK` | UK | *(null)* |
| `Canada` | Canada | Ontario |
| `Ireland` | Ireland | *(null)* |
| `Anonymous` | *(null)* | *(null)* |

**400 buildings across 5 sites** — Bobcat, Eagle, Gator, Hog, Swan — have no
geography. Fully usable for categories 1–7 and 9–12; excluded from spatial tasks by
`can_spatial_analysis = false`.

### Weather variable completeness

```
airTemperature   100.0%      windDirection    96.1%
dewTemperature    99.9%      seaLvlPressure   93.5%
windSpeed         99.8%      precipDepth1HR   59.8%
                             cloudCoverage    48.4%
                             precipDepth6HR    5.5%
```

Temperature-driven work — change-point models, degree days, weather normalisation
(Category 6) — rests on complete data. `cloudCoverage` and especially
`precipDepth6HR` are too sparse to rest an analysis on.

### As built

```
catalog tables        8          measurement rows    26,372,337
  datasets            1            OK                25,212,579
  sites              19            ZERO_MASKED        1,159,758
  buildings       1,636            MASKED_OTHER               0
  building_attributes 4,281      true gaps (omitted)  1,312,095
  electricity_streams 1,578      weather rows           331,166
  weather_sources    19
  data_quality    1,578          electricity parquet        93 MB
  capabilities    1,636          weather parquet           4.8 MB
```

Partitions: **37**, not 38 — Swan has no 2016 data (its series starts
2017-01-03), so only a 2017 partition exists.

Capability counts( — *can this analysis run*, not *does the field exist*):

| Capability | True | Note |
|---|---:|---|
| `can_consumption_analysis` | 1,572 | 1,578 streams less 6 empty |
| `can_eui` | 1,256 | floor area ∧ ≥90% coverage |
| `can_occupancy_normalisation` | **188** | 230 have `occupants`, but 41 record zero and 2 lack electricity |
| `can_spatial_analysis` | 1,191 | false for 400 anonymous ∪ 64 without electricity |

### Freezing

`MANIFEST.json` records source file checksums, ingestion parameters, row and entity
counts, the exclusion list, and the code version. Every answer is computed
against this snapshot and cites it. Re-ingestion that changes any value mints a new
version — it does not overwrite v1.

---

## 5. BYOD — bringing another dataset

Required minimum:
```
electricity:  timestamp, building_id, value, unit
metadata:     building_id, site_id, timezone
```
Strongly recommended: `resolution_minutes`, `building_type`, `city`, `country`.
Optional: floor area, year built, floors, occupants, heating type, lat/lng, weather.

Package layout:
```
my_dataset/
├── energyx.yaml          schema mapping and configuration
├── sites.csv             user-supplied metadata
├── buildings.csv
├── electricity/*.parquet
└── weather/*.parquet
```

Rule: **YAML = schema and configuration; CSV = user-supplied metadata;
Parquet = data and catalog.** Thousands of buildings never belong in YAML.

```yaml
dataset:
  id: my_dataset
  name: My Building Dataset
  sector: commercial
electricity:
  path: electricity/**/*.parquet
  timestamp: datetime
  building_id: building
  value: consumption
  unit: kWh
metadata:
  path: buildings.csv
  building_id: building
  site_id: site
  building_type: usage
  floor_area_m2: area
weather:
  path: weather/**/*.parquet
  timestamp: datetime
  site_id: site
  air_temperature: temp
timezone:
  source: metadata
  column: timezone
```

Onboarding runs `inspect_dataset → map_to_energyx_schema → validate_dataset →
register_dataset`, then reports capabilities honestly:

```
Dataset registered: my_dataset
Buildings 428 · Sites 12 · Streams 428 · Resolution 15 min · 2022–2025

✓ Consumption analytics   ✓ Load-profile analytics   ✓ Forecasting
✓ Anomaly detection       ✓ Weather analytics        ✓ Peer benchmarking
✗ EUI analysis — floor_area_m2 unavailable
```

The claim is **guided interoperability**: *"EnergyX is dataset-extensible through a
standardized onboarding interface"* — not *"EnergyX automatically supports any
energy dataset"*, which would not survive contact with a reviewer.

---

See also: `KNOWLEDGE_GRAPH.md` · `ARCHITECTURE.md` · …
