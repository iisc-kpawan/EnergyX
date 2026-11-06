# EnergyX Knowledge Graph (EKG)

An RDF graph **derived from** the canonical data model, plus the facts EnergyX
learns while answering questions.

Engine: **Oxigraph** (`pyoxigraph`) — embedded, SPARQL 1.1, no server.

---

## 1. What this is, and what it is not

| | |
|---|---|
| **Is** | A projection of the ECDM into triples, extended with derived analytical facts and provenance |
| **Is not** | An ingestion requirement. No dataset must conform to Brick, Haystack or any ontology to enter EnergyX |

This does not reverse the earlier rejection of Brick/Haystack. What was
rejected — correctly — was forcing heterogeneous whole-building meter datasets into
equipment ontologies they cannot populate. What is adopted is a graph *built from*
the canonical model EnergyX already produces.

**Direction of truth is one-way:** the catalog is authoritative; the graph
is generated from it. If they ever disagree, the catalog is right and the graph
projection is broken.

### Why it earns its place

1. **A fast path that never touches the time series.** Categories 1, 2 and much of
   8 become traversal — milliseconds instead of scanning 26.4M measurement rows.
2. **Multi-hop traversal where the join path is not known in advance.** SQL needs
   the path anticipated as a view; graphs do not.
3. **Derived findings become reusable facts.** Category 12 — combine EUI, baseload,
   peer percentile, weather-normalised excess and persistent anomalies — is
   literally a graph join once those are facts.
4. **Cross-dataset unification.** When BYOD datasets arrive with different schemas,
   the graph is where `Building`, `City` and `BuildingType` unify.

### Positioning

BuildingGPT, BrickQA and BuildingQA stop at the graph:
`NL → knowledge graph → SPARQL → answer`.

EnergyX closes the loop:
`NL → graph → data → analytics → evidence → back to the graph`.

No system in the related work does the round trip. And because the query language
is SPARQL, **NL→SPARQL accuracy is directly comparable** with the BuildingQA
precedent.

---

## 2. Namespaces

```
ekg:   http://energyx.org/ns#          ontology terms
ekgd:  http://energyx.org/id/          instance identifiers
ekgp:  http://energyx.org/prov#        provenance terms
xsd:   http://www.w3.org/2001/XMLSchema#
```

Entity IRIs are **stable and dereferenceable**, minted from ECDM primary keys:

```
ekgd:dataset/bdg2
ekgd:site/bdg2/Fox
ekgd:building/bdg2/Fox_office_Ana
ekgd:stream/bdg2/Fox_office_Ana/elec
ekgd:weather/bdg2/Fox
ekgd:city/US/Tempe
ekgd:country/US
ekgd:peergroup/<hash of criteria>
ekgd:finding/<uuid>
ekgd:run/<run_id>
```

Stability matters: a derived fact asserted in one run must still address the
same entity in the next, which is why ingestion mints IDs even before the graph
exists.

---

## 3. Ontology

### Asserted classes — mirror the ECDM

```
ekg:Dataset            ekg:Site              ekg:Building
ekg:ElectricityStream  ekg:WeatherSource
ekg:Country            ekg:Region            ekg:City
ekg:BuildingType       ekg:Industry          ekg:HeatingType
```

### Derived classes — what EnergyX learns

```
ekg:PeerGroup          a cohort constructed by stated criteria
ekg:LoadShapeCluster   a cluster over load profiles
ekg:Finding            base class for analytical results
  ├── ekg:ChangePoint        temperature change-point / balance point
  ├── ekg:AnomalyEvent       anomalous day or interval
  ├── ekg:TrendSegment       persistent behavioural shift
  ├── ekg:OperatingSchedule  inferred start / shutdown / after-hours
  └── ekg:PerformanceScore   EUI, percentile, normalised excess
ekg:Representation     handcrafted features or a learned embedding
```

### Provenance classes

```
ekgp:AnalysisRun       ekgp:ToolInvocation      ekgp:Method
```

### Structural relations

```
ekg:inDataset       Building|Site|Stream  → Dataset
ekg:atSite          Building              → Site
ekg:inCity          Site                  → City
ekg:inRegion        City                  → Region
ekg:inCountry       Region|City           → Country
ekg:hasStream       Building              → ElectricityStream
ekg:servedByWeather Site                  → WeatherSource
ekg:hasBuildingType Building              → BuildingType
ekg:hasIndustry     Building              → Industry
ekg:heatingType     Building              → HeatingType
```

### Datatype properties (selection)

```
ekg:floorAreaM2  xsd:double     ekg:yearBuilt        xsd:gYear
ekg:occupants    xsd:integer    ekg:numberOfFloors   xsd:integer
ekg:timezone     xsd:string     ekg:latitude         xsd:double
ekg:longitude    xsd:double     ekg:streamLevel      xsd:string
ekg:unit         xsd:string     ekg:resolutionMinutes xsd:integer
ekg:coverageFraction xsd:double ekg:startTime        xsd:dateTime
ekg:endTime      xsd:dateTime   ekg:qualityStatus    xsd:string
```

### Derived relations

```
ekg:memberOf         Building → PeerGroup
ekg:inCluster        Building → LoadShapeCluster
ekg:similarLoadShape Building → Building        (weighted — see §5)
ekg:hasFinding       Building → Finding
ekg:hasRepresentation Building → Representation
ekg:percentileIn     Building → PeerGroup       (weighted)
```

### Capability properties

Mirrored from the catalog's capability table so eligibility is answerable by
traversal:

```
ekg:canComputeEUI              xsd:boolean
ekg:canWeatherAnalysis         xsd:boolean
ekg:canOccupancyNormalise      xsd:boolean
ekg:canForecast                xsd:boolean
ekg:canPeerBenchmark           xsd:boolean
ekg:canSpatialAnalysis         xsd:boolean   -- false for the 400 Anonymous buildings
```

---

## 4. Worked shape

```turtle
ekgd:building/bdg2/Fox_office_Ana
    a                      ekg:Building ;
    ekg:inDataset          ekgd:dataset/bdg2 ;
    ekg:atSite             ekgd:site/bdg2/Fox ;
    ekg:hasBuildingType    ekgd:buildingtype/Office ;
    ekg:floorAreaM2        "9265.4"^^xsd:double ;
    ekg:yearBuilt          "1998"^^xsd:gYear ;
    ekg:canComputeEUI      true ;
    ekg:canSpatialAnalysis true ;
    ekg:hasStream          ekgd:stream/bdg2/Fox_office_Ana/elec .

ekgd:site/bdg2/Fox
    a                      ekg:Site ;
    ekg:siteName           "Arizona State Univ. (ASU)" ;
    ekg:inCity             ekgd:city/US/Tempe ;
    ekg:timezone           "US/Mountain" ;
    ekg:servedByWeather    ekgd:weather/bdg2/Fox .

ekgd:city/US/Tempe  ekg:inRegion  ekgd:region/US/Arizona .
ekgd:region/US/Arizona  ekg:inCountry  ekgd:country/US .
```

Note `Tempe → Arizona → US`: the source `Country` column says *"Arizona"*.
The graph carries the normalised hierarchy, so *"compare buildings across
countries"* works without every query re-deriving that Arizona is not a country.

---

## 5. Derived facts, pinned

Every derived fact carries metadata. Modelled with **RDF-star** to avoid classical
reification:

```turtle
<< ekgd:building/bdg2/Fox_office_Ana
   ekg:similarLoadShape
   ekgd:building/bdg2/Bull_office_Nell >>
       ekg:similarity        "0.94"^^xsd:double ;
       ekgp:assertionType    "derived" ;
       ekgp:snapshotVersion  "EnergyX-BDG2-v1" ;
       ekgp:method           "normalized_daily_profile_dtw" ;
       ekgp:params           '{"resolution":"hourly","window":"2017"}' ;
       ekgp:confidence       "0.88"^^xsd:double ;
       ekgp:computedAt       "2026-09-12T14:20:11Z"^^xsd:dateTime ;
       ekgp:computedByRun    ekgd:run/r_8f21a .
```

A `Finding` as a first-class node, when the result has temporal extent:

```turtle
ekgd:finding/9c2e1
    a                     ekg:ChangePoint ;
    ekg:aboutBuilding     ekgd:building/bdg2/Fox_office_Ana ;
    ekg:value             "18.2"^^xsd:double ;
    ekg:unit              "degC" ;
    ekg:validFrom         "2016-01-01T00:00:00Z"^^xsd:dateTime ;
    ekg:validTo           "2017-12-31T23:00:00Z"^^xsd:dateTime ;
    ekgp:assertionType    "derived" ;
    ekgp:snapshotVersion  "EnergyX-BDG2-v1" ;
    ekgp:method           "piecewise_changepoint" ;
    ekgp:computedByRun    ekgd:run/r_8f21a .
```

### The five required fields

| Field | Why it is mandatory |
|---|---|
| `assertionType` ∈ {asserted, derived} | Without it a wrong finding is indistinguishable from ground truth and errors compound silently |
| `snapshotVersion` | A fact pinned to a superseded snapshot is stale by definition and must not be served |
| `method` + `params` | Two change-point algorithms give different answers; the fact is meaningless without saying which |
| `computedAt` | Assertion time, for bi-temporal reasoning |
| `computedByRun` | Traceability back to the full run trace |

### Bi-temporal by design

- **Valid time** — `validFrom` / `validTo`: when the fact was true of the building
- **Assertion time** — `computedAt`: when EnergyX came to believe it

Building energy is inherently temporal: an operating schedule that changed in
March 2017 is not wrong, it is bounded. Collapsing the two would make
*"when did this building's behaviour change?"* unanswerable from the graph.

---

## 6. Write-back — the Knowledge Curator

```
question → plan → analyse → answer
                     │
                     ▼
            Knowledge Curator (Layer 3 service, deterministic)
             ├─ validate against ontology and ranges
             ├─ pin snapshot / method / params / run
             ├─ deduplicate against existing equivalent facts
             ├─ supersede rather than overwrite
             └─ assert with assertionType = "derived"
                     │
                     ▼
        Energy KG gains facts → next question answered without recomputing
```

The Curator is a **service, not an agent**, and it does not reason. Letting a
write-back path reason is how a speculative conclusion becomes a permanent fact.

**The measurable claim:** cost and latency per question fall as the knowledge layer
saturates. Work a portfolio once and peer groups, change-points and similarity
edges accumulate; later questions ride on them. That is a genuine "improves with
use" property, and it required no training.

---

## 7. Reproducibility — non-negotiable

**A measured run goes against a frozen, read-only graph.**

If one run writes a fact that a later run reads, the two are no longer
independent, and a single wrong cached finding poisons everything downstream of
it. `ENERGYX_WRITE_BACK=0` pins the graph to the fact set the snapshot shipped
with, so a result can be reproduced months later.

```
MEASURED                 write-back DISABLED
                         graph = snapshot-shipped fact set only
                         report graph-cold and graph-warm separately

EXPLORATORY              write-back ENABLED
                         this is where the saturation curve is measured

INTERACTIVE              write-back ENABLED, session-scoped
```

Reporting warm and cold numbers separately turns the ambiguity into a result.

---

## 8. Standards export

`export_semantic_model(target=brick|haystack|bedes)` projects the EKG outward.
Because the graph is already RDF, Brick and Haystack are a mapping rather than a
rewrite:

| EnergyX | Brick | Haystack |
|---|---|---|
| `ekg:Building` | `brick:Building` | `site` |
| `ekg:floorAreaM2` | `brick:area` | `area` |
| `ekg:ElectricityStream` (whole_building) | `brick:Building_Electrical_Meter` | `siteMeter` + `elec-meter` |
| `ekg:ElectricityStream` (submeter) | `brick:Electrical_Meter` | `submeterOf` |
| `ekg:servedByWeather` | `brick:hasLocation` (approx.) | `weatherStationRef` |
| `ekg:timezone` | — | `tz` |
| `ekg:yearBuilt` | — | `yearBuilt` |

This is what makes the interoperability claim real rather than aspirational, and it
is the third level of the hierarchy the original discussion left unspecified:
**source schema → canonical model → standards mappings.**

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Graph duplicates and diverges from the catalog | One generator, one direction; catalog authoritative; Verifier spot-checks overlapping answers |
| A cached fact contaminating a measured run | Frozen read-only graph when write-back is off; warm/cold reported separately |
| NL→SPARQL errors — the failure mode BuildingGPT reports | Constrain generation to the EKG ontology; validate every query against the schema before execution; Verifier cross-checks against catalog SQL |
| Ontology bloat | The ontology mirrors the ECDM and grows only with derived classes that a tool actually writes |
| Stale facts after re-ingestion | Every fact pinned to `snapshotVersion`; facts on a superseded snapshot are not served |

---

See also: `QUERY_ROUTING.md` for how a query is resolved across the three stores
`DATA_MODEL.md` for the ECDM the graph projects · …
