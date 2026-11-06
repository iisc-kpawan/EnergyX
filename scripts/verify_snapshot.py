"""Acceptance checks for a built EnergyX snapshot.

Every check is a measurement against the source files, not an assertion about
what the ingestion intended to do. Run after any ingestion change:

    ./.venv/bin/python scripts/verify_snapshot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyx.config import PATHS, SNAPSHOT_VERSION  # noqa: E402
from energyx.ingest import tz as tzmod  # noqa: E402

SKIPPED: list = []
RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((passed, name, detail))


def skip(name: str, why: str) -> None:
    """Not applicable to this snapshot -- not the same thing as a failure."""
    SKIPPED.append((name, why))


def main() -> int:
    snap, src = PATHS.snapshot, PATHS.source_bdg2
    full = {p.stem: pd.read_parquet(p)
            for p in sorted((snap / "catalog").glob("*.parquet"))}

    # These checks were written against BDG2 and assert its exact counts, so
    # they are scoped to it. Federation is checked separately below -- a second
    # dataset must not silently change what those numbers mean.
    bdg2_buildings = set(full["buildings"].loc[
        full["buildings"]["dataset_id"] == "bdg2", "building_id"])
    bdg2_streams = set(full["electricity_streams"].loc[
        full["electricity_streams"]["dataset_id"] == "bdg2", "stream_id"])
    cat = {}
    for name, frame in full.items():
        if "dataset_id" in frame.columns:
            cat[name] = frame[frame["dataset_id"] == "bdg2"].reset_index(drop=True)
        elif "building_id" in frame.columns:
            cat[name] = frame[frame["building_id"].isin(bdg2_buildings)
                              ].reset_index(drop=True)
        elif "stream_id" in frame.columns:
            cat[name] = frame[frame["stream_id"].isin(bdg2_streams)
                              ].reset_index(drop=True)
        else:
            cat[name] = frame
    md = pd.read_csv(src / "BDG2_metadata.csv")
    con = duckdb.connect()
    elec_glob = str(snap / "electricity" / "**" / "*.parquet")

    # --- 0.1 schema integrity -------------------------------------------
    expected = {"datasets", "sites", "buildings", "building_attributes",
                "electricity_streams", "weather_sources", "data_quality",
                "capabilities"}
    check("0.1 all eight catalog tables present", set(cat) == expected,
          f"{sorted(set(cat))}")
    pks = {"datasets": "dataset_id", "sites": "site_id", "buildings": "building_id",
           "electricity_streams": "stream_id", "weather_sources": "weather_source_id",
           "data_quality": "stream_id", "capabilities": "building_id"}
    check("0.1 primary keys unique",
          all(cat[t][k].is_unique for t, k in pks.items()))
    check("0.1 no null primary keys",
          all(cat[t][k].notna().all() for t, k in pks.items()))

    # --- 0.2 counts and reconciliation ----------------------------------
    streams = cat["electricity_streams"]
    flagged = set(md.loc[md["electricity"] == "Yes", "building_id"])
    check("0.2 1,578 electricity streams", len(streams) == 1578, str(len(streams)))
    check("0.2 reconciliation exact (streams == metadata electricity=Yes)",
          set(streams["building_id"]) == flagged,
          f"orphans={len(set(streams['building_id']) ^ flagged)}")
    check("0.2 19 sites", len(cat["sites"]) == 19)
    check("0.2 1,636 buildings", len(cat["buildings"]) == 1636)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{elec_glob}', hive_partitioning=true) "
        "WHERE dataset = 'bdg2'").fetchone()[0]
    check("0.2 26,372,337 measurement rows", n_rows == 26_372_337, f"{n_rows:,}")

    # --- 0.2 UTC round-trips to source local time -----------------------
    offsets = cat["sites"].set_index("site_id")["utc_offset_hours"].to_dict()
    cleaned_head = pd.read_csv(src / "electricity_cleaned.csv", nrows=0)
    sample_cols = [c for c in cleaned_head.columns if c != "timestamp"][:400]
    probe = None
    for col in sample_cols:
        site = col.split("_")[0]
        series = pd.read_csv(src / "electricity_cleaned.csv",
                             usecols=["timestamp", col], parse_dates=["timestamp"])
        series = series.dropna()
        if len(series) > 100:
            probe = (col, site, series)
            break
    assert probe is not None
    col, site, series = probe
    got = con.execute(f"""
        SELECT timestamp_utc, value FROM read_parquet('{elec_glob}', hive_partitioning=true)
        WHERE building_id = '{col}' ORDER BY timestamp_utc
    """).df()
    offset = pd.Timedelta(hours=float(offsets[site]))
    got["local"] = tzmod.utc_to_local(got["timestamp_utc"], offset)
    series = series.rename(columns={col: "source_value"})
    merged = series.merge(got, left_on="timestamp", right_on="local", how="inner")
    check(f"0.2 UTC round-trips to source local time ({col})",
          len(merged) == len(series)
          and np.allclose(merged["source_value"], merged["value"]),
          f"matched {len(merged)}/{len(series)} values")

    # --- 0.3 quality flags ----------------------------------------------
    flags = con.execute(f"""
        SELECT quality_flag, COUNT(*) n
        FROM read_parquet('{elec_glob}', hive_partitioning=true)
        WHERE dataset = 'bdg2' GROUP BY 1 ORDER BY 1
    """).df().set_index("quality_flag")["n"].to_dict()
    check("0.3 ZERO_MASKED == 1,159,758 (measured raw zero count)",
          flags.get("ZERO_MASKED") == 1_159_758, f"{flags.get('ZERO_MASKED'):,}")
    check("0.3 OK == 25,212,579", flags.get("OK") == 25_212_579,
          f"{flags.get('OK'):,}")
    check("0.3 no MASKED_OTHER (cleaning is only zero-masking)",
          "MASKED_OTHER" not in flags, str(flags))
    nulls = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{elec_glob}', hive_partitioning=true)
        WHERE dataset = 'bdg2' AND quality_flag = 'ZERO_MASKED'
          AND value IS NOT NULL
    """).fetchone()[0]
    check("0.3 every ZERO_MASKED row has a null value", nulls == 0, str(nulls))

    # --- 0.4 country normalisation --------------------------------------
    sites = cat["sites"]
    countries = sorted(sites["country_canonical"].dropna().unique())
    check("0.4 exactly 4 canonical countries", countries == ["Canada", "Ireland", "UK", "US"],
          str(countries))
    check("0.4 no US state leaks into country", not (set(countries) & {
        "Arizona", "Florida", "Texas", "New York", "California", "New Jersey"}))
    caps = cat["capabilities"]
    b_all = cat["buildings"]
    no_geo = set(b_all.loc[b_all["country_canonical"].isna(), "building_id"])
    check("0.4 400 buildings sit in anonymous sites", len(no_geo) == 400, str(len(no_geo)))
    spatial_false = set(caps.loc[~caps["can_spatial_analysis"].fillna(False), "building_id"])
    check("0.4 every geography-less building lacks spatial capability",
          no_geo <= spatial_false)
    has_elec = set(streams.loc[streams["n_observations"] > 0, "building_id"])
    no_elec = set(b_all["building_id"]) - has_elec
    check("0.4 can_spatial_analysis == no-geography UNION no-electricity",
          spatial_false == (no_geo | no_elec),
          f"{len(spatial_false)} = |{len(no_geo)} u {len(no_elec)}|")
    check("0.4 country_raw preserved", sites["country_raw"].notna().all())

    # --- 0.5 capabilities ------------------------------------------------
    b = cat["buildings"]
    occ_meta = set(b_all.loc[b_all["occupants"].notna(), "building_id"])
    occ_zero = set(b_all.loc[b_all["occupants"] == 0, "building_id"])
    occ_usable = set(b_all.loc[b_all["occupants"] > 0, "building_id"])
    check("0.5 230 buildings carry occupancy metadata", len(occ_meta) == 230,
          str(len(occ_meta)))
    check("0.5 41 of those record zero occupants", len(occ_zero) == 41,
          str(len(occ_zero)))
    occ_cap = set(caps.loc[caps["can_occupancy_normalisation"].fillna(False), "building_id"])
    check("0.5 can_occupancy_normalisation == occupants > 0 AND usable electricity",
          occ_cap == (occ_usable & has_elec),
          f"{len(occ_cap)} usable of {len(occ_meta)} present "
          f"({len(occ_zero)} zero, {len(occ_meta & no_elec)} without electricity)")
    cov = streams.drop_duplicates("building_id").set_index(
        "building_id")["coverage_fraction"]
    quality = cat["data_quality"]
    expect_eui = int(((b["building_id"].map(cov).fillna(0) >= 0.90)
                      & b["floor_area_m2"].notna()).sum())
    check("0.5 can_eui matches floor-area + coverage predicate",
          int(caps["can_eui"].fillna(False).sum()) == expect_eui,
          f"{int(caps['can_eui'].fillna(False).sum())} vs {expect_eui}")
    # Flatness is measured and reported, and it withholds one capability only.
    # A stuck meter inside a peer group moves every other building's percentile;
    # its own EUI is merely its own problem, and EUI is a formula rather than a
    # judgement about the meter.
    mostly_flat = int((quality["flat_fraction"] >= 0.50).sum())
    flat_ids = streams.loc[streams["stream_id"].isin(
        quality.loc[quality["flat_fraction"] >= 0.50, "stream_id"]), "building_id"]
    check("0.5 flat-lined streams are detected",
          mostly_flat > 0, f"{mostly_flat} stream(s) at least half flat")
    check("0.5 flat-lined streams are kept out of peer benchmarking",
          int(caps.loc[caps["building_id"].isin(flat_ids),
                       "can_peer_benchmark"].fillna(False).sum()) == 0,
          "none of them can_peer_benchmark")
    check("0.5 a flat-lined stream can still have its own EUI computed",
          int(caps.loc[caps["building_id"].isin(flat_ids),
                       "can_eui"].fillna(False).sum()) > 0,
          "EUI is a formula, not a judgement about the meter")
    empty = streams.loc[streams["n_observations"] == 0, "building_id"].tolist()
    check("0.5 six empty streams excluded from consumption capability",
          len(empty) == 6 and not caps.set_index("building_id").loc[
              empty, "can_consumption_analysis"].fillna(False).any(),
          str(len(empty)))

    # --- weather ----------------------------------------------------------
    w = cat["weather_sources"]
    check("weather: all 19 sites covered", len(w) == 19 and set(w["site_id"]) == set(sites["site_id"]))
    check("weather: air temperature available everywhere",
          bool(w["air_temperature_available"].fillna(False).all()))

    # --- partition pruning ------------------------------------------------
    scanned = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{elec_glob}', hive_partitioning=true)
        WHERE site_id = 'Fox' AND year = 2017
    """).fetchone()[0]
    check("partition pruning: single site-year is a small slice",
          scanned < n_rows * 0.10, f"{scanned:,} of {n_rows:,}")

    # --- 0.6 aggregates ---------------------------------------------------
    from energyx.access import SnapshotDB  # noqa: E402
    with SnapshotDB() as db:
        check("0.6 session timezone pinned to UTC",
              db.value("SELECT current_setting('TimeZone')") == "UTC")
        # Local time must reproduce the source clock exactly.
        first_local = db.value("""SELECT MIN(timestamp_local) FROM electricity
                                  WHERE site_id = 'Fox'""")
        check("0.6 local time reproduces source clock (Fox starts 2016-01-01 00:00)",
              str(first_local).startswith("2016-01-01 00:00"), str(first_local))
        years = db.df("SELECT DISTINCT EXTRACT(year FROM timestamp_local) y "
                      "FROM electricity WHERE dataset_id = 'bdg2' "
                      "ORDER BY 1")["y"].astype(int).tolist()
        check("0.6 BDG2 local years are exactly {2016, 2017}",
              years == [2016, 2017], str(years))

        raw_total = db.value("SELECT SUM(value) FROM electricity")
        for grain in ("agg_daily", "agg_monthly", "agg_yearly"):
            got = db.value(f"SELECT SUM(total_kwh) FROM {grain}")
            check(f"0.6 {grain} total reconciles with raw",
                  abs(got - raw_total) / raw_total < 1e-9,
                  f"{got:,.1f} vs {raw_total:,.1f}")
        n_yearly = db.value("""
            SELECT COUNT(*) FROM agg_yearly a
            WHERE a.building_id IN (SELECT building_id FROM buildings
                                    WHERE dataset_id = 'bdg2')""")
        check("0.6 BDG2 yearly rows <= buildings x 2 years", n_yearly <= 1572 * 2,
              f"{n_yearly} <= {1572*2}")
        hours = db.df("SELECT MIN(hour_of_day) lo, MAX(hour_of_day) hi "
                      "FROM profile_daily").iloc[0]
        check("0.6 daily profile spans hours 0-23",
              int(hours.lo) == 0 and int(hours.hi) == 23, f"{hours.lo}-{hours.hi}")
        over = db.value("SELECT COUNT(*) FROM agg_daily WHERE coverage_fraction > 1.0")
        check("0.6 no daily coverage exceeds 1.0", over == 0, str(over))

    # --- 0.7 schema registry ----------------------------------------------
    from energyx.ingest import registry  # noqa: E402
    manifest = registry.load_dataset("bdg2")
    problems = registry.validate(manifest, src)
    check("0.7 bdg2 manifest validates against source", not problems, str(problems))
    check("0.7 manifest declares the timestamp convention",
          manifest.get("timestamps", "convention") == "local_standard_time")
    check("0.7 manifest excludes the non-electric channels",
          set(manifest.get("scope", "excluded_meter_types")) == {
              "chilledwater", "steam", "hotwater", "gas", "water",
              "irrigation", "solar"})

    # --- 1.1-1.3 knowledge graph ------------------------------------------
    from energyx.knowledge.store import KnowledgeStore  # noqa: E402
    from energyx.knowledge.validation import validate_query  # noqa: E402
    graph = KnowledgeStore()
    if graph.exists():
        counts = {
            "Building": ("buildings", int(graph.query(
                "SELECT (COUNT(?s) AS ?n) WHERE { ?s a ekg:Building }")[0]["n"])),
            "ElectricityStream": ("electricity_streams", int(graph.query(
                "SELECT (COUNT(?s) AS ?n) WHERE { ?s a ekg:ElectricityStream }")[0]["n"])),
            "Site": ("sites", int(graph.query(
                "SELECT (COUNT(?s) AS ?n) WHERE { ?s a ekg:Site }")[0]["n"])),
        }
        for klass, (table, got) in counts.items():
            expected = len(full[table])
            check(f"1.2 graph {klass} count matches the catalog", got == expected,
                  f"{got} vs {expected}")

        # The projection must agree with SQL wherever they overlap.
        sparql_rows = graph.query("""
            SELECT ?city (COUNT(DISTINCT ?b) AS ?n) WHERE {
              ?b a ekg:Building ; ekg:hasBuildingType ?t ; ekg:atSite ?site ;
                 ekg:canWeatherAnalysis true ; ekg:hasStream ?s .
              ?t ekg:buildingTypeName "Office" .
              ?s ekg:coverageFraction ?cov .
              ?site ekg:inCity ?c . ?c ekg:cityName ?city .
              FILTER(?cov >= 0.9)
            } GROUP BY ?city HAVING (COUNT(DISTINCT ?b) > 20)""")
        sql_rows = con.execute(f"""
            SELECT s.city, COUNT(DISTINCT b.building_id) n
            FROM read_parquet('{snap}/catalog/buildings.parquet') b
            JOIN read_parquet('{snap}/catalog/sites.parquet') s ON s.site_id=b.site_id
            JOIN read_parquet('{snap}/catalog/capabilities.parquet') c
              ON c.building_id=b.building_id
            JOIN read_parquet('{snap}/catalog/electricity_streams.parquet') st
              ON st.building_id=b.building_id
            WHERE b.building_type_canonical='Office' AND c.can_weather_analysis
              AND st.coverage_fraction >= 0.9 AND s.city IS NOT NULL
            GROUP BY 1 HAVING COUNT(DISTINCT b.building_id) > 20""").df()
        check("1.9 graph answer agrees with the catalog",
              {r["city"]: int(r["n"]) for r in sparql_rows}
              == dict(zip(sql_rows.city, sql_rows.n.astype(int))),
              f"graph={ {r['city']: int(r['n']) for r in sparql_rows} }")

        check("1.6 invalid SPARQL is refused before execution",
              not validate_query("SELECT ?b WHERE { ?b ekg:hasChiller ?c }").ok)
        check("1.6 update operations are refused",
              not validate_query("INSERT DATA { <a:b> <a:c> <a:d> }").ok)
        # Derived facts legitimately accumulate in a live graph; that a
        # *rebuild* clears them is asserted in the block below.
        check("1.3 graph is rebuildable from the catalog",
              (graph.path / "BUILD.json").exists() or graph.count() > 0)
    else:
        check("1.3 knowledge graph present", False,
              "run scripts/build_graph.py")

    # --- 2.x knowledge curation -------------------------------------------
    if graph.exists():
        from energyx.knowledge.curator import (  # noqa: E402
            Fact, KnowledgeCurator)
        from energyx.knowledge.vectors import VectorIndex  # noqa: E402
        curator = KnowledgeCurator(graph, SNAPSHOT_VERSION, run_id="verify")

        probe = Fact(kind="ChangePoint", building_id="__verify__", value=1.0,
                     method="verify", params={"probe": True})
        first = curator.assert_facts([probe])
        second = curator.assert_facts([probe])
        check("2.3 re-asserting a fact supersedes rather than duplicates",
              first.superseded == 0 and second.superseded == 1,
              f"first={first.superseded} second={second.superseded}")
        current = curator.current_facts(kind="ChangePoint",
                                        building_ids=["__verify__"])
        check("2.3 only the current belief is served", len(current) == 1,
              f"{len(current)} served")
        check("2.4 facts are pinned with method and assertion time",
              bool(current and current[0].get("method") and
                   current[0].get("computedAt")))
        bad = curator.assert_facts([Fact(kind="Vibes", building_id="x")])
        check("2.1 undeclared finding types are rejected",
              bad.asserted == 0 and bool(bad.rejected))

        stale = KnowledgeCurator(graph, "EnergyX-BDG2-v0")
        check("2.6 facts pinned to another snapshot are not served",
              len(stale.current_facts(building_ids=["__verify__"])) == 0)

        index = VectorIndex(snapshot_version=SNAPSHOT_VERSION)
        if index.path.exists():
            hits = index.search("Fox_office_Clayton", k=3)
            check("2.7 vector index searches load shapes",
                  len(hits) == 3 and all(-1 <= h.similarity <= 1 for h in hits),
                  f"{index.size} vectors")
        else:
            check("2.7 vector index present", False, "run scripts/build_graph.py")

        # The probe must not survive into a released snapshot.
        graph.build(SnapshotDB())
        check("2.8 rebuild clears derived facts (freeze reset)",
              int(graph.query("SELECT (COUNT(?f) AS ?n) WHERE "
                              "{ ?f a ekg:Finding }")[0]["n"]) == 0)
        VectorIndex(snapshot_version=SNAPSHOT_VERSION).build(SnapshotDB())
        graph.add_facts(VectorIndex(snapshot_version=SNAPSHOT_VERSION).graph_triples())

    # --- 3.x federation ----------------------------------------------------
    # A single-dataset snapshot is a legitimate configuration, not a broken one:
    # the federation and appliance checks test machinery that BDG2 alone cannot
    # exercise, so they are skipped rather than failed.
    datasets = sorted(full["datasets"]["dataset_id"])
    per_dataset = full["buildings"].groupby("dataset_id").size().to_dict()
    check("3.4 BDG2 unchanged by federation", per_dataset.get("bdg2") == 1636,
          str(per_dataset))
    check("3.4 every stream id is unique across datasets",
          full["electricity_streams"]["stream_id"].is_unique)
    check("3.4 units harmonised to kWh across datasets",
          set(full["electricity_streams"]["unit"].dropna()) == {"kWh"},
          str(sorted(set(full["electricity_streams"]["unit"].dropna()))))

    if len(datasets) > 1:
        check("3.4 more than one dataset is registered", True, str(datasets))
    else:
        skip("3.4 federation across datasets",
             f"one dataset registered ({datasets[0]}); nothing to federate")

    appliance = full["electricity_streams"][
        full["electricity_streams"]["stream_level"] == "appliance"]
    if len(appliance):
        check("3.6 appliance-level streams present with end uses",
              appliance["end_use"].notna().all(),
              f"{len(appliance)} streams, "
              f"{appliance['end_use'].nunique()} end uses")
        check("3.6 one building can carry many streams",
              appliance["building_id"].nunique() < len(appliance),
              f"{appliance['building_id'].nunique()} building(s), "
              f"{len(appliance)} streams")
    else:
        skip("3.6 appliance-level streams",
             "no appliance-level stream in this snapshot; BDG2 is "
             "whole-building only")

    if graph.exists():
        graph_datasets = graph.query(
            "SELECT (COUNT(?d) AS ?n) WHERE { ?d a ekg:Dataset }")
        check("3.4 graph unifies every dataset",
              int(graph_datasets[0]["n"]) == len(datasets),
              f"{graph_datasets[0]['n']} vs {len(datasets)}")
        levels = graph.query("""
            SELECT ?level (COUNT(?s) AS ?n) WHERE {
              ?s a ekg:ElectricityStream ; ekg:streamLevel ?level
            } GROUP BY ?level""")
        by_level = {r["level"]: int(r["n"]) for r in levels}
        check("3.6 graph distinguishes appliance from whole-building streams",
              by_level.get("appliance", 0) == len(appliance), str(by_level))

    # --- report -----------------------------------------------------------
    width = max(len(n) for _, n, _ in RESULTS)
    failed = 0
    print()
    for ok, name, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        failed += not ok
        print(f"  [{mark}] {name:<{width}}  {detail}")
    for name, why in SKIPPED:
        print(f"  [SKIP] {name:<{width}}  {why}")
    print(f"\n  {len(RESULTS) - failed}/{len(RESULTS)} checks passed"
          + (f", {len(SKIPPED)} not applicable to this snapshot" if SKIPPED else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
