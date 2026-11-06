"""Decision Support & Reporting tools (5).

Covers task category 12 and the presentation of every other category's output.

The reporting agent receives *computed values*; it does not calculate. Ranking
here means applying a stated policy to evidence that other specialists produced,
and the policy is always returned with the ranking so a decision can be audited
or reproduced.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .base import ToolResult, empty, frame_result, insufficient, ok
from .registry import ToolContext, tool


def _as_frame(ctx: ToolContext, source) -> pd.DataFrame | None:
    """Accept a handle, a list of records, or a frame."""
    if source is None:
        return None
    if isinstance(source, str):
        return ctx.store.get_or_none(source)
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, list):
        return pd.DataFrame(source)
    return None


@tool("reporting")
def combine_evidence(
    ctx: ToolContext,
    sources: list[str],
    key: str = "building_id",
    labels: list[str] | None = None,
    mode: Literal["any", "all"] = "any",
) -> ToolResult:
    """Join several specialists' outputs into one evidence table.

    This is the fan-in for category 12: performance, weather, benchmarking and
    anomaly results arrive as separate handles and leave as one frame keyed by
    building.

    ``mode`` decides who survives the join. "any" keeps every entity any
    source mentioned, which is what an evidence table for scoring wants.
    "all" keeps only those every source named, which is the intersection --
    "buildings above the 90th percentile of EUI *and* of nighttime baseload",
    "offices in the top decile by consumption *and* by intensity". Without it
    a conjunction had no expression: each criterion could be resolved on its
    own and the two sets could only be glued side by side, so the answer came
    back as one criterion's count or as a ranking of their union.
    """
    if not sources:
        return insufficient("no_sources", "combine_evidence needs at least one handle")
    merged = None
    used: list[str] = []
    for i, handle in enumerate(sources):
        df = _as_frame(ctx, handle)
        if df is None or df.empty or key not in df.columns:
            continue
        label = (labels[i] if labels and i < len(labels) else f"src{i}")
        renamed = df.rename(columns={c: f"{label}.{c}" for c in df.columns if c != key})
        how = "inner" if mode == "all" else "outer"
        merged = renamed if merged is None else merged.merge(renamed, on=key, how=how)
        used.append(label)
    if merged is None:
        return empty(f"no source contained the key {key!r}")
    noun = key.replace("_", " ").replace(" id", "")
    if mode == "all":
        # The count is the answer to a conjunction, so the tool says it rather
        # than leaving a reader to count rows in a truncated preview.
        headline = {"value": len(merged), "unit": f"{noun}s",
                    "label": (f"{len(merged):,} {noun}s in all "
                              f"{len(used)} sets: " + " and ".join(used))}
    else:
        headline = {"value": len(merged), "unit": f"{noun}s",
                    "label": (f"{len(merged):,} {noun}s named by any of "
                              f"{len(used)} sources")}
    return frame_result(merged, ctx.store, name="evidence",
                        evidence={"key": key, "sources_used": used,
                                  "mode": mode, "entities": len(merged),
                                  "headline": headline})


@tool("reporting")
def prioritize_entities(
    ctx: ToolContext,
    evidence_handle: str,
    criteria: dict,
    key: str = "building_id",
    limit: int = 10,
    higher_is_worse: bool = True,
) -> ToolResult:
    """Rank entities by a weighted, explicitly stated policy.

    ``criteria`` maps column name to weight, e.g.
    ``{"perf.eui": 0.3, "anomaly.score": 0.25}``. Columns are min-max normalised
    before weighting so units do not dominate. The policy travels with the
    result -- a ranking whose policy is unstated cannot be defended.
    """
    df = _as_frame(ctx, evidence_handle)
    if df is None or df.empty:
        return empty("evidence table is empty")
    # combine_evidence prefixes each source's columns to keep them apart --
    # "src0.anomalies", "src1.mape" -- so a policy written in the question's
    # own words ("anomalies", "mape") matched nothing and the whole
    # prioritisation refused. A criterion is resolved against the suffix when
    # it is unambiguous there, which is what lets a caller name the measure
    # rather than the plumbing.
    resolved, missing, ambiguous = {}, [], {}
    for name, weight in criteria.items():
        if name in df.columns:
            resolved[name] = weight
            continue
        matches = [c for c in df.columns
                   if c.split(".", 1)[-1] == name and c != key]
        if len(matches) == 1:
            resolved[matches[0]] = weight
        elif matches:
            ambiguous[name] = matches
        else:
            missing.append(name)
    if ambiguous:
        return insufficient(
            "ambiguous_columns",
            "; ".join(f"{name!r} could be {opts}" for name, opts
                      in ambiguous.items())
            + " -- name the column outright")
    if missing:
        return insufficient("missing_columns",
                            f"evidence lacks {missing}; available: "
                            f"{[c for c in df.columns if c != key]}")
    criteria = resolved
    total_weight = sum(abs(w) for w in criteria.values())
    if not total_weight:
        return insufficient("zero_weights", "criteria weights sum to zero")

    scored = df[[key]].copy()
    contributions = {}
    for column, weight in criteria.items():
        values = pd.to_numeric(df[column], errors="coerce")
        lo, hi = values.min(), values.max()
        norm = (values - lo) / (hi - lo) if hi > lo else values * 0
        if not higher_is_worse:
            norm = 1 - norm
        contributions[column] = norm * (weight / total_weight)
        scored[f"norm.{column}"] = norm.round(4)
    scored["priority_score"] = sum(contributions.values())
    # A candidate with no evidence cannot be prioritised. Their score is NaN,
    # which sorts to the end and then filled ranks 3 to 10 of a ten-row
    # shortlist because only two of 691 buildings carried both measures: the
    # answer recommended eight buildings to investigate on the strength of
    # knowing nothing about them. Absence of evidence is a reason to compute,
    # not a reason to appear on a worklist.
    unscored = int(scored["priority_score"].isna().sum())
    scored = scored.dropna(subset=["priority_score"])
    if scored.empty:
        return insufficient(
            "no_scored_candidates",
            f"none of the {len(df):,} candidates carry all of "
            f"{list(criteria)}; nothing has been computed for them yet")
    scored = scored.sort_values("priority_score", ascending=False).head(limit)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))
    policy = {k: round(v / total_weight, 4) for k, v in criteria.items()}
    # A ranking without its policy is a list of names with a number beside
    # them. What a reader has to be able to disagree with is the weighting and
    # what drove each entry to the top, so both are stated.
    rows = scored.to_dict("records")
    scoreable = len(df) - unscored
    narrative = [
        f"{len(rows)} of {scoreable:,} scoreable "
        f"{key.replace('_', ' ')}s to look at first, worst first.",
    ]
    for row in rows[:5]:
        drivers = sorted(
            ((column[len("norm."):], value) for column, value in row.items()
             if column.startswith("norm.") and isinstance(value, (int, float))),
            key=lambda pair: -pair[1])
        worst = ", ".join(f"{name.replace('_', ' ')} {value:.0%} of the "
                          f"cohort's range" for name, value in drivers[:2])
        narrative.append(f"  {row['rank']}. {row[key]}   "
                         f"score {row['priority_score']:.3f}"
                         + (f"   — {worst}" if worst else ""))
    if len(rows) > 5:
        narrative.append(f"  ... and {len(rows) - 5} more in the table below")
    narrative.append(
        "  Policy: " + ", ".join(f"{name.replace('_', ' ')} {weight:.0%}"
                                 for name, weight in policy.items())
        + f"; each measure scaled min-max within these {scoreable:,} candidates, "
        + ("higher is worse" if higher_is_worse else "lower is worse") + ".")
    if unscored:
        narrative.append(
            f"  {unscored:,} of the {len(df):,} candidates carry none of these "
            f"measures yet and were not ranked -- nothing has been computed "
            f"for them, which is not the same as their being fine.")
    narrative.append(
        "  The score ranks; it does not diagnose. A high score says a building "
        "sits far up this cohort's range on the weighted measures, not that "
        "anything is wrong with it.")
    return frame_result(scored, ctx.store, name="priorities",
                        evidence={"policy": policy,
                                  "normalisation": "min-max within this cohort",
                                  "higher_is_worse": higher_is_worse,
                                  "candidates": scoreable,
                                  "candidates_seen": len(df),
                                  "candidates_unscored": unscored,
                                  "narrative": narrative,
                                  "headline": {
                                      "value": (rows[0][key] if rows else None),
                                      "unit": None,
                                      "label": f"the {key.replace('_', ' ')} to "
                                               f"investigate first"}})


@tool("reporting")
def generate_decision_support(
    ctx: ToolContext,
    priorities_handle: str,
    question: str,
    evidence_handle: str | None = None,
    limit: int = 5,
) -> ToolResult:
    """Turn a ranking into a recommendation that states its evidence and limits.

    Conclusions are phrased as association, never causation: observational meter
    data cannot establish why a building behaves as it does.
    """
    ranked = _as_frame(ctx, priorities_handle)
    if ranked is None or ranked.empty:
        return empty("no priorities to explain")
    evidence = _as_frame(ctx, evidence_handle)
    key = "building_id" if "building_id" in ranked.columns else ranked.columns[1]

    # This tool explains a ranking, and a plan can hand it the evidence table
    # instead -- combine_evidence joins the specialists' output, and the step
    # that turns that join into an order is easy to leave out. It used to
    # raise KeyError: 'rank' from inside the loop, which reads as a code fault
    # and lost the whole question. A missing order is not a fault: it is a
    # policy nobody stated, so one is stated here, in the open, and applied
    # the same way prioritize_entities applies a given one.
    if "rank" not in ranked.columns or "priority_score" not in ranked.columns:
        # Each source that was ranked says so: rank_buildings emits a `rank`
        # column, and the join keeps it as "High EUI.rank". Those columns are
        # the orders the specialists already decided, so they are what an
        # assumed policy should combine -- position 1 is worst.
        #
        # Everything numeric is not a measure. The join also carries the
        # metadata each source travelled with, and taking all of it ranked a
        # cohort by floor area: the worst building was the biggest one, with
        # "floor area at 100% of the cohort's range" given as the reason. Size
        # is a denominator, not a finding.
        blocked = {"floor_area_m2", "year_built", "occupants", "n_points",
                   "n_observations", "n_buildings", "hours_observed",
                   "hours_in_year", "questionable_denominator", "rank",
                   "coverage_fraction", "n"}
        by_rank = [c for c in ranked.columns
                   if c != key and c.split(".", 1)[-1] == "rank"]
        numeric = by_rank or [
            c for c in ranked.columns
            if c != key and c.split(".", 1)[-1] not in blocked
            and pd.to_numeric(ranked[c], errors="coerce").notna().any()]
        if not numeric:
            return insufficient(
                "not_a_ranking",
                f"{priorities_handle} carries no order and no measure to build "
                f"one from: columns are {[c for c in ranked.columns if c != key]}. "
                f"Rank it with prioritize_entities first, naming the criteria "
                f"and their weights")
        scored = ranked[[key]].copy()
        share = 1.0 / len(numeric)
        total = None
        for column in numeric:
            values = pd.to_numeric(ranked[column], errors="coerce")
            lo, hi = values.min(), values.max()
            norm = (values - lo) / (hi - lo) if hi > lo else values * 0
            # A rank counts upward from the worst, so first place is the top
            # of the range, not the bottom.
            if column in by_rank:
                norm = 1 - norm
            scored[f"norm.{column}"] = norm.round(4)
            total = norm * share if total is None else total + norm * share
        scored["priority_score"] = total
        scored = scored.dropna(subset=["priority_score"])
        if scored.empty:
            return insufficient(
                "no_scored_candidates",
                f"none of the {len(ranked):,} candidates carry any of "
                f"{numeric}; nothing has been computed for them yet")
        scored = scored.sort_values("priority_score", ascending=False)
        scored.insert(0, "rank", np.arange(1, len(scored) + 1))
        ranked = scored
        assumed_policy = {c: round(share, 4) for c in numeric}
    else:
        assumed_policy = None

    lines, entries = [], []
    for _, row in ranked.head(limit).iterrows():
        entity = row[key]
        drivers = {c.replace("norm.", ""): float(row[c])
                   for c in ranked.columns if c.startswith("norm.")}
        top = sorted(drivers.items(), key=lambda kv: kv[1], reverse=True)[:3]
        reason = ", ".join(f"{name} ({value:.2f} normalised)" for name, value in top)
        lines.append(f"{int(row['rank'])}. {entity} - driven by {reason}")
        entries.append({"rank": int(row["rank"]), key: entity,
                        "priority_score": float(row["priority_score"]),
                        "drivers": dict(top)})
    recommendation = {
        "question": question,
        "recommended": entries,
        "narrative": "\n".join(lines),
        "limitations": [
            "Rankings are associations observed in metered data, not causal claims.",
            "No equipment, BMS, occupancy schedule or control data was available.",
            "Scores are normalised within this cohort, so they are not comparable "
            "across differently-constructed cohorts.",
        ],
    }
    if evidence is not None:
        recommendation["evidence_columns"] = [c for c in evidence.columns if c != key]
    if assumed_policy:
        recommendation["assumed_policy"] = assumed_policy
    # The same shortlist as prose, which is what a reader acts on. The raw
    # payload carries normalised driver scores and a `narrative` string that
    # composition prints as one long key-value line; this arrives as lines,
    # above the table, with the limitation that matters stated rather than
    # filed under "limitations" where nobody reads it.
    noun = key.replace("_", " ").replace(" id", "")
    narrative = [f"{len(entries)} {noun}(s) to investigate first, worst first."]
    for entry in entries[:5]:
        drivers = ", ".join(
            f"{name.replace('_', ' ')} at {value:.0%} of the cohort's range"
            for name, value in list(entry["drivers"].items())[:2])
        narrative.append(f"  {entry['rank']}. {entry[key]}   "
                         f"score {entry['priority_score']:.3f}"
                         + (f"   — {drivers}" if drivers else ""))
    if len(entries) > 5:
        narrative.append(f"  ... and {len(entries) - 5} more below")
    if assumed_policy:
        narrative.append(
            "  No ranking policy was given, so each measure was weighted "
            "equally and scaled min-max within this cohort: "
            + ", ".join(
                f"{(c.split('.', 1)[0] if c.split('.', 1)[-1] == 'rank' else c.split('.', 1)[-1]).replace('_', ' ')}"
                f" {w:.0%}" for c, w in assumed_policy.items())
            + ". State the weights with prioritize_entities to choose them.")
    narrative.append(
        "  These are associations in metered data, not diagnoses: no equipment, "
        "BMS, occupancy or control data was available, and the scores are "
        "normalised within this cohort only.")
    return ok(recommendation, evidence={"ranked": len(ranked),
                                        "reported": len(entries),
                                        "narrative": narrative,
                                        "headline": {
                                            "value": (entries[0][key]
                                                      if entries else None),
                                            "unit": None,
                                            "label": f"the {noun} to "
                                                     f"investigate first"}})




def _table_holding(ctx, columns: list[str]):
    """The one stored table of this run holding all these columns, or nothing.

    Two candidates is ambiguous and returns nothing, rather than guessing.
    """
    store = getattr(ctx, "store", None)
    if store is None or not hasattr(store, "handles"):
        return None
    found = []
    for handle in store.handles():
        try:
            frame = store.get(handle)
        except Exception:                                      # noqa: BLE001
            continue
        if all(c in frame.columns or _nearest_column(c, list(frame.columns))
               for c in columns):
            found.append(frame)
    return found[0] if len(found) == 1 else None
#: Column names that share no substring with the ones the frames use.
COLUMN_ALIASES = {
    "n_clusters": ("k",), "num_clusters": ("k",), "cluster_count": ("k",),
    "n_cluster": ("k",), "clusters": ("k",),
    "score": ("silhouette",), "silhouette_score": ("silhouette",),
}


def _nearest_column(wanted: str, columns: list[str]) -> str | None:
    """The one column a name plainly refers to, or nothing if it is ambiguous."""
    def key(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())

    target = key(wanted)
    exact = [c for c in columns if key(c) == target]
    if len(exact) == 1:
        return exact[0]
    near = [c for c in columns if key(c) in target or target in key(c)]
    if len(near) == 1:
        return near[0]
    aliased = [c for c in columns
               if any(key(a) == key(c)
                      for a in COLUMN_ALIASES.get(wanted.lower(), ()))]
    return aliased[0] if len(aliased) == 1 else None
@tool("reporting")
def create_visualization(
    ctx: ToolContext,
    handle: str,
    kind: Literal["line", "load_profile", "bar", "heatmap", "box", "scatter",
                  "map"] = "line",
    x: str | None = None,
    y: str | None = None,
    group: str | None = None,
    title: str | None = None,
) -> ToolResult:
    """Render one chart to the run artifact store. One tool, seven chart types."""
    df = _as_frame(ctx, handle)
    if df is None or df.empty:
        return empty("nothing to plot")
    # Checked once here rather than in each of the seven chart branches, so a
    # column the frame does not hold is named rather than becoming a KeyError.
    # A near miss on a column name should not cost the whole chart.
    # Columns absent here but present in exactly one sibling table mean the
    # chart was pointed at the wrong handle.
    wanted = [c for c in (x, y) if c]
    if wanted and any(c not in df.columns for c in wanted):
        better = _table_holding(ctx, wanted)
        if better is not None:
            df = better

    named = {}
    renamed = []
    for role, column in (("x", x), ("y", y), ("group", group)):
        if column and column not in df.columns:
            match = _nearest_column(column, list(df.columns))
            if match:
                renamed.append(f"{column} read as {match}")
                column = match
        named[role] = column
    x, y, group = named["x"], named["y"], named["group"]

    # A group only colours series, so an unresolvable one is dropped.
    if group and group not in df.columns:
        group = None
        named["group"] = None

    absent = {role: column for role, column in named.items()
              if column and column not in df.columns}
    if absent:
        return insufficient(
            "unknown_column",
            "this frame has no column "
            + ", ".join(f"{c!r} (asked for as {role})" for role, c in absent.items())
            + f"; it holds {list(df.columns)[:8]}. Name a column it has, or "
              "omit the argument and let the chart choose.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    try:
        if kind == "load_profile":
            xcol = x or ("hour_of_day" if "hour_of_day" in df else "hour")
            ycol = y or ("mean_demand" if "mean_demand" in df else "value")
            # A profile chart needs an hour to plot against. A frame that has
            # none is a summary, not a curve -- one row holding the
            # correlation between two profiles, say -- and the honest result
            # is to say so rather than raise KeyError at the reader.
            missing = [c for c in (xcol, ycol) if c not in df.columns]
            if missing:
                return insufficient(
                    "not_a_profile",
                    f"a load-profile chart needs {missing} and this frame has "
                    f"{list(df.columns)[:6]}; it summarises profiles rather "
                    f"than holding one, so there is no curve to draw")
            if group and group in df:
                for name, grp in df.groupby(group):
                    ax.plot(grp[xcol], grp[ycol], marker="o", ms=3, label=str(name))
                ax.legend(fontsize=8)
            else:
                ax.plot(df[xcol], df[ycol], marker="o", ms=3)
            ax.set_xlabel("hour of day (local)"); ax.set_ylabel("mean demand (kW)")
        elif kind == "line":
            xcol = x or ("timestamp" if "timestamp" in df else df.columns[0])
            ycol = y or ("value" if "value" in df else df.columns[-1])
            # A line chart needs a time axis, and a frame of summary statistics
            # has none: pd.to_datetime then raised DateParseError on a building
            # id and the whole run reported "the analysis failed". Say what is
            # missing instead, so a plan can be corrected rather than lost.
            if not pd.api.types.is_datetime64_any_dtype(df[xcol]):
                probe = pd.to_datetime(df[xcol], errors="coerce")
                if probe.isna().all():
                    plt.close(fig)
                    return insufficient(
                        "no_time_axis",
                        f"a line chart plots something against time and column "
                        f"{xcol!r} holds none; this frame has "
                        f"{list(df.columns)[:6]}. Name x explicitly, or plot a "
                        "frame that carries a timestamp.")
            if group and group in df:
                for name, grp in df.groupby(group):
                    ax.plot(pd.to_datetime(grp[xcol]), grp[ycol], lw=0.8, label=str(name))
                ax.legend(fontsize=8)
            else:
                ax.plot(pd.to_datetime(df[xcol]), df[ycol], lw=0.8)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        elif kind == "bar":
            xcol = x or df.columns[0]
            ycol = y or df.select_dtypes("number").columns[-1]
            plot = df.head(30)
            ax.bar(plot[xcol].astype(str), plot[ycol])
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.set_ylabel(ycol)
        elif kind == "heatmap":
            xcol = x or "hour_of_day"; ycol = y or "day_of_week"
            zcol = "mean_demand" if "mean_demand" in df else df.select_dtypes("number").columns[-1]
            pivot = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc="mean")
            im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            fig.colorbar(im, ax=ax, label=zcol)
        elif kind == "box":
            ycol = y or df.select_dtypes("number").columns[-1]
            if group and group in df:
                groups = [g[ycol].dropna().to_numpy() for _, g in df.groupby(group)]
                ax.boxplot(groups, tick_labels=[str(n) for n, _ in df.groupby(group)])
                ax.tick_params(axis="x", rotation=45, labelsize=7)
            else:
                ax.boxplot(df[ycol].dropna().to_numpy())
            ax.set_ylabel(ycol)
        elif kind == "scatter":
            xcol = x or df.select_dtypes("number").columns[0]
            ycol = y or df.select_dtypes("number").columns[-1]
            ax.scatter(df[xcol], df[ycol], s=8, alpha=0.6)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        else:  # map
            if not {"latitude", "longitude"} <= set(df.columns):
                return insufficient("no_coordinates",
                                    "map needs latitude and longitude columns")
            size = df.select_dtypes("number").columns[-1]
            points = ax.scatter(df["longitude"], df["latitude"],
                                c=df[size], s=40, cmap="viridis")
            fig.colorbar(points, ax=ax, label=size)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(title or f"{kind} chart")
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        path = ctx.store.root / f"figure_{kind}_{abs(hash(str(title))) % 10**6}.png"
        fig.savefig(path)
    finally:
        plt.close(fig)
    return ok({"figure_path": str(path), "kind": kind},
              evidence={"rows_plotted": len(df), "title": title})


@tool("reporting")
def generate_report(
    ctx: ToolContext,
    question: str,
    answer: str,
    evidence: dict | None = None,
    tables: list[str] | None = None,
    figures: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> ToolResult:
    """Assemble the final answer with its evidence, assumptions and provenance."""
    sections = [f"# {question}", "", answer, ""]
    if evidence:
        sections += ["## Evidence", ""]
        sections += [f"- **{k}**: {v}" for k, v in evidence.items()] + [""]
    if assumptions:
        sections += ["## Assumptions", ""]
        sections += [f"- {a}" for a in assumptions] + [""]
    attached = []
    for handle in tables or []:
        df = _as_frame(ctx, handle)
        if df is not None and not df.empty:
            attached.append({"handle": handle, "rows": len(df),
                             "columns": list(df.columns)})
            sections += [f"## Table ({handle})", "",
                         df.head(15).to_markdown(index=False), ""]
    if figures:
        sections += ["## Figures", ""] + [f"- {f}" for f in figures] + [""]
    sections += ["---",
                 f"_Snapshot: {ctx.snapshot_version} - run {ctx.store.run_id}_"]
    markdown = "\n".join(sections)
    path = ctx.store.root / "report.md"
    path.write_text(markdown)
    return ok({"report_path": str(path), "markdown": markdown},
              evidence={"tables": attached, "figures": figures or [],
                        "snapshot": ctx.snapshot_version})
