"""Interaction & Intent, and Planning.

Two planners share one output type, the plan DAG:

* :class:`DeterministicPlanner` pattern-matches a grounded intent onto a tool
  chain with no model at all. It makes the pipeline runnable and testable today
  and serves as the floor baseline in the results table -- what the tools alone
  achieve without agentic reasoning.
* :class:`LLMPlanner` asks a model for the same structure, validates it against
  the registry, and falls back to the deterministic plan when the model returns
  something unusable.

Intent resolution is **grounded**: cities, countries, sites and building types
are read from the catalog, so a question about a city that does not exist is
caught before any tool runs rather than producing a confidently empty answer.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from dataclasses import dataclass

import pandas as pd

from ..tools import registry
from .plan import Plan, PlanStep
from .state import Intent

WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
                "fifteen": 15, "twenty": 20}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
SEASONS = {"summer": (6, 8), "winter": (12, 2), "spring": (3, 5), "autumn": (9, 11),
           "fall": (9, 11)}

# A semester is not a season. Spring is March to May; the spring semester runs
# January to May, and the fall semester September to December -- the academic
# calendar of the campuses in this snapshot. July belongs to neither, so a
# question about semesters excludes the summer rather than splitting it.
# Reading "the spring semester" as the season puts the boundary two months out.
SEMESTERS = {"fall": [9, 10, 11, 12], "autumn": [9, 10, 11, 12],
             "spring": [1, 2, 3, 4, 5]}


def resolve_semesters(text: str) -> list[tuple[str, list[int]]]:
    """The semesters a question names, in the order it names them.

    Only where the question says "semester" or "term": otherwise "spring" is
    the season and SEASONS has it.
    """
    lowered = (text or "").lower()
    if not re.search(r"\bsemesters?\b|\bterms?\b", lowered):
        return []
    found = [(name, re.search(rf"\b{name}\b", lowered).start())
             for name in SEMESTERS if re.search(rf"\b{name}\b", lowered)]
    seen, ordered = set(), []
    for name, _ in sorted(found, key=lambda pair: pair[1]):
        months = tuple(SEMESTERS[name])
        if months in seen:          # "fall" and "autumn" are one semester
            continue
        seen.add(months)
        ordered.append((name, SEMESTERS[name]))
    return ordered

# The last calendar year the snapshot covers end to end. "Annual" with no year
# named means this one.
LATEST_FULL_YEAR = 2017

# question shape -> (task category, primary intent tag)
PATTERNS: list[tuple[str, str, str]] = [
    # Specific intents first. `rank` is deliberately near the end: words like
    # "most" and "highest" appear in many questions that are not rankings.
    # Graph-shaped questions: relationships, what is already known, and
    # standards export. These need no measurement scan at all.
    (r"(how\s+(is|are)\s+.*\brelated\b|relationship between .* and |"
     r"what\s+connects|how\s+.*\bconnected\b)", "1", "entity_path"),
    (r"(what do we (already )?know|previously computed|cached (fact|result)|"
     r"derived facts?|stored findings?)", "1", "known_facts"),
    (r"(export .*(brick|haystack|bedes)|as (brick|haystack|bedes)|"
     r"(brick|haystack|bedes) (model|format|export))", "1", "semantic_export"),
    (r"(prioriti[sz]\w*|investigat\w*\s+first|should\s+(be\s+)?"
     r"(investigat\w*|audit\w*|look\w*\s+at)|worth\s+investigating|"
     r"where\s+should\s+.*(start|focus))", "12", "prioritize"),
    # `\bforecast\b` does not match "forecasting", which is how the word is
    # most often used -- "the LightGBM forecasting model" -- so a question
    # about forecast accuracy fell past this tag entirely and was claimed by
    # the weather-value tag on the strength of naming air temperature as a
    # model input. The suffix is the whole fix.
    # A question may name forecasting as its *purpose* -- "fitness-for-use in
    # advanced load forecasting" is about metadata completeness -- and this
    # tag claims it. That costs nothing: the named metadata route is tried
    # before tag dispatch and answers it correctly, and a report takes its
    # category from the request rather than from here.
    (r"\b(forecast\w*|predict\w*|next \d+ ?(hour|day)|tomorrow)\b",
     "9", "forecast"),
    (r"\b(anomal\w*|unusual|abnormal|suspicious)\b", "10", "anomaly"),
    # An anomaly question need not use the word. These three shapes are how
    # the question is actually put -- a multiple of the usual ("exceed twice
    # its typical daily average"), a distance from it in standard deviations,
    # and a departure of the day's *shape* from the building's own profile --
    # and none of them matched, so each was answered by a statistics tool that
    # reported the mean it was compared against and never made the comparison.
    (r"\b(?:exceed\w*|above|over|more\s+than|greater\s+than)\b[^.?]{0,60}"
     r"\b(?:twice|double|triple|\d+(?:\.\d+)?\s*(?:x|times))\b[^.?]{0,40}"
     r"\b(?:typical|usual|normal|average|baseline|mean)\b"
     r"|\b(?:twice|double|\d+(?:\.\d+)?\s*(?:x|times))\s+(?:its|their|the)\s+"
     r"(?:typical|usual|normal|average|baseline|mean)\b",
     "10", "anomaly"),
    (r"\b\d+(?:\.\d+)?\s*(?:standard\s+deviations?|sigma|sds?)\b[^.?]{0,40}"
     r"\b(?:from|above|below|away)\b|\bmore\s+than\s+\d+(?:\.\d+)?\s*"
     r"(?:standard\s+deviations?|sigma)\b", "10", "anomaly"),
    (r"\bload\s+shape\b[^.?]{0,60}\bdeviat\w+|\bdeviat\w+[^.?]{0,60}"
     r"\b(?:typical|usual|normal)\s+(?:weekday\s+)?(?:profile|load\s+shape|shape)\b"
     r"|\birregular\s+load[-\s]shape\b", "10", "anomaly"),
    # A "change-point temperature" is a thermal balance point, not a regime
    # shift, so the generic pattern must not claim it.
    (r"(change[- ]?point(?!\s+temperature)\w*|shifted|regime|behaviou?r change|"
     r"when did .*change)", "10", "change_point"),
    # ...and neither does a structural break have to be called a change point.
    # "Detect whether the baseline has undergone a structural change (e.g., a
    # step change)" names the thing twice and matched on neither.
    (r"\bstructural\s+(?:change|break|shift)\b|\bstep[-\s]?change\b"
     r"|\blevel\s+shift\b|\bbreak\s?points?\b"
     r"|\bbaseline\b[^.?]{0,60}\bunder(?:gone|go|went)\b", "10", "change_point"),
    # "Segmented by subindustry" says how to group an answer by a column that
    # already exists; clustering discovers groups that do not. The word alone
    # sent "day-to-day variability segmented by subindustry" to
    # cluster_timeseries, which returned cluster numbers per building.
    (r"\b(cluster\w*|segment\w*)(?!\w)(?!\s+by\s+\w)|group .*(profile|pattern)",
     "11", "cluster"),
    (r"\b(similar\w*|most like|resembl\w*)\b", "11", "similar"),
    # Similarity asked as a correlation between two buildings' load shapes.
    # "Whose average daily load shape correlates most strongly with X's" is
    # exactly what find_similar_timeseries(metric="correlation") computes, and
    # it was being answered by drawing X's own profile.
    (r"\b(?:correlat\w*|matches?|aligns?)\b[^.?]{0,40}\bmost\s+(?:strongly|closely)\b"
     r"|\bmost\s+(?:strongly|closely)\s+correlat\w*\b"
     r"|\bclosest\s+(?:match|load\s+shape|profile)\b"
     r"|\bnearest\s+(?:neighbou?rs?|profiles?)\b", "11", "similar"),
    # A weather threshold makes a question a weather question even when no
    # relationship word appears in it: "did consumption spike during the
    # extreme heatwave (temperatures > 35C)" was read as plain consumption and
    # answered with July's total.
    (r"\bheat\s?waves?\b|\bcold\s?snaps?\b|\bextreme\s+(?:heat|cold)\b"
     r"|\b(?:temperature|weather)s?\b[^.?]*(?:above|over|exceed\w*|greater\s+than|>)"
     r"\s*(?:a\s+)?\d+(?:\.\d+)?\s*(?:°\s*)?c\b",
     "6", "weather_relationship"),
    # The weather *while* a building was busy, which is a different question
    # from the weather at its site: "the average outdoor temperature during
    # the top 5% of demand hours" selects hours by consumption and then reads
    # the weather over them. Matched before the plain weather-variable rule,
    # which would answer with the site's average over the whole window.
    (r"\b(?:during|while|when)\b[^.?]*\b(?:top|highest|peak)\b[^.?]*"
     r"\b(?:demand|consumption|load)\b|\b(?:top|highest|peak)\b[^.?]*"
     r"\b(?:demand|consumption|load)\b[^.?]*\bhours?\b",
     "6", "weather_during_peak"),
    (r"(weather[- ]normali[sz]\w*|after (accounting|controlling) for weather|"
     r"explained by (hotter|colder|the) weather)", "6", "weather_normalize"),
    (r"(change[- ]?point temperature|balance temperature|"
     r"(cooling|heating) (sensitivity|slope))", "6", "temperature_response"),
    # Relationship first: a lookahead cannot exclude a word that appears
    # *before* the variable, and "is X sensitive to outdoor temperature" puts it
    # there. Ordering decides it instead.
    (r"\b(sensitiv\w*|correlat\w*|relationship|depend\w*|driv\w*|respond\w*|"
     r"response|normali[sz]\w*|change[\s-]?point|balance[\s-]?point)\b"
     r".*\b(temperature|weather|dew|humidity|wind|cloud|precip\w*)\b"
     r"|\b(temperature|weather|dew|humidity|wind|cloud|precip\w*)\b.*"
     r"\b(sensitiv\w*|correlat\w*|relationship|depend\w*|driv\w*|respond\w*|"
     r"response|normali[sz]\w*|change[\s-]?point|balance[\s-]?point)\b",
     "6", "weather_relationship"),
    # Then a weather variable asked about for its own value -- "what was the
    # average air temperature at site Bobcat" -- which otherwise fell through to
    # the default tag and was answered with electricity statistics.
    # ...but not when the variable is named as something a model is *given*.
    # "Adding air temperature to the LightGBM forecasting model" and "trained
    # on historical air temperature" name an input, not a subject, and asking
    # for its mean answers a question about the weather where one about a
    # forecast was asked.
    (r"(?<!adding )(?<!add )(?<!with )(?<!using )(?<!on )"
     r"\b(air[\s_]?temperature|outdoor[\s_]?temperature|dew[\s_]?point|"
     r"dew[\s_]?temperature|wind[\s_]?speed|wind[\s_]?direction|"
     r"cloud[\s_]?cover\w*|precipitation|precip|sea[\s_]?level[\s_]?pressure|"
     r"humidity)\b", "6", "weather_value"),
    # A year-on-year comparison names two things. "than" followed by a number
    # is a threshold -- one thing and a bound -- and a comparison word beside a
    # "between" is not enough either: a year has to be in play.
    (r"\b(?:more|less|higher|lower|greater|increase\w*|decrease\w*|change\w*)\b"
     r"[\w\s,'%-]{0,40}\b(?:than(?!\s+\d)|versus|vs\.?|compared\s+to|between)\b"
     r"[\w\s,'-]{0,25}\b(?:20\d\d|years?|last\s+year|previous\s+year)\b"
     r"|\byear[\s-]?(?:on|over)[\s-]?year\b"
     r"|\b(?:compare|contrast)\b[\w\s,'-]{0,30}\b(?:years?|20\d\d)\b"
     r"|\b20\d\d\b[\w\s,'-]{0,20}\b(?:than|versus|vs\.?)\b[\w\s,'-]{0,10}\b20\d\d\b",
     "3", "year_over_year"),
    # "The impact of A on B" is a question about B. "Quantify the impact of
    # missing values on the calculation of annual EUI" names a data condition
    # first and the measure second, so the quality pattern below claimed it and
    # the router sent it to the data-quality specialist, which profiles gaps
    # and can never say what they cost. What a question asks to *compute* is
    # the thing after "on", not the condition before it.
    (r"\b(?:impact|effect|influence|consequences?)\b[\w\s,'-]{0,60}\bon\b"
     r"[\w\s,'-]{0,40}\b(?:eui|energy\s+use\s+intensity|energy\s+intensity|"
     r"kwh/m|per\s+(?:square|m2|sqm))\b", "5", "eui"),
    # Zeros in this dataset are the cleaning mask, so a question about zero
    # readings is a data-quality question however it words them. "zero runs"
    # alone missed "consecutive days of zero electricity readings", which then
    # fell through to consumption and was answered with statistics over the
    # readings it was asking about the absence of.
    (r"\b(missing|coverage|data quality|gaps?|incomplete|"
     r"zero[\s-]?(?:runs?|readings?|values?)|consecutive\s+days?\s+of\s+zero|"
     r"completeness|duplicat\w+|timestamp\w*)\b", "2", "quality"),
    # A geography question that names a metric is an aggregation, not a
    # catalog listing -- this must precede the generic discovery pattern.
    (r"which \b(cit(?:y|ies)|countr(?:y|ies)|sites?)\b.*\b(eui|energy intensity|"
     r"consumption|demand|kwh|intensity|load)\b", "8", "compare_geo"),
    # Which calendar periods hold data is a catalog question, not a scan.
    # "What years does this snapshot cover" matches none of the generic
    # discovery forms below, so it fell through to consumption and answered
    # with statistics over 490 buildings.
    (r"\b(?:what|which|how\s+many)\s+(?:calendar\s+)?years?\b"
     r"|\b(?:what|which)\s+(?:time\s+)?(?:period|date\s+range|time\s+range|span)\b"
     r"|\byears?\s+(?:covered|of\s+(?:data|readings|measurements))\b"
     r"|\btime\s+span\b", "1", "discovery"),
    (r"\b(how many|which (cit\w+|countr\w+|sites?|datasets?)|list |"
     r"available|what data)\b", "1", "discovery"),
    # "What percentile is this building in" is a peer question; "the 5th
    # percentile of demand" is a level of one building's own distribution, and
    # routing it here sent a baseload question to identify_peer_outliers. A
    # number in front of the word is what tells the two apart.
    (r"\b(peer\w*|comparable|benchmark\w*)\b|"
     r"(?<!\dst\s)(?<!\dnd\s)(?<!\drd\s)(?<!\dth\s)(?<!\d\s)(?<!\dst-)"
     r"(?<!\dnd-)(?<!\drd-)(?<!\dth-)(?<!\d-)\bpercentile\b|"
     r"similar (building|office|school)s?|"
     r"compare\w*\s+(with|to|against)\s+(other|similar|comparable)",
     "7", "benchmark"),
    # "For site Fox" pins the scope; it does not make the question a
    # comparison *between* sites. "Compare the average weekday load profile
    # versus the weekend load profile for site Fox" was read as a geographic
    # comparison on the strength of that trailing phrase, and answered with a
    # cross-site aggregate where two curves for one site were asked for. The
    # same distinction the output-format rules already draw: a named site is a
    # filter, an unnamed one is a dimension.
    (r"(compare|versus|vs\.?|difference between).*"
     r"(?<!for\s)(?<!at\s)(?<!in\s)(?<!within\s)"
     r"\b(cit(?:y|ies)|countr(?:y|ies)|sites?)\b"
     r"|(across|between)\s+\b(cit(?:y|ies)|countr(?:y|ies)|sites?)\b",
     "8", "compare_geo"),
    # A per-building peak *hour* is a load-shape question; a cross-geography one
    # is spatio-temporal. The branch below picks based on whether a building is
    # named.
    (r"(peak (timing|hour|time)|when .*peak|"
     r"(at )?what (local )?hour .*(peak|highest|maximum))", "8", "peak_timing"),
    (r"per[- ](occupant|person|capita|head)|kwh/person", "5", "per_occupant"),
    # A "/m2" of any unit is a question about intensity, not about demand.
    # "Peak demand exceeding 50 W/m2" went to the demand route and came back
    # with 604 rows of unnormalised kW, where the whole question is the
    # division.
    (r"\b(eui|energy use intensity|energy intensity|kwh/m|"
     r"per (square|m2|sqm))\b|\b[kmw]?w(?:h)?\s*(?:/|per\s*)\s*m\s*(?:2|²)",
     "5", "eui"),
    # "Night-time load" was the only spelling this matched, so "unusually high
    # nighttime CONSUMPTION" fell through to consumption statistics and came
    # back as a general table with no night/day split in it at all -- an answer
    # to a question nobody asked. The measured noun varies and the window is
    # what matters, so every way of naming demand is admitted, plus the bare
    # "at night".
    (r"\b(baseload|base\s?load|overnight)\b"
     r"|\bnight[-\s]?time\s+(load|consumption|demand|usage|use|energy|kwh)\b"
     r"|\bnight\s+(load|consumption|demand|usage|use|energy)\b"
     r"|\b(consumption|demand|load|usage|energy)\s+at\s+night\b"
     r"|\bat\s+night\b", "4", "baseload"),
    (r"(operating (hours|schedule|pattern)|after[- ]hours|start[- ]?up|shutdown|"
     r"business\s+hours|working\s+hours|outside\s+\w*\s*hours|"
     r"ramp\w*\s+up|start\w*\s+ramp|open\w*.*close\w*)", "4", "operating"),
    # Two seasons named in one question is a seasonal contrast however it is
    # joined -- "summer vs winter", "between winter and summer" -- and it has
    # to be recognised before "daily profile" sends it to the profile route,
    # which draws one curve and has nothing to compare it against.
    (r"\b(weekday|weekend|seasonal)\b"
     r"|\b(?:summer|winter)\b[\w\s,'-]{0,14}\b(?:winter|summer)\b",
     "4", "compare_temporal"),
    # "Intraday consumption pattern", "24-hour pattern", "average daily shape"
    # are all the load profile said another way, and the narrow form sent them
    # to consumption statistics -- one mean where a curve was asked for.
    (r"(load (profile|shape)|daily profile|hourly profile"
     r"|intra[- ]?day\s+\w*\s*pattern|24[- ]?hour\s+\w*\s*pattern"
     r"|(?:daily|hourly|average)\s+\w*\s*(?:pattern|shape|curve))",
     "4", "load_profile"),
    (r"(peak demand|load factor|peak[- ]to[- ]average)", "3", "demand"),
    (r"\b(highest|lowest|top \d+|bottom \d+|rank\w*|most|least|"
     r"worst|best)\b", "5", "rank"),
    (r"\b(total|consumption|how much|average|median|statistic\w*)\b",
     "3", "consumption"),
]


# Subtype values that are also ordinary words a question uses for other
# things. "Research", "Services" and "Other" name a subtype in this catalog
# and name nothing in particular in a sentence.
GENERIC_SUBTYPE_WORDS = {"other", "research", "services", "general", "mixed",
                         "unknown", "none", "shop", "plant", "central"}


@dataclass
class Vocabulary:
    countries: set[str]
    cities: set[str]
    sites: set[str]
    building_types: set[str]
    building_ids: set[str]
    # The finer type dimensions. Subtypes are only kept where they cannot be
    # confused with something else the question might mean: 104 of them exist
    # and several -- Office, Education, Library, Research, Classroom -- are
    # also a canonical type, an industry, or an ordinary English word, and
    # matching those would turn "office buildings" into a subtype filter of
    # 306 where the type holds 307.
    # Which site each building actually belongs to, so a question that names
    # both can be checked for agreeing with itself.
    site_of_building: dict = None
    # One side of a slashed subtype label -> the full labels it can reach.
    subtype_sides: dict = None
    building_subtypes: set[str] = None
    industries: set[str] = None
    subindustries: set[str] = None
    facts: dict = None
    years: list = None

    @classmethod
    def from_db(cls, db) -> "Vocabulary":
        def column(sql: str) -> set[str]:
            return {str(v) for v in db.df(sql).iloc[:, 0].dropna().tolist()}
        vocab = cls(
            countries=column("SELECT DISTINCT country_canonical FROM sites"),
            cities=column("SELECT DISTINCT city FROM sites"),
            sites=column("SELECT DISTINCT site_id FROM sites"),
            building_types=column(
                "SELECT DISTINCT building_type_canonical FROM buildings"),
            building_ids=column("SELECT building_id FROM buildings"),
        )
        vocab.site_of_building = dict(
            db.df("SELECT building_id, site_id FROM buildings "
                  "WHERE site_id IS NOT NULL").itertuples(index=False,
                                                          name=None))
        types_lower = {t.lower() for t in vocab.building_types}
        industries = column(
            "SELECT DISTINCT industry FROM buildings WHERE industry IS NOT NULL")
        subindustries = column(
            "SELECT DISTINCT subindustry FROM buildings"
            " WHERE subindustry IS NOT NULL")
        subtypes = column(
            "SELECT DISTINCT building_subtype FROM buildings"
            " WHERE building_subtype IS NOT NULL")
        # A subtype earns a place in the vocabulary by being unmistakable:
        # more than one word, or a single word that is not already a type, an
        # industry, or one of the words a question uses for something else.
        vocab.building_subtypes = {
            s for s in subtypes
            if (" " in s or "-" in s or "/" in s
                or (s.lower() not in types_lower
                    and s.lower() not in {i.lower() for i in industries}
                    and s.lower() not in GENERIC_SUBTYPE_WORDS))}
        # A compound label joined by slashes covers both its sides:
        # "Entertainment/public assembly" is one category, and the slash means
        # and/or rather than a choice -- the data does not say which of the 84
        # "Primary/Secondary Classroom" buildings is primary. So the label is
        # left exactly as BDG2 publishes it, and each side is recorded as a
        # way of *reaching* it. Someone asking about dormitories should find
        # "Residence Hall/Dormitory" without having to spell the whole thing.
        sides: dict = {}
        for value in subtypes:
            if "/" not in value:
                continue
            head, _, tail = value.partition(" - ")
            for piece in (tail or head).split("/"):
                word = piece.strip().lower()
                if (len(word) > 3 and word not in types_lower
                        and word not in GENERIC_SUBTYPE_WORDS):
                    sides.setdefault(word, set()).add(value)
        vocab.subtype_sides = sides
        # An industry that is spelled like a canonical building type is not
        # resolvable from a question: "education buildings" means the type,
        # which holds 604, not the industry, which holds 491, and resolving
        # both filtered a cohort twice over. The industry is still reachable
        # as a *grouping* -- "by industry" -- and as an explicit tool
        # argument; what is dropped is guessing it from a bare word.
        vocab.industries = {i for i in industries
                            if i.lower() not in types_lower}
        vocab.subindustries = {s for s in subindustries
                               if s.lower() not in types_lower}
        span = db.df("""
            SELECT e.dataset_id,
                   MIN(a.period_start) AS lo, MAX(a.period_start) AS hi,
                   COUNT(DISTINCT a.building_id) AS n
            FROM agg_daily a
            JOIN (SELECT DISTINCT building_id, dataset_id FROM electricity
                  WHERE quality_flag = 'OK') e
              USING (building_id)
            GROUP BY 1 ORDER BY 1""")
        vocab.years = [int(y) for y in db.df(
            "SELECT DISTINCT YEAR(period_start) AS y FROM agg_yearly"
            " ORDER BY 1")["y"].tolist()]
        vocab.facts = {
            "datasets": [{"dataset": r.dataset_id, "start": str(r.lo)[:10],
                          "end": str(r.hi)[:10], "buildings": int(r.n)}
                         for r in span.itertuples()],
            "resolution": "hourly",
        }
        return vocab

    def grounding(self) -> str:
        """What the snapshot actually contains -- stated, not guessed.

        Without this a model invents plausible-but-absent values: a request for
        2023 against a 2016-2017 snapshot returns EMPTY, which is
        indistinguishable from a genuinely empty result.
        """
        facts = self.facts or {}
        spans = "; ".join(
            f"{d['dataset']} {d['start']}..{d['end']} ({d['buildings']} buildings)"
            for d in facts.get("datasets", []))
        # Two populations, and a plan that confuses them truncates its own
        # answer. The measurement span names the buildings with readings, and a
        # metadata question was capped at that number -- 1,576 of 1,636 -- so
        # sixty buildings that do have floor area were left out of an answer
        # about which buildings have floor area.
        return (
            "Snapshot facts (do not use values outside these):\n"
            f"- buildings in the catalog: {len(self.building_ids)}. This is the "
            f"population for any question about metadata or attributes, and the "
            f"floor a limit must clear\n"
            f"- measurement coverage by dataset: {spans}. That count is the "
            f"buildings with readings, which is fewer than the catalog holds. A "
            f"period outside a dataset's range returns nothing for that dataset\n"
            f"- countries: {sorted(c for c in self.countries if c)}\n"
            f"- cities: {sorted(c for c in self.cities if c)}\n"
            f"- building types: {sorted(t for t in self.building_types if t)}\n"
            f"- electricity is whole-building only; no circuit or appliance data\n"
            # An annual figure over a snapshot holding more than one year has to
            # pick one, and nothing said which. Single mode picked 2017 and
            # multi picked 2016 for the same question, so the same system gave
            # two different right-looking answers depending on the mode. The
            # most recent complete year is the ordinary convention for "annual",
            # and naming it makes the choice reproducible instead of arbitrary.
            f"- calendar years with readings: {self.years or []}. A question asking "
            f"for an annual figure without naming a year means the most recent "
            f"complete one, {self.years[-1] if self.years else 'n/a'}"
        )


# Quantities this snapshot does not hold. scopes EnergyX to electricity;
# a question about gas, water or indoor conditions has no answer here, and
# answering it with electricity instead is the worst failure the system can
# produce -- it looks exactly like a correct answer.
OUT_OF_SCOPE = (
    (r"\bgas\b|\bnatural gas\b", "gas consumption"),
    (r"\bwater\b(?!\s*heater)|\bhotwater\b|\bchilled\s*water\b", "water use"),
    (r"\bsteam\b", "steam"),
    (r"\bsolar\b|\bpv\b|\bgeneration\b", "on-site generation"),
    (r"\birrigation\b", "irrigation"),
    (r"\bindoor\b.*\b(temperature|humidity|air)\b"
     r"|\b(thermal\s+)?comfort\b|\bco2\b|\boccupancy sensor\b", "indoor conditions"),
    (r"\b(cost|price|tariff|bill|billing|\$|usd|gbp|eur)\b", "cost or tariff"),
    (r"\b(emission|carbon|co2e|ghg)\b", "emissions"),
)

# What the answer has to be about. Order matters: the most specific noun wins,
# and a scope phrase like "in this dataset" is never the subject.
SUBJECTS = (
    # Which calendar years hold data. Anchored on the interrogative forms so a
    # period used as a filter -- "total over the period 2017" -- is not caught,
    # and placed first because "what years does this snapshot cover" otherwise
    # resolves to the dataset and is answered with a 13-column provenance row.
    ("years",
     r"\b(?:what|which|how\s+many)\s+(?:calendar\s+)?years?\b"
     r"|\byears?\s+(?:of\s+(?:data|readings|measurements)\b"
     r"|are\s+(?:available|covered|present)\b|does\s+(?:it|this)\b)"
     r"|\b(?:what|which)\s+(?:time\s+)?(?:period|span|date\s+range|time\s+range)\b"
     r"|\btime\s+span\b|\byears?\s+covered\b"),
    # "types of buildings", "kinds of building", "building types", "usage
    # categories" -- the subject is the category, not the buildings in it.
    # Without the "types of X" form, "how many unique types of buildings in US"
    # counted 965 buildings instead of the categories they fall into.
    ("building_types",
     r"\bbuilding[ _]types?\b|\b(?:unique\s+|distinct\s+)?"
     r"(?:types?|kinds?|categor(?:y|ies)|classes)\s+of\s+buildings?\b"
     # "primary space usage category" is the same thing said the way the BDG2
     # paper says it, and the building-only form missed it: the question was
     # answered with a building, Eagle_health_Lucinda, where a category was
     # asked for.
     r"|\b(?:building|space|primary\s+space)\s+(?:usage|use)\s+categor"
     r"|\busage\s+categor|\bspace\s+usage\b"),
    ("buildings", r"\bbuildings?\b|\bmeters?\b|\bpremises\b"),
    ("sites", r"\bsites?\b|\bcampus(es)?\b"),
    ("cities", r"\bcit(?:y|ies)\b"),
    ("countries", r"\bcountr(?:y|ies)\b"),
    ("datasets", r"\bdatasets?\b|\bsnapshot\b"),
)


# Groups of countries a question can name. The catalog holds countries, so
# "in Europe" matched nothing and was dropped in silence: an education question
# scoped to Europe was answered with 604 buildings across four countries where
# 101 qualify. A region is not a country and cannot be guessed from one, so the
# membership is written down.
REGIONS = {
    "europe": ("UK", "Ireland"),
    "north america": ("US", "Canada"),
    "britain": ("UK",),
    "british isles": ("UK", "Ireland"),
    "the americas": ("US", "Canada"),
}
REGION_PATTERN = re.compile(
    r"\b(europe|european|north america|north american|britain|british isles|"
    r"the americas)\b", re.I)


def resolve_region(text: str, known: set[str] | None = None) -> list[str] | None:
    """The countries a named region covers, restricted to those in the catalog.

    Returns None when no region is named, and an empty list when a region is
    named that this snapshot holds nothing for -- which is a refusal, not a
    reason to answer about everywhere.
    """
    found = REGION_PATTERN.search(text or "")
    if not found:
        return None
    name = found.group(1).lower().rstrip("n").replace("europea", "europe")
    countries = REGIONS.get(name) or REGIONS.get(found.group(1).lower()) or ()
    if known is not None:
        countries = tuple(c for c in countries if c in known)
    return list(countries)


def resolve_scope(text: str) -> str | None:
    """The unavailable quantity a question asks for, if any."""
    lowered = text.lower()
    if re.search(r"\belectric|\bkwh\b|\bdemand\b|\bload\b|\benergy use\b", lowered):
        # An electricity question that merely mentions another fuel in passing
        # ("electricity, not gas") is in scope.
        pass
    for pattern, label in OUT_OF_SCOPE:
        if re.search(pattern, lowered):
            return label
    return None


def resolve_subject(text: str) -> str | None:
    """The entity the answer must be about."""
    lowered = text.lower()
    for name, pattern in SUBJECTS:
        if re.search(pattern, lowered):
            return name
    return None


# A data requirement stated in a question maps to a capability predicate the
# catalog already carries. "office buildings that have both electricity and
# complete weather data" ran consumption statistics over 296 buildings instead
# of filtering on can_weather_analysis, which answers it without touching a
# measurement.
REQUIREMENT_PHRASES = (
    ("weather_analysis",
     r"\b(?:complete\s+|matching\s+|available\s+|valid\s+)?weather\s+data\b"
     r"|\bweather\s+(?:coverage|availability|records?)\b"
     r"|\bmatching\s+weather\b"),
    ("eui_analysis",
     r"\beui\b|\benergy\s+use\s+intensity\b"
     # "floor-area metadata available": hyphenated, words in between.
     r"|\b(?:floor[\s-]*area|gross[\s-]*floor)\b[\w\s-]{0,24}"
     r"\b(?:available|known|complete|present|recorded|populated)\b"
     r"|\bnormali[sz]ed\s+by\s+(?:floor\s*)?area\b"),
    ("occupancy_normalisation",
     r"\bper[\s-]occupant\b"
     r"|\boccupan(?:cy|t)\s+(?:counts?|data|numbers?)\b"),
    ("consumption_analysis",
     r"\belectricity\s+(?:meter\s+)?data\b|\bconsumption\s+data\b"
     r"|\bmeter\s+data\b|\belectricity\s+records?\b"
     r"|\bhave\s+electricity\b|\bwith\s+electricity\b"),
    ("spatial_analysis",
     r"\bcoordinates?\b|\blat(?:itude)?\s*/?\s*lon|\bgeo(?:graphic|located)\b"
     r"|\bmapp?(?:ed|able)\b"),
    ("forecasting", r"\bforecast(?:able|ing)?\b|\bpredict(?:able|ion)\b"),
)

# Dimensions a question can group by, and the catalog entity that answers it.
# Catalog values that are also ordinary words. They name a building type only
# when the sentence uses them as one.
AMBIGUOUS_TYPE_WORDS = {"other", "services", "utility", "parking"}

GROUP_DIMENSIONS = (
    ("sites", r"\bsites?\b|\bcampus(?:es)?\b"),
    ("cities", r"\bcit(?:y|ies)\b"),
    # The snapshot's geography is country -> city -> site, so "global
    # regions" is the country level. A question asking for a matrix "across
    # all global regions" is asking for one row per country.
    ("countries", r"\bcountr(?:y|ies)\b|\bnations?\b|\bregions?\b"),
    ("building_types", r"\bbuilding[ _]types?\b|\busage\s+types?\b"),
    # Subindustry is matched before industry, because "subindustry" contains
    # "industry" and the coarser reading would otherwise claim both.
    ("subindustry", r"\bsub[ _]?industr(?:y|ies)\b"),
    ("industry", r"\bindustr(?:y|ies)\b"),
    ("building_subtype", r"\bsub[ _-]?type\b|\bsub[ _-]?categor(?:y|ies)\b"
                        r"|\bbuilding\s+sub[\w]*\b"),
)


def resolve_requirements(text: str) -> list[str]:
    """Capability predicates a question asks the cohort to satisfy."""
    lowered = text.lower()
    # Only a question that states a requirement -- "have", "with", "possess",
    # "available" -- is filtering. A question that merely mentions weather is
    # asking about it, not filtering on it.
    if not re.search(r"\b(?:have|has|having|with|possess(?:es|ing)?|contain(?:s|ing)?"
                     r"|available|provide[sd]?|complete|both|and)\b", lowered):
        return []
    found = []
    for name, pattern in REQUIREMENT_PHRASES:
        if re.search(pattern, lowered) and name not in found:
            found.append(name)
    return found


def resolve_group_by(text: str) -> str | None:
    """The dimension a question wants one row per, if any."""
    lowered = text.lower()
    # The dimension does not always follow the grouping word directly.
    # "across all global regions" puts two words in between, and matching only
    # the next token read the dimension as "all" and reported no grouping at
    # all -- the question asked for a matrix by country and was answered with
    # one portfolio-wide figure.
    explicit = re.search(r"\b(?:per|by|for each|each|across|grouped by)\b", lowered)
    if explicit:
        window = lowered[explicit.end():explicit.end() + 40]
        for name, pattern in GROUP_DIMENSIONS:
            found = re.search(pattern, window)
            if not found:
                continue
            # "Across all industrial sites" names the population, not the
            # grouping: industrial sites are the industrial buildings, and
            # reading it as one row per site answered a question about one
            # figure with eighteen. "Across all global regions" has no such
            # qualifier and does name a dimension.
            before = window[:found.start()]
            if re.search(r"\b(?:assembly|education|food\s*service|health\w*|"
                         r"industrial|lodging|office|parking|public\s*services|"
                         r"religious\w*|retail|services|technology\w*|utility|"
                         r"warehouse|commercial|residential)\s*$", before):
                continue
            return name
    # "how many education buildings are available at <dimension>" -- a grouping
    # dimension named without a specific value is a request for one row each.
    if re.search(r"\bhow many\b", lowered):
        for name, pattern in GROUP_DIMENSIONS:
            if re.search(rf"\b(?:at|in|across|for|by)\b[^.?]*?(?:{pattern})", lowered):
                return name
    return None


# Braces are doubled because this template is str.format()ed with the month
# vocabulary; a bare {1,2} would be read as a field name.
DAY_MONTH = (r"(?:(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})|({months})\s+"
             r"(\d{{1,2}})(?:st|nd|rd|th)?)(?:[,\s]+(20\d\d))?")


def _spoken_month_range(text: str) -> dict | None:
    """A period given as two months, with no days: "January 2016 to June 2016".

    The day-level range parser deliberately requires a day at both ends, and
    the single-month branch matches the FIRST month it sees -- so a six-month
    window collapsed to its opening month and a question about 4,368 hours was
    answered from 744. The end month is inclusive as spoken, so the stored
    half-open end is the first day of the month after it.
    """
    names = "|".join(MONTHS)
    pattern = (rf"\b(?:from|between|during)?\s*({names})\s*(\d{{4}})?\s*"
               rf"(?:to|through|until|till|and|-|–|—)\s*({names})\s*(\d{{4}})?\b")
    match = re.search(pattern, text or "", re.I)
    if not match:
        return None
    first, first_year, second, second_year = match.groups()
    # A day anywhere between the two month names means this is a day-level
    # range and belongs to the other parser.
    if re.search(r"\d{1,2}\s*(?:st|nd|rd|th)?\s*$", text[:match.start(1)][-6:] or ""):
        return None
    year_a = int(first_year) if first_year else (int(second_year) if second_year else None)
    year_b = int(second_year) if second_year else year_a
    if year_a is None:
        return None
    start_month, end_month = MONTHS[first.lower()], MONTHS[second.lower()]
    start = f"{year_a:04d}-{start_month:02d}-01"
    end_year, end_month_next = (year_b, end_month + 1) if end_month < 12 else (year_b + 1, 1)
    end = f"{end_year:04d}-{end_month_next:02d}-01"
    if end <= start:
        return None
    return {"start": start, "end": end,
            "label": f"{first.lower()} {year_a} to {second.lower()} {year_b}",
            "year": year_a}


def _spoken_range(text: str) -> dict | None:
    """A period given as two dates in words, as a half-open window.

    Both endpoints carry a day, so this cannot fire on "March 2016 to April
    2017", which names two months and no days and is a different question the
    month branch already answers. A year stated once applies to both ends,
    which is how the range is written: "from 1 March to 30 April 2016".
    """
    pattern = DAY_MONTH.format(months="|".join(MONTHS))
    joined = (rf"\b(?:from|between)?\s*{pattern}\s*"
              rf"(?:to|through|until|till|and|-|–|—)\s*{pattern}\b")
    match = re.search(joined, text or "", re.I)
    if not match:
        return None
    groups = match.groups()
    first, second = groups[:5], groups[5:]

    def one(parts):
        day_first, month_a, month_b, day_second, year = parts
        month = (month_a or month_b or "").lower()
        day = day_first or day_second
        if not month or not day:
            return None, None
        return (MONTHS[month], int(day)), (int(year) if year else None)

    (start_md, start_year) = one(first)
    (end_md, end_year) = one(second)
    if not start_md or not end_md:
        return None
    year = start_year or end_year
    if year is None:
        return None
    start_year = start_year or year
    end_year = end_year or year
    # "1 December to 31 January" crosses the year boundary; the end cannot
    # precede the start.
    if (end_md[0], end_md[1]) < (start_md[0], start_md[1]) and end_year == start_year:
        end_year += 1
    try:
        opens = date(start_year, *start_md)
        closes = date(end_year, *end_md) + timedelta(days=1)
    except ValueError:
        return None
    return {"start": opens.isoformat(), "end": closes.isoformat(),
            "label": f"{opens:%-d %B %Y} to {closes - timedelta(days=1):%-d %B %Y}",
            "year": opens.year,
            "explicit_dates": [opens.isoformat(), closes.isoformat()],
            "end_is_exclusive": True}


#: Phrases that name a *column* rather than a cohort. BDG2 calls the building
#: type "primaryspaceusage", so "which primary space usage category ..." is a
#: question about every building, grouped by type -- but "primary" inside it
#: reaches the subtype "Primary/Secondary Classroom", and the cohort collapsed
#: from 1,636 buildings to one. These are blanked (same length, so every span
#: offset still lines up) before any vocabulary is matched, because a value
#: found inside a schema term is an echo of the schema word, not a filter.
SCHEMA_PHRASES = (
    "primary space usage", "primaryspaceusage", "sub_primaryspaceusage",
    "primary space use", "space usage", "building type", "building types",
    "building subtype", "building subtypes", "building category",
    "building categories", "site id", "building id", "meter type",
    # "within their own primary-use category" is a phrase about the *schema*:
    # it names the column to group by. Unmasked, "primary" matched the subtype
    # "Primary/Secondary Classroom" and the cohort collapsed from 1,547
    # buildings to the classrooms, so a portfolio question was answered over
    # one subtype.
    "primary use category", "primary-use category", "primary use",
    "primary building use", "primary building-use", "use category",
    "primary category", "primary space",
)

_SCHEMA_RE = re.compile(
    "|".join(sorted((re.escape(p).replace(r"\ ", r"[\s_\-]+")
                     for p in SCHEMA_PHRASES), key=len, reverse=True)))


def _mask_schema_phrases(text: str) -> str:
    """Blank schema terms, preserving length so match offsets stay valid."""
    return _SCHEMA_RE.sub(lambda m: " " * len(m.group(0)), text)


class IntentAgent:
    """Interaction & Intent: resolve entities, period, metric, output shape.

    Never touches measurements -- it resolves *what was asked*, grounded in the
    catalog, and surfaces ambiguity rather than guessing.
    """

    def __init__(self, vocabulary: Vocabulary):
        self.vocab = vocabulary

    def resolve(self, question: str, carried: dict | None = None) -> Intent:
        text = _mask_schema_phrases(question.lower())
        entities: dict = {}

        # Which value a question names, when it names more than one, is decided
        # by where they appear in the question -- not by which the vocabulary
        # happens to yield first. The vocabularies are sets, so the old
        # first-match-and-break took whichever the set iterated to, which is
        # arbitrary and differs between processes: "compare office buildings
        # with healthcare buildings" resolved to Healthcare, and a rerun could
        # resolve it to Office.
        #
        # Every match is kept, in the order the question mentions them, because
        # a comparison needs both cohorts and only the first was surviving.
        # Longer names win an overlapping position, so a city called "York"
        # cannot claim the "New York" a question actually wrote.
        def mentioned(vocabulary, plural: bool = False) -> list[str]:
            found: list[tuple[int, int, str]] = []
            for value in vocabulary:
                if not value:
                    continue
                suffix = "s?" if plural else ""
                # The catalog spells compound names closed -- PublicServices,
                # FoodService, ReligiousWorship, TechnologyScience -- and a
                # person writes them open: "Public services in the UK" matched
                # nothing, so the cohort silently became every UK building.
                # The separator is derived from the value's own case
                # boundaries rather than listed, so a new compound type needs
                # no change here.
                parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|\d+|[a-z]+",
                                   str(value)) or [str(value)]
                body = r"[\s\-_/]*".join(re.escape(part.lower())
                                         for part in parts)
                # English pluralises the last word, and not always with an s.
                # "Dormitory" is written "dormitories" and "College
                # Laboratory" is "college laboratories"; a bare "s?" matched
                # neither, so a question about dormitories resolved no subtype
                # at all and was answered over the whole portfolio.
                if suffix and body.endswith("y") and not body.endswith(
                        ("ay", "ey", "iy", "oy", "uy")):
                    body, suffix = body[:-1], "(?:y|ies)"
                elif suffix:
                    # A type is also written as the adjective it makes:
                    # "educational buildings" is the Education type, and a
                    # bare "s?" matched neither it nor "educational". The
                    # alternation is the noun, its plural, and the -al form;
                    # anything more elaborate would start matching words that
                    # merely begin the same way.
                    suffix = "(?:s|al)?"
                match = re.search(rf"\b{body}{suffix}\b", text)
                # "Others of similar floor area" is not the building type
                # Other. A catalog value that is also an ordinary English word
                # only counts when the sentence is using it as the name of a
                # type: "compared to others" invented a second cohort of
                # Other-type buildings and mirrored the whole plan onto it.
                if match and str(value).lower() in AMBIGUOUS_TYPE_WORDS:
                    tail = text[match.end():match.end() + 14]
                    if not re.match(r"\s*(?:building|type|propert|facilit|"
                                    r"space|categor)", tail):
                        continue
                if match:
                    found.append((match.start(), -len(str(value)), str(value)))
            return [value for _, _, value in sorted(found)]

        for key, vocabulary, plural in (
                ("country", self.vocab.countries, False),
                ("city", self.vocab.cities, False),
                ("site_id", self.vocab.sites, False),
                ("building_type", self.vocab.building_types, True),
                # The finer dimensions, resolved after the canonical type so a
                # longer match cannot be shadowed by a shorter one, and only
                # from the vocabularies that were filtered for ambiguity.
                ("building_subtype", self.vocab.building_subtypes or set(), True),
                ("industry", self.vocab.industries or set(), False),
                ("subindustry", self.vocab.subindustries or set(), False)):
            names = mentioned(vocabulary, plural)
            if names:
                entities[key] = names[0]
                if len(names) > 1:
                    entities[f"{key}_all"] = names
        # Match building ids by membership in the catalog, not by shape, so a
        # second dataset's naming convention still resolves.
        named = [token for token in re.findall(r"\b[A-Za-z][\w\-]*\b", question)
                 if token in self.vocab.building_ids]
        if named:
            entities["building_ids"] = list(dict.fromkeys(named))

        # A follow-up may elide entities the previous turn established.
        for key, value in (carried or {}).items():
            entities.setdefault(key, value)

        time_range = self._time_range(text)
        limit = self._limit(text)
        if limit:
            entities["limit"] = limit

        category, tag, recognised = "3", "consumption", False
        for pattern, cat, name in PATTERNS:
            if re.search(pattern, text):
                category, tag, recognised = cat, name, True
                break

        # "Find all office buildings that have both electricity and complete
        # weather data" states requirements and names no measure: it is a
        # catalog filter, not an analysis. Falling through to the default
        # consumption tag made it scan 296 buildings' measurements to answer a
        # question the capabilities table answers directly.
        requirements = resolve_requirements(question)
        if (requirements and tag == "consumption"
                and not re.search(
                    r"\b(?:total|sum|mean|average|median|peak|max(?:imum)?|"
                    r"min(?:imum)?|baseload|profile|trend|forecast|anomal\w+|"
                    r"cluster\w*|rank\w*|correlat\w*|percentile|eui|"
                    r"intensity|consumption of|demand of)\b", text)):
            category, tag = "1", "discovery"

        ambiguities = []
        # One phrase cannot filter on two dimensions at once. This catalog
        # holds a *type* called PublicServices and a *subtype* called "Public
        # services", so "Public services in the UK" resolved to both and the
        # plan required a building to be both -- which matches nothing, and
        # would have answered a question about eight buildings with silence.
        # The type wins: it is the coarser reading and a superset of the
        # subtype, so keeping it can only be too generous, never empty.
        def _flat(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(value).lower())

        def _span(value: str):
            """Where this catalog value appears in the question, if it does."""
            parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|\d+|[a-z]+",
                               str(value)) or [str(value)]
            body = r"[\s\-_/]*".join(re.escape(p.lower()) for p in parts)
            found = re.search(rf"\b{body}s?\b", text)
            return (found.start(), found.end()) if found else None

        # One side of a compound subtype, where nothing fuller matched. This
        # is what lets "dormitories" reach "Residence Hall/Dormitory" without
        # anyone having to know how BDG2 spells it. Every label the side
        # reaches is kept, because the side is genuinely ambiguous between
        # them and dropping either would answer a narrower question than was
        # asked.
        if not entities.get("building_subtype"):
            for word, labels in (self.vocab.subtype_sides or {}).items():
                if re.search(rf"\b{re.escape(word)}s?\b", text):
                    ordered = sorted(labels)
                    entities["building_subtype"] = ordered[0]
                    if len(ordered) > 1:
                        entities["building_subtype_all"] = ordered
                    entities["building_subtype_matched_on"] = word
                    break

        # The four type dimensions overlap in their vocabularies, and one
        # phrase can match several: "entertainment/public assembly" is a
        # subtype, and "public assembly" *inside* it is a subindustry, and
        # "assembly" inside that is a canonical type. All three resolved, the
        # plan required a building to be all three at once, and the cohort
        # came out empty where the question means nine buildings.
        #
        # The rule is span containment, not a table of exceptions: where one
        # dimension's match sits inside another's, the longer match is the
        # fuller reading of those words and the shorter is an echo of it.
        # Dimensions matching *different* words are left alone -- "offices and
        # schools" really does name two things, and that is handled below.
        dimensions = [(key, entities.get(key)) for key in
                      ("building_type", "building_subtype", "industry",
                       "subindustry") if entities.get(key)]
        spans = {key: _span(value) for key, value in dimensions}
        for key, value in dimensions:
            mine = spans.get(key)
            if not mine:
                continue
            swallowed = any(
                other != key and spans.get(other)
                and spans[other][0] <= mine[0] and spans[other][1] >= mine[1]
                and (spans[other][1] - spans[other][0]) > (mine[1] - mine[0])
                for other, _ in dimensions)
            if swallowed:
                entities.pop(key, None)
                entities.pop(f"{key}_all", None)

        # What remains may still be two dimensions matching different words --
        # a type and a subtype naming separate cohorts. A building cannot be
        # both, so requiring both matches nothing; the type is kept because a
        # comparison route can act on it, and the dropped half is said aloud.
        if entities.get("building_type") and entities.get("building_subtype"):
            entities.pop("building_subtype", None)
            entities.pop("building_subtype_all", None)
            ambiguities.append(
                "the question names both a building type and a subtype; a "
                "building cannot be both, so the subtype was dropped and only "
                "the type is filtered on")

        # A question that names a building *and* a site should agree with
        # itself. "Moose_education_Lori at the Peacock site" does not --
        # that building is at Moose -- and the building id wins, because it
        # names one thing where a site names many. The contradiction is
        # recorded rather than resolved without comment: whoever wrote the
        # request needs to know which half was used.
        stated_site = entities.get("site_id")
        named_buildings = entities.get("building_ids") or []
        mapping = self.vocab.site_of_building or {}
        if stated_site and named_buildings and mapping:
            elsewhere = {b: mapping[b] for b in named_buildings
                         if b in mapping and mapping[b] != stated_site}
            if elsewhere and len(elsewhere) == len(named_buildings):
                listed = ", ".join(f"{b} is at site {s}"
                                   for b, s in list(elsewhere.items())[:3])
                ambiguities.append(
                    f"the question names site {stated_site}, but {listed}; "
                    f"the building id is used and the site is ignored")
                entities.pop("site_id", None)
                entities["site_contradicted"] = stated_site

        if not entities.get("building_ids") and not any(
                k in entities for k in ("country", "city", "site_id", "building_type")):
            ambiguities.append("no building, site, city, country or usage type named; "
                               "the whole portfolio will be considered")
        if tag in {"anomaly", "benchmark"} and not time_range:
            ambiguities.append("no period stated; the full snapshot window is used")

        # "How many warehouses does the dataset contain?" resolved subject=datasets,
        # because "dataset" is a noun in the sentence and "warehouses" was not
        # recognised as buildings -- so it answered 1.
        region = resolve_region(question, set(self.vocab.countries))
        if region:
            entities["country_all"] = region
            entities.setdefault("country", region[0])

        subject = resolve_subject(question)
        if subject in (None, "datasets") and entities.get("building_type"):
            subject = "buildings"

        return Intent(question=question, task_category=category, entities=entities,
                      time_range=time_range, metric=tag,
                      output_format=self._output_format(text),
                      subject=subject,
                      out_of_scope=resolve_scope(question),
                      requirements=requirements,
                      group_by=resolve_group_by(question),
                      recognised_shape=recognised,
                      ambiguities=ambiguities)

    @staticmethod
    def _limit(text: str) -> int | None:
        """How many results were asked for -- digits or words."""
        words = "|".join(WORD_NUMBERS)
        patterns = [
            # "Top 10%" is a decile of the portfolio and "top 10" is ten
            # buildings. Reading the first as the second turned a comparison of
            # 163 buildings against 820 into a ten-row ranking.
            rf"\b(?:top|bottom|first|best|worst)\s+(\d+|{words})\b(?!\s*%)",
            rf"\bwhich\s+(\d+|{words})\b",
            rf"\b(\d+|{words})\s+(?:buildings?|offices?|sites?|meters?|cities)\b",
        ]
        for pattern in patterns:
            found = re.search(pattern, text)
            if found:
                token = found.group(1)
                return int(token) if token.isdigit() else WORD_NUMBERS.get(token)
        return None

    @staticmethod
    def _time_range(text: str) -> dict:
        # An explicit date is the most specific thing a question can say, so it
        # wins over a bare year. "after 2017-07-31" must not become 2017-01-01.
        iso = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso:
            start = iso[0]
            end = iso[1] if len(iso) > 1 else None
            return {"start": start, "end": end, "label": " to ".join(iso),
                    "year": int(start[:4]), "explicit_dates": iso}
        # A range with two spoken endpoints -- "from 1 March 2016 to 30 April
        # 2016", "between March 1 and April 30, 2016". The month branch below
        # reads the *first* month-and-day it finds and treats it as one day,
        # so a two-month evaluation window resolved to 1-2 March: a question
        # about 1,464 hours was answered with 24. The end is inclusive as
        # spoken and exclusive as stored, so the day named is added on.
        span = _spoken_range(text)
        if span:
            return span
        # Two months and no days -- checked before the single-month branch,
        # which would otherwise take the first month and drop the rest.
        span = _spoken_month_range(text)
        if span:
            return span
        years = sorted({int(y) for y in re.findall(r"\b(201[5-9]|202\d)\b", text)})
        # A calendar quarter is a period like any other, and nothing resolved
        # it: "during Q4 2017" fell through to the bare-year branch and ran
        # over all twelve months, so a question about three months was answered
        # from four times the days it named.
        quarter = re.search(r"\bq([1-4])\b(?:\s*(?:of\s*)?(20\d\d))?"
                            r"|\b(first|second|third|fourth)\s+quarter\b", text)
        if quarter and (years or quarter.group(2)):
            number = (int(quarter.group(1)) if quarter.group(1)
                      else ["first", "second", "third", "fourth"].index(
                          quarter.group(3)) + 1)
            year = int(quarter.group(2)) if quarter.group(2) else years[0]
            first = 3 * (number - 1) + 1
            end_year, end_month = ((year + 1, 1) if first + 3 > 12
                                   else (year, first + 3))
            return {"start": f"{year}-{first:02d}-01",
                    "end": f"{end_year}-{end_month:02d}-01",
                    "label": f"Q{number} {year}", "year": year,
                    "months": [first, first + 1, first + 2]}
        month = next((m for m in MONTHS if re.search(rf"\b{m}\b", text)), None)
        season = next((s for s in SEASONS if re.search(rf"\b{s}\b", text)), None)
        # "Annual" names a period even though it names no year: annual EUI and
        # annual consumption are a year's worth, and treating them as periodless
        # ran them over the whole record -- two years of it -- while every
        # reference for such a question uses one. The latest complete calendar
        # year is the one meant by default, and the answer says which.
        annual = bool(re.search(r"\bannual(?:ly)?\b|\bper\s+year\b"
                                r"|\bin\s+a\s+year\b", text))
        if not years and not month and not season:
            if annual:
                return {"start": f"{LATEST_FULL_YEAR}-01-01",
                        "end": f"{LATEST_FULL_YEAR + 1}-01-01",
                        "label": f"{LATEST_FULL_YEAR}", "year": LATEST_FULL_YEAR,
                        "assumed": "annual with no year named; the latest "
                                   "complete calendar year is used"}
            return {}
        year = years[0] if years else 2017
        if month:
            index = MONTHS[month]
            # A month name with a day beside it is a *day*: "January 1, 2017",
            # "1 January 2017", "March 3rd". Reading only the month turned
            # "which building recorded the minimum demand on January 1" into a
            # question about the whole of January, and the plan that followed
            # was faithful to an intent that had already lost the day.
            day = re.search(rf"\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text) \
                or re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{month}\b", text)
            if day and 1 <= int(day.group(1)) <= 31:
                number = int(day.group(1))
                start = date(year, index, number)
                return {"start": start.isoformat(),
                        "end": (start + timedelta(days=1)).isoformat(),
                        "label": f"{month} {number} {year}", "year": year,
                        "explicit_dates": [start.isoformat()]}
            if not years:
                # A bare month is every one of them the record holds, for the
                # same reason a bare season is: "the heatwave in July" names
                # no year, and answering over July 2017 alone drops half the
                # hot days the snapshot has. The season branch below already
                # settled this; a month is the same question asked smaller.
                return {"months": [index], "label": month,
                        "spans_all_years": True}
            end_year, end_month = (year + 1, 1) if index == 12 else (year, index + 1)
            return {"start": f"{year}-{index:02d}-01",
                    "end": f"{end_year}-{end_month:02d}-01",
                    "label": f"{month} {year}", "year": year}
        if season:
            lo, hi = SEASONS[season]
            months = ([12, 1, 2] if lo > hi
                      else list(range(lo, hi + 1)))
            if not years:
                # "Over the summer" with no year is every summer the record
                # holds, not the last one. Defaulting to 2017 halved the days
                # behind the answer and nothing said so: the daily load factor
                # for site Peacock came out 0.8076 against 0.8063 for the
                # season the question actually names.
                return {"months": months, "label": season,
                        "spans_all_years": True}
            if lo > hi:  # winter spans the year boundary
                return {"start": f"{year}-12-01", "end": f"{year + 1}-03-01",
                        "label": f"winter {year}", "year": year,
                        "months": months}
            return {"start": f"{year}-{lo:02d}-01", "end": f"{year}-{hi + 1:02d}-01",
                    "label": f"{season} {year}", "year": year, "months": months}
        if len(years) > 1:
            return {"start": f"{years[0]}-01-01", "end": f"{years[-1] + 1}-01-01",
                    "label": f"{years[0]}-{years[-1]}", "years": years}
        return {"start": f"{year}-01-01", "end": f"{year + 1}-01-01",
                "label": str(year), "year": year}

    @staticmethod
    def _output_format(text: str) -> str:
        if re.search(r"\b(plot|chart|graph|visuali[sz]|figure)\b", text):
            return "figure"
        # A pattern, a profile, a curve or a shape over a period is a figure
        # however the question words it, "compare" included: two curves are
        # not answered by one mean.
        if re.search(r"\b(?:show|display|give|describe|compar\w*|contrast)\b"
                     r"[\w\s,'-]{0,40}"
                     r"\b(patterns?|profiles?|curves?|shapes?|signatures?)\b"
                     r"|\b(?:daily|hourly|intraday|24-hour|24\s*hour|weekly|"
                     r"seasonal|weekday|weekend)\b[\w\s,'-]{0,20}"
                     r"\b(patterns?|profiles?|curves?|shapes?)\b", text):
            return "figure"
        if re.search(r"\b(map|geograph)\b", text):
            return "map"
        # A scalar is right only when one entity is asked about one measure.
        # These three rules, checked first, are what separates "what is this
        # building's EUI" from "what is the mean and median for all healthcare
        # buildings" -- both of which used to resolve to a single number
        #.
        #
        # 1. an explicit grouping phrase
        if re.search(r"\b(?:per|by|for each|each|across)\s+"
                     r"(?:site|city|cities|country|countries|building[ _]type"
                     r"|dataset|month|year|hour|day|season)s?\b", text):
            return "table"
        # 2. a grouping dimension mentioned but not pinned to one value:
        #    "how many education buildings are available at <...> site"
        if (re.search(r"\bhow many\b", text)
                and re.search(r"\b(?:at|in|across|for)\b[^.?]*?\b"
                              r"(?:site|city|country|building type)s?\b", text)
                # A named site pins the dimension: "at site Bear" asks about
                # Bear, not one row per site. This tested for a capital letter
                # in text that had already been lowercased, so it never fired
                # and every such question became a grouped table.
                and not re.search(r"\b(?:site|city|country)\s+\w", text)):
            return "table"
        # A counting verb, with no grouping phrase above, asks for one number
        # however wide the cohort: "count all buildings in London" is 68, not a
        # listing of them. The grouping rules run first, so "count buildings per
        # site" still returns one row per site.
        if re.match(r"^\s*(?:count|tally|how many)\b", text):
            return "number"
        # "What years does this snapshot cover" asks for the set of periods that
        # hold data -- not one number, and not a table of statistics about them.
        # The counting rule above still takes "how many years", and the coverage
        # verb keeps "which months had the highest demand" out: that is a
        # ranking, not an inventory.
        if (re.search(r"\b(?:what|which)\s+(?:calendar\s+|time\s+)?"
                      r"(?:years?|months?|periods?)\b", text)
                and re.search(r"\b(?:covers?|covered|coverage|available|"
                              r"spans?|spanned|present|exists?|"
                              r"(?:have|has|with)\s+data)\b", text)):
            return "list"

        # "Which building recorded the minimum demand" asks for one building by
        # name. The answer is that name -- not a table with it inside, and not
        # the number of rows that table has, which is what three such questions
        # actually returned: "1". This runs before the cohort rule below,
        # because that rule matches the singular noun too and calls it a table.
        # Singular only: "which buildings" is a membership question.
        # The words allowed between "which" and the noun are adjectives, not a
        # prepositional phrase: "which hours of the day" would otherwise skip
        # the plural it asks about and match the singular "day" three words
        # later, turning a 24-row profile question into a single hour.
        if re.search(r"\b(?:(?:which|what)\s+(?:single\s+)?|find\s+the\s+"
                     r"(?:single\s+)?|the\s+single\s+)"
                     r"(?:(?!of\b|in\b|during\b|across\b)\w+\s+){0,3}?"
                     r"(?:building|site|city|country|region|month|category|"
                     r"location|meter|stream|day|hour)\b(?!s\b)", text):
            return "entity"

        # 3. more than one statistic, or a cohort rather than an entity
        if re.search(r"\b(?:mean|median|average|min(?:imum)?|max(?:imum)?|"
                     r"p\d{1,2}|percentile|std(?:dev)?|variance|total|sum)\b"
                     r"\s*(?:,|and|&|/)\s*"
                     r"\b(?:mean|median|average|min(?:imum)?|max(?:imum)?|"
                     r"p\d{1,2}|percentile|std(?:dev)?|variance|total|sum)\b",
                     text):
            return "table"
        # "What percentile does X fall into among all office buildings" asks for
        # one number about X. The cohort after "among" is what X is compared
        # *to*, not what the answer is about.
        comparison = re.sub(
            r"\b(?:among|compared\s+to|relative\s+to|versus|vs\.?|against|"
            r"within)\b.*$", "", text)
        if re.search(r"\b(?:all|every|each)\s+\w*\s*"
                     r"(?:buildings?|facilit(?:y|ies)|sites?|premises|homes?|"
                     r"properties|meters?)\b"
                     r"|\bwhich\s+\w*\s*(?:buildings?|sites?|cit(?:y|ies)"
                     r"|countr(?:y|ies)|locations?|portfolios?|regions?"
                     r"|days?|dates?|hours?|weeks?|months?|periods?)\b"
                     r"|\bgroup\s+the\b|\bclassify\b|\bsegment\b", comparison):
            return "table"
        # A yes/no question wants a verdict with its evidence, not a table the
        # reader must interpret. Without this, "does this meter contain any
        # duplicated timestamps" returns a data-availability table and leaves the
        # yes/no to the reader.
        if re.match(r"^\s*(?:does|do|is|are|was|were|did|can|has|have|will)\b",
                    text) and not re.search(r"\bhow\b|\bwhich\b|\bwhat\b", text):
            return "boolean"
        # "its top 5% highest-demand hours" names a threshold, not a ranking to
        # return: the question asks for one number *about* those hours. Reading
        # it as a ranking request returned a table.
        # A schedule, pattern or breakdown has rows: "what is the apparent
        # operating schedule" cannot be one number.
        if re.search(r"\b(?:operating\s+)?schedule\b|\bbreakdown\b"
                     r"|\bload\s+shape\b|\bhourly\s+pattern\b", text):
            return "table"
        if re.search(r"\b(rank|top \d+(?!\s*%)|list|which buildings)\b", text):
            return "ranking"
        # A question about one measure is a number question however it is
        # phrased, possessives included.
        if re.search(
            # "What floor area does X have", "what EUI does X show" -- the
            # measure sits between "what" and the verb, so a pattern anchored on
            # "what is" misses it entirely.
            r"\bwhat\s+[\w\s-]{2,28}\s+(?:does|do|did|has|have|had)\b"
            r"|\b(how many|how much|how long|what (?:is|was|are|were)\b"
            r"|what percentage|what share|what fraction|what proportion"
            r"|at what (?:\w+\s+){0,2}(?:hour|time|point)|what time"
            r"|total|average|mean|median|number of"
            # "count of", "count the", "count all", "count distinct"
            r"|\bcount\s+(?:of|the|all|distinct)\b"
            r"|percentile|ratio|correlation)\b", text):
            return "number"
        return "table"


# Weather variables a question can ask about by name, mapped to the column.
WEATHER_VARIABLES = (
    ("air_temperature", r"\bair[\s_]?temperature|outdoor[\s_]?temperature|\btemperature\b"),
    ("dew_temperature", r"\bdew[\s_]?(?:point|temperature)\b"),
    ("wind_speed", r"\bwind[\s_]?speed\b"),
    ("wind_direction", r"\bwind[\s_]?direction\b"),
    ("cloud_coverage", r"\bcloud[\s_]?cover\w*\b"),
    ("precip_depth_1hr", r"\bprecip\w*\b|\brain\w*\b"),
    ("sea_level_pressure", r"\bpressure\b"),
)

PRIORITY_METHODS = ("normalised:eui", "baseload:night_mean:baseload_ratio")


# Which measure a geographic or performance question is about. The planner used
# to name "eui" wherever a metric was required, so "which city has the highest
# aggregate total consumption" was answered with median intensity per square
# metre -- a different quantity, ranked differently, presented as the answer.
GEOGRAPHIC_METRICS = (
    ("total_kwh", r"\btotal\b|\baggregate\b|\bconsumption\b|\bkwh\b|\busage\b|\bused\b"),
    ("peak_demand", r"\bpeak\b|\bmaximum demand\b|\bmax demand\b"),
    ("mean_demand", r"\b(?:mean|average|typical)\s+demand\b"),
    ("eui", r"\beui\b|\bintensity\b|\bper\s+(?:square|sq|m2|sqm)\b|\bnormali[sz]"),
)

INTENSITY_WORDS = r"\beui\b|\bintensity\b|\bper\s+(?:square|sq|m2|sqm)\b|kwh/m"


# A profile question is about shape only when it asks whether the rhythm
# moved. "Does the daily profile change between winter and summer" does;
# "compare Monday's profile to Sunday's" asks which day draws more, and its
# answer is the ratio of the two means. One definition, used by the rule-based
# route and by the repair that holds a model plan to the same reading.
SHAPE_SUBJECT = r"\b(profile|shape|curve|pattern|rhythm)\b"
SHAPE_CHANGE = (r"\bchang\w*|\bdiffer\w*|\bsame\b|\bshift\w*|\balike\b"
                r"|\bsimilar\w*|\bshape\b")


#: Phrases that make a question about the *record* rather than the readings:
#: how much is missing, how long the gaps are, how complete the coverage is.
#: The distinction matters because the model reliably plans the right
#: data-quality tool first and then appends a consumption tool whose number it
#: answers with -- "what percentage of timestamps are missing" came back as
#: 1.343e+06 kWh.
COMPLETENESS_RE = re.compile(
    r"\b(?:missing|gaps?|incomplete|completeness|coverage|"
    r"duplicate[sd]?|absent|no\s+data|without\s+data)\b", re.I)

#: Tools that answer a question about the record.
COMPLETENESS_TOOLS = ("get_data_availability", "profile_data_quality",
                      "detect_data_issues", "impute_missing_values",
                      "profile_metadata_completeness")


#: "the 95th percentile", "p95", "95th pct" -- the quantile a question names.
PERCENTILE_RE = re.compile(
    # "95th percentile", "95 pct", "95 quantile" -- the number then the word.
    r"\b(\d{1,2}(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*"
    r"(?:percentile|pct\b|quantile)"
    # or the bare "p95" / "P99" form, which a question uses on its own.
    r"|\bp(\d{1,2})\b", re.I)


def resolve_percentiles(text: str) -> list[float]:
    """Percentiles a question asks for, as percents. Empty when it asks none.

    The tools report a peak, a mean and a minimum, so without this a question
    after the 95th percentile is answered with the maximum -- a number of the
    right shape and the wrong meaning.
    """
    out: list[float] = []
    for match in PERCENTILE_RE.finditer(text or ""):
        try:
            share = float(match.group(1) or match.group(2))
        except (TypeError, ValueError):
            continue
        if 0.0 < share < 100.0:
            out.append(share)
    return sorted(set(out))


#: "after controlling for hour-of-day and day-of-week", "adjusted for time of
#: day", "holding the working day constant" -- the phrase that turns a
#: correlation into a partial one.
CONTROLS_RE = re.compile(
    r"\b(?:control(?:l(?:ing|ed))?\s+for|adjust(?:ing|ed)?\s+for|"
    r"account(?:ing|ed)?\s+for|holding)\b", re.I)

CONTROL_TERMS = (
    ("hour_of_day", re.compile(r"hour[-\s]of[-\s]day|time[-\s]of[-\s]day|hourly\s+pattern", re.I)),
    ("day_of_week", re.compile(r"day[-\s]of[-\s]week|weekday|day\s+type", re.I)),
    ("month", re.compile(r"\bmonth(?:ly)?\b|season", re.I)),
)


def resolve_weather_controls(text: str) -> list[str]:
    """Time terms a question asks to hold constant. Empty when it asks none.

    Without these the weather tool fits a plain univariate slope, which on
    hourly data is mostly measuring the working day: consumption and
    temperature both rise in the morning, so a building with no cooling still
    correlates. "After controlling for hour-of-day and day-of-week" is asking
    for a different number, not a caveat on the same one.
    """
    if not CONTROLS_RE.search(text or ""):
        return []
    return [name for name, pattern in CONTROL_TERMS if pattern.search(text or "")]


#: Questions about how consumption moves with the weather. The tools that
#: answer them are the weather ones; a plan that also ranks or fetches is not
#: wrong, but the ranking is not the answer.
WEATHER_RELATION_RE = re.compile(
    r"\b(?:correlat\w*|sensitiv\w*|associat\w*|relationship|respond\w*|"
    r"depend\w*)\b", re.I)
WEATHER_SUBJECT_RE = re.compile(
    r"\b(?:temperature|weather|outdoor|air\s+temp\w*|degree[-\s]days?|"
    r"dew|humidity|wind|cloud)\b", re.I)

WEATHER_TOOLS = ("analyze_weather_relationship", "fit_temperature_response",
                 "weather_normalize_consumption")

#: An eligibility question is answered by the step that resolves the set. A
#: measurement over that set, planned afterwards, re-screens it on the floor
#: its own metric needs and reports a smaller number: the eligible cohort came
#: to 1,468 and the EUI step that followed answered 1,335.
ELIGIBILITY_TOOLS = ("find_buildings", "check_task_eligibility")

#: A conjunction is answered by the step that intersects the criteria. Each
#: criterion's own step is a component of the answer, never the answer: asked
#: which offices clear both the EUI and the nighttime-baseload 90th
#: percentile, the reply came from the second criterion alone -- 28 buildings
#: where 25 clear both.
CONJUNCTION_TOOLS = ("combine_evidence",)


#: Questions that count or characterise a *population* rather than judging
#: individual buildings, and questions about the effect of data completeness.
#: Both need every building the metric can be computed for, not only the ones
#: whose record the snapshot already judged long enough.
POPULATION_RE = re.compile(
    r"\b(?:how\s+many|which\s+buildings|all\s+buildings|eligible|"
    r"across\s+all|of\s+\d+\s+buildings|every\s+building)\b", re.I)


#: "at least 12 consecutive months", "6 continuous months of data",
#: "365 consecutive days", "8760 consecutive hourly readings".
UNBROKEN_RE = re.compile(
    r"\b(?:at\s+least\s+)?(\d+(?:\.\d+)?)\s*"
    r"(consecutive|continuous|unbroken|uninterrupted)?\s*"
    r"(month|day|hour|week)s?\b(?:[^.]{0,40}?\b(consecutive|continuous|"
    r"unbroken|uninterrupted)\b)?", re.I)

_UNBROKEN_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 730}


#: "How many times did daily consumption increase by more than 20% compared
#: with the previous day" counts ordinary period-over-period moves. Nothing in
#: the anomaly vocabulary answers it: detect_change_points finds one persistent
#: regime shift per building and first_difference scores a move in standard
#: deviations of that building's own changes, so the same 20% is 0.8 sigma in
#: one building and 4 in another. The count asked for is on the stated scale.
STEP_CHANGE_RE = re.compile(
    r"\b(?:how\s+many\s+times|how\s+often|number\s+of\s+(?:times|days|"
    r"occasions))\b[^.?]{0,120}?"
    r"\b(?:increas\w+|ros\w+|rise|rising|jump\w*|decreas\w+|fell|fall\w*|"
    r"drop\w*|chang\w+)\b[^.?]{0,80}?"
    r"\b(?:compared\s+with|compared\s+to|versus|vs\.?|than|relative\s+to|"
    r"over|from)\s+(?:the\s+)?(?:previous|preceding|prior|day\s+before)\b",
    re.I)

#: Which way the move has to go for it to count.
STEP_DIRECTION = (
    (r"\bincreas\w+|\bros\w+|\brise|\brising|\bjump\w*|\bhigher\b|\bup\b",
     "increase"),
    (r"\bdecreas\w+|\bfell\b|\bfall\w*|\bdrop\w*|\blower\b|\bdown\b",
     "decrease"),
)


def resolve_step_change(text: str) -> dict | None:
    """A count of period-over-period moves past a stated percentage, or None.

    The percentage and the grain both come from the question. A question that
    names no percentage is not this one -- "how many times did consumption
    increase over the previous day" counts every up day, which is roughly half
    of them -- so the threshold is required rather than defaulted.
    """
    lowered = text or ""
    if not STEP_CHANGE_RE.search(lowered):
        return None
    size = re.search(r"\b(?:more|greater|larger)\s+than\s+(\d+(?:\.\d+)?)\s*%"
                     r"|\b(?:by|above|over)\s+(?:more\s+than\s+)?"
                     r"(\d+(?:\.\d+)?)\s*%", lowered, re.I)
    if not size:
        return None
    percent = float(next(g for g in size.groups() if g))
    direction = next((name for pattern, name in STEP_DIRECTION
                      if re.search(pattern, lowered, re.I)), "either")
    grain = ("weekly" if re.search(r"\bweek(?:ly)?\b", lowered, re.I)
             and not re.search(r"\bdai?ly\b|\bday\b", lowered, re.I)
             else "monthly" if re.search(r"\bmonth(?:ly)?\b", lowered, re.I)
             and not re.search(r"\bdai?ly\b|\bday\b", lowered, re.I)
             else "daily")
    return {"change_at_least_pct": percent, "direction": direction,
            "grain": grain}


#: "Gaps longer than seven consecutive days", "a gap exceeding a week". A run
#: of *missing* data, which is the opposite selection from a run of present
#: data and goes to a different catalog parameter: gap_longer_than_hours
#: returns the broken streams, gap_at_most_hours the sound ones. Reading one
#: as the other answers "which buildings are broken" with the list of the
#: buildings that are fine.
_GAP_COUNT = r"(?:(\d+(?:\.\d+)?|a|an|one|" + "|".join(WORD_NUMBERS) + r")\s+)?"
GAP_LENGTH_RE = re.compile(
    r"\bgaps?\b[^.?]{0,40}?\b(?:longer\s+than|exceeding|over|more\s+than|"
    r"greater\s+than|of\s+more\s+than|above)\s+"
    + _GAP_COUNT + r"(?:consecutive\s+)?(hour|day|week|month)s?\b"
    r"|\b(?:longer\s+than|exceeding|over|more\s+than)\s+"
    + _GAP_COUNT + r"(?:consecutive\s+)?(hour|day|week|month)s?\b"
    r"[^.?]{0,40}?\bgaps?\b", re.I)


def resolve_gap_hours(text: str) -> int | None:
    """The gap length a question calls too long, in hours, or None.

    "A week" with no number is one week, so the count is optional; a bare
    number with no unit is not a gap length and returns nothing rather than a
    guess at the unit.
    """
    match = GAP_LENGTH_RE.search(text or "")
    if not match:
        return None
    groups = [g.lower() for g in match.groups() if g is not None]
    unit = next((g for g in groups if g in _UNBROKEN_HOURS), None)
    if unit is None:
        return None
    spoken = next((g for g in groups if g != unit), "one")
    count = (float(spoken) if spoken[0].isdigit()
             else float(WORD_NUMBERS.get(spoken, 1)))
    return int(round(count * _UNBROKEN_HOURS[unit]))


def resolve_unbroken_hours(text: str) -> int | None:
    """An unbroken run the question requires, in hours. None when it asks none.

    "Twelve consecutive months with no gap" is a run, and coverage cannot
    stand in for it: a building can be 99% complete across two years without
    a single unbroken year in them, and one can sit at 60% with a clean twelve
    months inside. Asked for the run, coverage answered 628 where 285 qualify.
    """
    lowered = text or ""
    # "72 consecutive hours of zero consumption" is a run of a *value*, not a
    # run of *data*: the question is about the meter reading zero, and reading
    # it as a continuity requirement sent the cohort through a data-gap filter
    # and answered with total kWh. A continuous run of readings and a
    # continuous run of zeros are opposite findings.
    if re.search(r"\b(?:zero|flat|flatline|constant|unchanging|no\s+consumption|"
                 r"idle|off)\b", lowered, re.I):
        return None
    # A window with a stated coverage floor is not an unbroken run either.
    # "12 consecutive months of hourly data with at least 95% valid readings"
    # says in the same breath that 5% may be missing; read as a run it asks
    # for 8,760 readings in a row, which is a far stricter cohort -- and on
    # this snapshot, combined with the window filter, an empty one.
    if re.search(r"\d{1,3}\s*%\s*(?:of\s+)?(?:valid|complete|coverage|present|"
                 r"non-?missing|hourly|readings?|data)", lowered, re.I):
        return None
    for match in UNBROKEN_RE.finditer(lowered):
        if not (match.group(2) or match.group(4)):
            continue                      # a plain duration, not an unbroken one
        try:
            count = float(match.group(1))
        except (TypeError, ValueError):
            continue
        unit = (match.group(3) or "month").lower()
        hours = int(round(count * _UNBROKEN_HOURS.get(unit, 730)))
        # 12 months of hourly readings is 8,760, the figure the question means
        # by "a year", rather than 8,766 from an average month.
        if unit == "month" and count == 12:
            hours = 8760
        return hours
    return None


#: The dimension a "which X has the most Y" question groups by, and the
#: catalogue entity that counts over it.
GROUPED_COUNT_ENTITY = {
    "site": "sites", "sites": "sites",
    "city": "cities", "cities": "cities",
    "country": "countries", "countries": "countries",
    "dataset": "datasets", "datasets": "datasets",
    "building type": "building_types", "building types": "building_types",
    "use type": "building_types", "category": "building_types",
    "industry": "industries", "industries": "industries",
}

#: "which site has the largest number of education buildings", "which city has
#: the most offices", "how many buildings in each country".
GROUPED_COUNT_RE = re.compile(
    r"\bwhich\s+(?P<dim>site|city|country|dataset|building\s+type|use\s+type|"
    r"industry)\b[^.?]*?\b(?:has|have|holds?|contains?)\b[^.?]*?"
    r"\b(?:most|largest|greatest|highest|maximum|fewest|smallest)\b"
    r"[^.?]*?\b(?:number|count|buildings?)\b", re.I)


def resolve_grouped_count(text: str) -> str | None:
    """The catalogue entity a "which X has the most Y" question counts over.

    Answered with find_buildings the reply is the cohort itself -- 617 rows of
    education buildings -- which contains the answer and does not state it.
    The question asks which *site* has the most, so the count has to be
    grouped by site and ranked, which is what the catalogue does.
    """
    lowered = text or ""
    # "Which city has the highest median office EUI" also says "highest" and
    # "building", and is a question about a metric, not a headcount. A metric
    # word anywhere settles it: counting cannot answer it.
    if re.search(r"\b(?:eui|intensity|consumption|demand|load\s*factor|"
                 r"baseload|kwh|median|mean|average|correlation|peak)\b",
                 lowered, re.I):
        return None
    # "Which site pairs are most similar" counts nothing: it compares sites
    # with each other. Answering it with a headcount per site replaced a
    # similarity ranking with a league table of building counts.
    if re.search(r"\bpairs?\b|\bsimilar\w*|\bcompare[ds]?\b|\bdiffer\w*|"
                 r"\bprofiles?\b|\bcorrelat\w*", lowered, re.I):
        return None
    match = GROUPED_COUNT_RE.search(lowered)
    if not match:
        return None
    dimension = re.sub(r"\s+", " ", match.group("dim").strip().lower())
    return GROUPED_COUNT_ENTITY.get(dimension)


#: "more than 72 consecutive hours of zero electricity consumption",
#: "buildings dark for three days", "flatlined for a week".
ZERO_RUN_RE = re.compile(
    # "zero-consumption periods" is the same criterion as "zero consumption",
    # and the hyphen is how a question writes it when the phrase modifies a
    # noun. Requiring the space meant the clause was not recognised at all.
    r"\b(?:zero|no|nil)[-\s]+(?:electricity[-\s]+|energy[-\s]+)?consumption\b"
    r"|\bzero[-\s]+readings?\b"
    r"|\bflat[-\s]?lin\w*\b|\bdark\s+for\b|\bconsumption\s+of\s+zero\b",
    re.I)


#: "above the 90th percentile of EUI within their own primary-use category",
#: "in the top 10% of intensity among offices at the same site".
WITHIN_GROUP_RE = re.compile(
    r"\bwithin\s+(?:their|its)\s+own\s+(?P<grp>[a-z -]{3,30}?)\b"
    r"|\bamong\s+(?:their|its)\s+(?P<grp2>[a-z -]{3,30}?)\s+peers\b"
    r"|\bof\s+(?:the\s+)?same\s+(?P<grp3>site|city|country|category|type)\b",
    re.I)

#: Words a question uses for a grouping, mapped to the peer dimension.
PEER_WORDS = {
    "use": "building_type", "category": "building_type",
    "use category": "building_type", "type": "building_type",
    "building type": "building_type", "class": "building_type",
    "site": "site", "city": "city", "country": "country",
    "subtype": "building_subtype",
}

PERCENTILE_TAIL_RE = re.compile(
    r"\b(?:above|top|highest)\b[^.?]{0,30}?(\d{1,2})(?:st|nd|rd|th)?\s*"
    r"(?:percentile|%)|\btop\s+(\d{1,2})\s*%", re.I)


def resolve_within_group_percentile(text: str) -> dict | None:
    """A "top N% within its own group" test: the group, the tail and the width.

    Run against the whole cohort it is a different question: on this snapshot
    the portfolio-wide top tenth and the per-category top tenth share barely
    half their members, because the categories differ in level far more than
    the buildings within them do.
    """
    lowered = text or ""
    grouped = WITHIN_GROUP_RE.search(lowered)
    tail = PERCENTILE_TAIL_RE.search(lowered)
    if not grouped or not tail:
        return None
    # The phrase names the grouping a few words after "within their own", and
    # a non-greedy capture stops at the first boundary -- "primary" out of
    # "primary-use category". Read the run of words instead.
    phrase = re.sub(r"[-_]+", " ", lowered[grouped.start():grouped.start() + 60]).lower()
    peers = next((v for k, v in sorted(PEER_WORDS.items(), key=lambda kv: -len(kv[0]))
                  if k in phrase), None)
    if not peers:
        return None
    pct = next((g for g in tail.groups() if g), None)
    if pct is None:
        return None
    above = int(pct)
    # "above the 90th percentile" and "the top 10%" name the same tail.
    width = 100 - above if above >= 50 else above
    return {"peers": peers, "threshold": float(width), "side": "high"}


#: "how much does consumption differ between cool and warm hours", "compare
#: use on hot days against cold ones".
TEMPERATURE_SPLIT_RE = re.compile(
    r"\b(?:cool|cold|cooler|mild)\b[^.?]{0,40}?\b(?:vs\.?|versus|and|against|"
    r"compared\s+(?:to|with))\b[^.?]{0,40}?\b(?:warm|hot|warmer)\b"
    r"|\b(?:warm|hot|warmer)\b[^.?]{0,40}?\b(?:vs\.?|versus|and|against|"
    r"compared\s+(?:to|with))\b[^.?]{0,40}?\b(?:cool|cold|cooler|mild)\b",
    re.I)


#: "at least one complete calendar year", "a full year of data", "a complete
#: year of electricity and overlapping weather".
COMPLETE_YEAR_RE = re.compile(
    r"\b(?:complete|full|entire|whole)\s+(?:calendar\s+)?year\b"
    # "12 consecutive months of hourly data" is a year by another name, and
    # "95% valid readings during 2016" states the same requirement against a
    # named year. Neither matched, so the coverage the question asked for was
    # never applied and the cohort was whatever the plan happened to select.
    r"|\b(?:12|twelve)\s+consecutive\s+months\b"
    r"|\b\d{1,3}\s*%\s*(?:of\s+)?(?:valid|complete|present|non-?missing)"
    r"[^.?]{0,40}\b(?:readings?|data|timestamps?|hours?)\b", re.I)

#: "which buildings are eligible for", "which are suitable for" -- a question
#: about the cohort an analysis can be run on, rather than about its result.
ELIGIBILITY_RE = re.compile(
    r"\belig\w+\b|\bsuitable\s+for\b|\bqualif\w+\b|\bcan\s+support\b", re.I)

#: An analysis described as annual needs a year to be annual *over*.
ANNUAL_RE = re.compile(r"\bannual\w*\b|\byearly\b", re.I)

#: Normalising or benchmarking is comparison against like buildings, so the
#: attribute the comparison groups by has to be recorded. The snapshot already
#: takes this view: can_peer_benchmark requires a building type.
NORMALISED_RE = re.compile(
    r"\bnormali[sz]\w*\b|\bbenchmark\w*\b|\bpeer\w*\b", re.I)


#: "the highest median within-building variability", "average nighttime
#: consumption by building type" -- a statistic of the typical *building* in a
#: group, not of the group's pooled readings.
WITHIN_BUILDING_RE = re.compile(
    r"\bwithin[-\s]building\b|\bper[-\s]building\b|\beach\s+building'?s?\b",
    re.I)
GROUP_STATISTIC_RE = re.compile(r"\b(median|mean|average|typical)\b", re.I)


#: The statistic a question names. These are exact words, not preferences: a
#: question that says "median" is answered wrongly by a mean, however close
#: the two happen to be. Leaving the choice to the planner made the same
#: question answer 0.61 one run and 0.53 the next.
NAMED_STATISTIC_RE = re.compile(
    r"\b(median|mean|average|total|sum|maximum|minimum|max|min)\b", re.I)

_STATISTIC_CANON = {
    "median": "median", "mean": "mean", "average": "mean",
    "total": "total", "sum": "total",
    "maximum": "max", "max": "max", "minimum": "min", "min": "min",
}


#: "Which buildings ...", "which office buildings ...", "which streams ...".
#: A question of this shape is answered by a set, and the first thing a reader
#: needs about a set is how big it is.
WHICH_SET_RE = re.compile(
    # The noun must be plural -- "which site has the most" asks for one site,
    # not a set -- and must not be the first half of a compound, or
    # "building-use category" reads as a question about buildings.
    r"^\s*which\s+(?:\w+[-\s]){0,3}?(buildings|streams|sites|meters)\b(?!-)",
    re.I)


def asks_which_set(text: str) -> str | None:
    """The entity a "which ..." question is asking for a set of, if any.

    Every one of these is answered with a count first -- "285 buildings
    have...", "0 buildings...", "68 office buildings..." -- because the size of
    the set *is* the finding. Answered with a bare table the reply reads
    "Answer (rank_buildings): 217 rows", which is a description of the output
    rather than of the portfolio.
    """
    match = WHICH_SET_RE.match(text or "")
    if not match:
        return None
    word = match.group(1).lower()
    return word if word.endswith("s") else word + "s"


def resolve_named_statistic(text: str) -> str | None:
    """The statistic the question asks for, if it names one.

    The first one named wins: "which building had the lowest median daily load
    factor" is a question about medians, and the superlative in front of it
    says which end of them to report, not which statistic to compute.
    """
    # A measure can carry a statistic's name inside it. "Peak-to-average
    # ratio" is one measure, not a request for the mean, and reading it as one
    # set statistic="mean" on a comparison the reference reports as medians.
    lowered = text or ""
    masked = re.sub(r"\bpeak[\s-]?to[\s-]?average(?:\s+ratio)?\b"
                    r"|\bmoving\s+average\b|\baverage\s+ratio\b",
                    " ", lowered, flags=re.I)
    match = NAMED_STATISTIC_RE.search(masked)
    return _STATISTIC_CANON.get(match.group(1).lower()) if match else None


def resolve_group_statistic(text: str) -> dict | None:
    """How a per-group figure should be built: pooled, or building by building.

    Pooling every reading answers a question about the group's hours; the
    question usually asks about its typical building. On the Bull site that
    difference names a different winner, because seventeen assembly buildings
    outvote two public-service ones. And "median within-building variability"
    is the middle building's CV, not a CV computed from rolled-up columns --
    median(stddev)/median(mean) read 1.08 where the middle building sits
    at 0.70.
    """
    lowered = text or ""
    if not WITHIN_BUILDING_RE.search(lowered):
        return None
    stat = GROUP_STATISTIC_RE.search(lowered)
    word = (stat.group(1).lower() if stat else "mean")
    return {"weight": "building",
            "group_aggregate": "median" if word == "median" else "mean"}


def resolve_complete_year(text: str) -> dict | None:
    """A complete-calendar-year requirement, and whether weather must overlap.

    A coverage fraction over the whole record cannot express it: a building
    60% covered across two years can hold one clean year inside them, and one
    80% covered across both holds neither. Nor can it express "with
    overlapping weather", which is a property of the same year rather than of
    the site in general.
    """
    lowered = text or ""
    # "Eligible for annual EUI analysis" states the same requirement without
    # the word "complete": an annual intensity divides a year, so a building
    # that holds no year it can be computed over is not eligible for it. Both
    # signals are needed. "Annual EUI" alone describes the measure, not the
    # cohort, and narrowing every EUI question to buildings with a clean year
    # would answer a different question from the one asked; "eligible" alone
    # covers analyses, like a forecast, that need no calendar year at all.
    asks_eligibility = bool(ELIGIBILITY_RE.search(lowered))
    annual_eligibility = asks_eligibility and bool(ANNUAL_RE.search(lowered))
    if not COMPLETE_YEAR_RE.search(lowered) and not annual_eligibility:
        return None
    wants_weather = bool(re.search(
        r"\bweather\b|\btemperature\b|\boverlapping\b", lowered, re.I))
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", lowered)]
    # The floor the question states, when it states one. This was pinned at
    # 0.9 whatever was asked, so "at least 80% valid hourly electricity data"
    # screened at 90% -- a smaller set than the question describes, and no
    # sign in the answer that a different bar had been used.
    stated = re.search(r"\b(?:at\s+least\s+|>=\s*|minimum\s+(?:of\s+)?)?"
                       r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s+)?"
                       r"(?:valid|complete|coverage|present|non-?missing|"
                       r"hourly|readings?|data)", lowered, re.I)
    floor = 0.9
    if stated:
        value = float(stated.group(1))
        if 1.0 <= value <= 100.0:
            floor = round(value / 100.0, 4)
    # "One complete calendar year" and "at least 12 consecutive months" are
    # both twelve months of data and they are not the same cohort: a building
    # complete from July 2016 through June 2017 has the second and not the
    # first, and reading one as the other drops it or keeps it wrongly. On
    # this snapshot at a 95% floor the two rules differ by 28 buildings. A
    # question means the calendar only where it says so.
    basis = ("rolling"
             if re.search(r"\b(?:\d{1,2}|twelve|12)\s+consecutive\s+months\b",
                          lowered, re.I)
             and not re.search(r"\bcalendar\s+year\b", lowered, re.I)
             else "calendar")
    return {
        "coverage": floor,
        "basis": basis,
        "years": sorted(set(years)) or None,
        "weather": floor if wants_weather else None,
        # A normalised figure is read against comparable buildings, so a
        # building whose use is unrecorded cannot be placed and is not
        # eligible for the analysis, only for the arithmetic.
        "grouping_attribute": (annual_eligibility
                               and bool(NORMALISED_RE.search(lowered))),
    }


def resolve_temperature_split(text: str) -> str | None:
    """The basis for a cool/warm split, when a question asks for one.

    "Cool hours" and "warm hours" name a partition without naming a
    temperature, and no fixed number serves: 20 C is a cool hour in Austin and
    a warm one in Cardiff. The median of the site's own hours is the line;
    terciles are the sensitivity check on it.
    """
    lowered = text or ""
    if not TEMPERATURE_SPLIT_RE.search(lowered):
        return None
    if re.search(r"\btercile|\bthird|\bquantile|\bquartile", lowered, re.I):
        return "tercile"
    return "median"


def resolve_zero_run(text: str) -> dict | None:
    """A zero-consumption run the question asks about: length and which zeros.

    "Excluding missing observations" is not a footnote: ingest masks an exact
    zero to missing, so the two readings of "zero" differ completely -- the
    cleaned series holds none at all, while the days that were zero before
    cleaning number 71. The question says which it means.
    """
    lowered = text or ""
    found = ZERO_RUN_RE.search(lowered)
    if not found:
        return None
    hours = None
    # The length belongs to the zero-run clause, not to whichever duration
    # comes first in the sentence. "More than 2% missing timestamps, at least
    # one gap longer than 7 days, and repeated zero-consumption periods longer
    # than 72 hours" holds three criteria and two durations; reading from the
    # front took the gap's 7 days and asked for zero runs of a week.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(hour|day|week)s?",
                      lowered[found.end():], re.I)
    if match is None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(hour|day|week)s?", lowered, re.I)
    if match:
        unit = match.group(2).lower()
        hours = float(match.group(1)) * {"hour": 1, "day": 24, "week": 168}[unit]
    excluding_missing = bool(re.search(
        r"exclud\w*\s+missing|non-?missing|observed|present\s+readings", lowered, re.I))
    counting_masked = bool(re.search(
        r"\bbefore\s+clean\w*|\bpre-?clean\w*|\braw\s+(?:data|series|readings?)\b"
        r"|\bmasked\b|\bwent\s+dark\b|\bdark\s+(?:days|periods)\b", lowered, re.I))
    # "more than 72 hours" is a run of 73; the parameter is a floor, so the
    # bound the question states has to be converted into one.
    exceeding = bool(re.search(r"more\s+than|longer\s+than|exceed\w*|over\s+\d",
                               lowered, re.I))
    floor_hours = int(hours or 72) + (1 if exceeding else 0)
    return {
        "hours": floor_hours,
        "days": max(1, int(round((hours or 72) / 24))),
        # Ingest masks every exact zero to missing, so the cleaned series holds
        # no observed zeros at all and that basis can only ever answer nobody.
        # It is the right answer when the question excludes missing readings
        # and says so; otherwise "zero consumption" means the hours that were
        # zero, which are the masked ones.
        "basis": "observed" if excluding_missing else "masked",
    }


def asks_over_whole_population(text: str) -> bool:
    """Should the metric be computed for everyone it *can* be computed for?

    The fitness gate exists so that a building with three weeks of data does
    not appear third by EUI. That protection is right when the question ranks
    buildings and wrong when it counts them: "how many buildings are eligible"
    and "how much does EUI change when badly-covered buildings are excluded"
    are questions *about* the excluded set, and answering them over a cohort
    that already excluded it is answering a different question.
    """
    lowered = text or ""
    if asks_about_completeness(lowered) and re.search(r"\beui\b|intensity", lowered, re.I):
        return True
    return bool(POPULATION_RE.search(lowered))


def resolve_weather_grain(text: str) -> str | None:
    """The resolution a weather question names, if it names one.

    "The Pearson correlation between *hourly* air temperature and electricity"
    is a correlation of hourly pairs; computed on daily totals it is a
    different statistic and a different number -- 0.39 against 0.46 for the
    same building and year. The word is in the question and the tool takes the
    argument; nothing was carrying it across.
    """
    lowered = text or ""
    if re.search(r"\bhourly\b|\bper\s+hour\b|\bhour-by-hour\b", lowered, re.I):
        return "hourly"
    if re.search(r"\bdaily\b|\bper\s+day\b|\bday-by-day\b", lowered, re.I):
        return "daily"
    return None


def asks_weather_relationship(text: str) -> bool:
    """Is the question about the link between weather and consumption?"""
    lowered = text or ""
    return bool(WEATHER_RELATION_RE.search(lowered)
                and WEATHER_SUBJECT_RE.search(lowered))


def asks_about_completeness(text: str) -> bool:
    """Is the question about what the record is missing, not what it measures?"""
    lowered = (text or "").lower()
    if not COMPLETENESS_RE.search(lowered):
        return False
    # "missing" also appears in questions that merely qualify a measurement --
    # "mean consumption excluding missing hours" is a consumption question.
    return not re.search(r"\b(?:excluding|ignoring|after\s+removing|net\s+of)\b"
                         r"[^.]{0,40}\bmissing\b", lowered)


#: "both above X and above Y", "which remain in the top decile after ...",
#: "meets all three criteria" -- an entity has to satisfy every criterion, not
#: any of them. The union answers a different question and is usually larger:
#: 32 offices clear one of the two 90th percentiles, 25 clear both.
CONJUNCTION_RE = re.compile(
    r"\bboth\b[^.?]{0,120}\band\b"
    r"|\bremain\w*\b|\bstill\s+(?:in|above|among)\b"
    r"|\bas\s+well\s+as\b|\bsimultaneous\w*\b|\bat\s+the\s+same\s+time\b"
    r"|\b(?:meet|satisfy|clear)\w*\s+(?:all|every)\b"
    r"|\ball\s+(?:two|three|four|five)\s+\w+", re.I)


#: "above the 90th percentile", "in the top 10%", "the bottom decile" -- a
#: tail of the cohort, named as a share rather than as a number of buildings.
# "The largest 10% of buildings by floor area" is the same tail as "the top
# 10%", written the way an analyst says it. Only the superlatives that name a
# *side* count: "worst 10%" is a side on a metric whose direction the question
# supplies, and is left out here rather than guessed at.
TAIL_RE = re.compile(
    r"\b(?:above|over|in|within|among)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\s+percentile"
    r"|\b(?:top|largest|biggest|highest)\s+(\d{1,2})\s*%"
    r"|\b(?:bottom|smallest|lowest)\s+(\d{1,2})\s*%"
    r"|\b(top|bottom|highest|lowest)\s+decile\b", re.I)


def resolve_tail(text: str) -> dict | None:
    """The tail a question asks for, as a width in percent and a side.

    "Above the 90th percentile" and "the top 10%" name the same 10% of the
    cohort from opposite ends, and identify_peer_outliers takes the width. A
    ranking is not the same answer: it orders every building and leaves the
    line undrawn.
    """
    match = TAIL_RE.search(text or "")
    if not match:
        return None
    # Not every tail is a tail of the cohort. "The top 5% of hours" ranks a
    # building's own readings and says nothing about which buildings stand
    # out; selecting a tenth of the portfolio there would answer a question
    # about load concentration with a list of buildings.
    after = (text or "")[match.end():match.end() + 60].lower()
    if re.match(r"[\s\w-]{0,30}?\b(?:hours?|readings?|timestamps?|days?|months?)\b",
                after):
        return None
    percentile, top, bottom, decile = match.groups()
    if percentile is not None:
        value = int(percentile)
        # A percentile names a point; the tail is what lies beyond it, and
        # which side that is depends on which end the point sits near.
        return {"threshold": min(value, 100 - value),
                "side": "high" if value >= 50 else "low"}
    if top is not None:
        return {"threshold": int(top), "side": "high"}
    if bottom is not None:
        return {"threshold": int(bottom), "side": "low"}
    return {"threshold": 10,
            "side": "low" if decile.lower() in ("bottom", "lowest") else "high"}


#: "50% above the median EUI of office buildings in the same site", "twice its
#: category average" -- a threshold that is a property of the building's own
#: group, not of the cohort.
RELATIVE_GROUP_RE = re.compile(
    r"\b(?:the\s+)?same\s+(site|city|country|building[-\s]type|category|use)\b"
    r"|\b(?:its|their)\s+own\s+(site|city|country|category|building[-\s]type|use)\b"
    r"|\bwithin\s+(?:its|their)\s+(site|city|country|category|peer\s+group)\b", re.I)

#: "more than 50% above", "twice", "1.5x" -- how far above the group's figure.
ABOVE_BY_RE = re.compile(
    r"\b(?:more\s+than\s+)?(\d+(?:\.\d+)?)\s*%\s*(above|below|higher|lower)\b"
    r"|\b(twice|double)\b|\b(\d+(?:\.\d+)?)\s*(?:x|times)\s+(?:the\s+)?"
    r"(?:group|site|category|peer)?\s*(?:median|mean|average)\b", re.I)


#: Counts a question spells out. "Which five office buildings" is a top-5 and
#: the existing limit repair only ever looked for digits.
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "twelve": 12, "fifteen": 15, "twenty": 20}
TOP_N_RE = re.compile(
    r"\b(?:top|first|highest|worst)\s+(\d+|" + "|".join(WORD_NUMBERS) + r")\b"
    r"|\b(?:bottom|last|lowest|best)\s+(\d+|" + "|".join(WORD_NUMBERS) + r")\b"
    r"|\b(?:which|name|list)\s+(\d+|" + "|".join(WORD_NUMBERS) + r")\b", re.I)
DIRECTION_RE = re.compile(
    r"\b(highest|largest|greatest|most|worst|top|maximum|strongest|"
    r"biggest|longest)\b", re.I)


#: "exceeding their own annual mean by more than 3 standard deviations" --
#: the baseline is each building's own series, and the test is a z-score.
OWN_BASELINE_RE = re.compile(
    r"\b(?:its|their)\s+own\b[^.?]{0,60}\b(?:mean|average)\b"
    r"|\b(?:mean|average)\b[^.?]{0,30}\b(?:its|their)\s+own\b"
    # "above its annual mean" is the same baseline without the word "own":
    # the possessive already says whose mean it is. Requiring "own" missed
    # it, and the scan would have measured against the cohort instead.
    r"|\b(?:its|their)\s+(?:\w+\s+){0,2}(?:mean|average)\b"
    r"|\b(?:mean|average)\s+of\s+(?:its|their)\s+own\b", re.I)
SIGMA_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:standard\s+deviations?|sd|sigma|σ)\b", re.I)
#: "which buildings have at least one ..." counts buildings, not incidents.
AT_LEAST_ONE_RE = re.compile(
    r"\bat\s+least\s+(?:one|1|a\s+single)\b|\bany\s+(?:hourly\s+)?reading\b", re.I)


#: "when the threshold is reduced from mean + 3 SD to mean + 2 SD" -- the
#: question is about the change between two settings, not about either one.
SWEEP_RE = re.compile(
    r"\b(?:reduced|lowered|raised|changed|moved|relaxed|tightened)\s+from\b"
    r"|\bhow\s+does\b[^.?]{0,80}\bchange\b[^.?]{0,60}\bthreshold\b"
    r"|\bthreshold\b[^.?]{0,40}\bfrom\b[^.?]{0,40}\bto\b", re.I)


#: "remain in the top 10% ... after normalizing consumption by floor area" --
#: one tail drawn twice, on the measure before and the measure after, and the
#: answer is who survives both.
NORMALISATION_PAIRS = (
    (r"\bby\s+floor[-\s]?area\b|\bper\s+(?:square|sq\.?\s?m|m2|sqm)\b"
     r"|\bfloor[-\s]?area[-\s]?normali[sz]\w*", "total_kwh", "eui"),
    (r"\bper\s+occupant\b|\bby\s+occupan\w+\b", "total_kwh", "per_occupant"),
    (r"\bper\s+operating\s+hour\b", "total_kwh", "eui_per_operating_hour"),
)
AFTER_NORMALISING_RE = re.compile(
    r"\b(?:remain|stay|still)\w*\b[^.?]{0,80}\bafter\b|\bafter\s+normali[sz]\w*", re.I)


#: Every measure a question names, in the order it names them. resolve_measure
#: answers "which one" and returns the first; a comparison can ask for several.
METRIC_VOCABULARY = (
    ("daily_load_factor", r"\bdaily\s+load\s*factor\b"),
    ("load_factor", r"\bload\s*factor\b"),
    ("peak_to_average_ratio", r"\bpeak[\s-]?to[\s-]?average\b"),
    ("peak_power_density", r"\bpeak\s+power\s+density\b"),
    ("seasonal_cv", r"\bseasonal\s+variation\b|\bseasonal\s+variability\b"),
    ("eui", r"\beui\b|\benergy\s+use\s+intensity\b"),
    ("per_occupant", r"\bper\s+occupant\b"),
    ("overnight_baseload_per_area", r"\b(?:night|overnight)\w*\s+baseload\b"),
    ("baseload_per_area", r"\bbase[\s-]?load\b"),
    ("total_kwh", r"\btotal\s+(?:electricity\s+)?consumption\b"),
)


#: A specific measure contains the words of a general one: "daily load factor"
#: holds "load factor", "nighttime baseload" holds "baseload". When both
#: match the same span of text, only the specific one was named.
SUBSUMES = {"daily_load_factor": "load_factor",
            "overnight_baseload_per_area": "baseload_per_area",
            "peak_power_density": "peak_demand"}


def resolve_named_metrics(text: str) -> list[str]:
    """Each measure the question names, deduplicated, in order of appearance.

    "Compare load factor and peak-to-average ratio between offices and
    education" names two and the comparison takes one at a time unless told
    otherwise, so half the question went unanswered.
    """
    lowered = text or ""
    found = []
    for metric, pattern in METRIC_VOCABULARY:
        match = re.search(pattern, lowered, re.I)
        if match:
            found.append((match.start(), match.end(), metric))
    keep = []
    for start, end, metric in found:
        general = SUBSUMES.get(metric)
        # The general match sits inside the specific one's span, so it names
        # no measure of its own.
        covered = {g for s2, e2, m in found
                   for g in (SUBSUMES.get(m),)
                   if g and s2 <= start and end <= e2 or
                   (g == metric and any(s2 <= start and end <= e2
                                        for s3, e3, m3 in found if m3 != metric
                                        for s2, e2 in ((s3, e3),)))}
        if metric in covered:
            continue
        keep.append((start, metric))
    named = [m for _, m in sorted(keep)]
    for specific, general in SUBSUMES.items():
        if specific in named and general in named:
            named = [m for m in named if m != general]
    # "Load factor" on its own names two measures that differ by a third --
    # each day's mean over its own peak (0.762 for offices) and the year's
    # mean over its single highest hour (0.457). Neither is the obvious
    # reading, so an unqualified mention reports both, labelled, rather than
    # picking one and saying nothing about the choice.
    if named == ["load_factor"] and not re.search(
            r"\b(?:annual|yearly|overall|whole[-\s]year)\s+load\s*factor\b",
            lowered, re.I):
        named = ["daily_load_factor", "load_factor"]
    elif "load_factor" in named and "daily_load_factor" not in named and not re.search(
            r"\b(?:annual|yearly|overall)\s+load\s*factor\b", lowered, re.I):
        named = [("daily_load_factor" if m == "load_factor" else m) for m in named]
        named.insert(named.index("daily_load_factor") + 1, "load_factor")
    return named


def resolve_before_after_metrics(text: str) -> dict | None:
    """The two measures a "does it still qualify after X" question compares.

    "Which offices remain in the top 10% of intensity after normalizing by
    floor area" names one tail and two measures: the raw total before, the
    intensity after. Read as a single tail it answered the second half only --
    the 27 offices in the top decile by EUI, where the question asks which 9
    of the top decile by consumption are also there.
    """
    lowered = text or ""
    if not AFTER_NORMALISING_RE.search(lowered):
        return None
    for pattern, before, after in NORMALISATION_PAIRS:
        if re.search(pattern, lowered, re.I):
            return {"before": before, "after": after}
    return None


#: "breakout detection with a minimum segment length of 168 hours" -- a
#: persistent level shift, not an outlying hour.
BREAKOUT_RE = re.compile(
    r"\bbreakout\w*\b|\bbreak[-\s]?point\w*\b|\bchange[-\s]?point\w*\b"
    r"|\bregime\s+(?:shift|change)\b|\blevel\s+shift\b"
    r"|\bstructural\s+(?:break|change)\b", re.I)


#: "how does the weekday profile differ from the weekend profile" -- two
#: hour-of-day curves compared, not one curve over all days.
WEEKDAY_WEEKEND_RE = re.compile(
    # A building name sits between the two words often enough that a short
    # window misses them: "weekday hourly load profile of
    # Cockatoo_office_Alton differ from its weekend profile" is 68 characters
    # apart.
    r"\bweekday\w*\b[^.?]{0,120}\bweekend\w*\b"
    r"|\bweekend\w*\b[^.?]{0,120}\bweekday\w*\b", re.I)


def asks_weekday_weekend_profile(text: str) -> bool:
    """Does the question contrast weekday and weekend shapes?

    Answered with a single hour-of-day profile the difference it asks about
    is nowhere in the reply: one 24-row curve averaged over every day of the
    week, where the two curves differ by 19%.
    """
    lowered = text or ""
    # "load profiles" is the commoner phrasing and \bprofile\b does not match
    # it: the singular routed and the plural did not.
    return bool(WEEKDAY_WEEKEND_RE.search(lowered)
                and re.search(r"\bprofiles?\b|\bload\s+shapes?\b|\bhourly\b"
                              r"|\bcurves?\b|\bby\s+hour\b"
                              r"|\bhour[-\s]of[-\s]day\b", lowered, re.I))


def resolve_breakout(text: str) -> dict | None:
    """A changepoint question, with the minimum segment it asks for."""
    lowered = text or ""
    if not BREAKOUT_RE.search(lowered):
        return None
    segment = re.search(
        r"\b(?:minimum|min\.?|least)\s+(?:segment\s+)?(?:length\s+)?(?:of\s+)?"
        r"(\d+)\s*(hours?|days?|weeks?)\b", lowered, re.I)
    hours = 168
    if segment:
        unit = segment.group(2).lower()
        hours = int(segment.group(1)) * {"hour": 1, "day": 24, "week": 168}[
            unit.rstrip("s")]
    return {"min_segment_hours": hours}


def resolve_threshold_sweep(text: str) -> dict | None:
    """Two thresholds a question compares, in the order it states them.

    "How does the number of outliers change when the threshold is reduced from
    mean + 3 SD to mean + 2 SD" asks for both counts and the difference. Run
    at one threshold the answer is half the question: 0 readings above 3 SD,
    with nothing said about the 12 above 2 SD that the change adds.
    """
    lowered = text or ""
    if not SWEEP_RE.search(lowered):
        return None
    found = [float(m) for m in re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:standard\s+deviations?|sd|sigma|σ)\b",
        lowered, re.I)]
    seen, ordered = set(), []
    for value in found:
        if value not in seen:
            seen.add(value); ordered.append(value)
    if len(ordered) < 2:
        return None
    return {"thresholds": ordered[:4], "method": "zscore"}


#: "the Spearman correlation" -- ranks, not levels. Named explicitly or not
#: at all: unqualified "correlation" means Pearson.
SPEARMAN_RE = re.compile(r"\bspearman\b|\brank\s+correlation\b", re.I)
#: "on how many days did at least one outlier occur" counts dates, not
#: readings and not buildings.
DAY_COUNT_RE = re.compile(
    r"\bhow\s+many\s+days\b|\bon\s+how\s+many\s+days\b"
    r"|\bnumber\s+of\s+days\b|\bdays\s+(?:did|on\s+which)\b", re.I)


def asks_spearman(text: str) -> bool:
    """Is the correlation asked for a rank correlation?"""
    return bool(SPEARMAN_RE.search(text or ""))


def asks_day_count(text: str) -> bool:
    """Does the question count the days an event occurred on?"""
    return bool(DAY_COUNT_RE.search(text or ""))


def resolve_self_baseline(text: str) -> dict | None:
    """An anomaly test measured against each entity's own history.

    "Exceeding their own annual mean by more than 3 standard deviations" names
    the baseline (the building's own series), the test (a z-score) and the
    threshold. Scanned with the default contextual test the answer counted 571
    incidents against a rolling neighbourhood, where the question asks how
    many buildings ever cross their own line: 34 of 45.
    """
    lowered = text or ""
    if not OWN_BASELINE_RE.search(lowered):
        return None
    # A z-score scan needs a z. "20% above their annual average" names the
    # same baseline and a different test: broadening the baseline pattern made
    # every such question a three-sigma scan, because the threshold defaulted
    # to 3 when none was stated.
    sigma = SIGMA_RE.search(lowered)
    if not sigma:
        return None
    above = re.search(r"\bexceed\w*|\babove\b|\bover\b|\bhigher\b", lowered, re.I)
    below = re.search(r"\bbelow\b|\bunder\b|\bbeneath\b|\blower\s+than\b",
                      lowered, re.I)
    # One side or the other when the question names one. "Below their own mean
    # by 3 sigma" is the low tail, and scanning both returns the high tail's
    # buildings alongside, which is a different set.
    direction = ("high" if above and not below
                 else "low" if below and not above else "both")
    # What is being counted is the noun the question asks for, and it can be
    # asked either way round: "which buildings have at least one reading" and
    # "how many buildings exceed" both count buildings, not readings.
    # Whichever noun comes first after the interrogative is the one being
    # counted: "how many readings exceed their own building mean" counts
    # readings, and a pattern that merely finds "building" somewhere after
    # "how many" counts the wrong thing.
    asked = re.search(r"\b(?:which|what|how\s+many|list|name|identify)\b", lowered, re.I)
    counts_entities = bool(AT_LEAST_ONE_RE.search(lowered))
    if asked and not counts_entities:
        noun = re.search(r"\b(buildings?|sites?|meters?"
                         r"|readings?|observations?|hours?|timestamps?)\b",
                         lowered[asked.end():], re.I)
        counts_entities = bool(noun) and noun.group(1).lower().rstrip("s") in (
            "building", "site", "meter")
    return {"method": "zscore",
            "threshold": float(sigma.group(1)) if sigma else 3.0,
            "direction": direction,
            "report": "buildings" if counts_entities else "readings"}


def resolve_top_n(text: str) -> dict | None:
    """How many entities the question wants, and from which end.

    "Which five office buildings have the highest annual EUI" asks for an
    ordered five. A whole cohort ordered the wrong way contains the answer
    without stating it.
    """
    lowered = text or ""
    match = TOP_N_RE.search(lowered)
    if not match:
        return None
    # "the top 10%" is a share of the cohort, not ten buildings, and "its top
    # 5% highest-consumption hours" is a share of one building's readings.
    # Both parse as a count and neither is one; resolve_tail owns them.
    if lowered[match.end():match.end() + 2].lstrip().startswith("%"):
        return None
    raw = next((g for g in match.groups() if g), None)
    if raw is None:
        return None
    count = int(raw) if raw.isdigit() else WORD_NUMBERS[raw.lower()]
    if not 1 <= count <= 50:
        return None
    return {"limit": count,
            "order": "desc" if DIRECTION_RE.search(lowered) else "asc"}


#: "more than 20% above their annual average" -- the reference is the entity's
#: own figure over a longer window, not a group's and not a fixed number.
OWN_REFERENCE_RE = re.compile(
    r"\b(?:its|their)\s+(?:own\s+)?(?:annual|yearly|overall|average)\b"
    r"[^.?]{0,40}\b(?:average|mean|consumption|baseline)\b"
    r"|\bcompared\s+with\s+(?:its|their)\s+own\b", re.I)


def resolve_own_reference(text: str) -> dict | None:
    """A threshold measured against the entity's own longer-run figure.

    "Months when average daily consumption was more than 20% above their
    annual average" compares each building with itself over a longer window.
    Read as a cohort-wide figure it selects on the wrong line entirely, and
    read as a group figure it answers about the type rather than the building.
    """
    lowered = text or ""
    if not OWN_REFERENCE_RE.search(lowered):
        return None
    above = re.search(r"\b(\d+(?:\.\d+)?)\s*%\s*(above|below|higher|lower)\b",
                      lowered, re.I)
    if not above:
        return None
    multiple = 1 + float(above.group(1)) / 100.0
    return {"reference": "self",
            "at_least": multiple if above.group(2).lower() in ("above", "higher")
                        else None,
            "at_most": None if above.group(2).lower() in ("above", "higher")
                       else multiple}


def resolve_relative_threshold(text: str) -> dict | None:
    """A threshold stated as a multiple of the building's own group statistic.

    "More than 50% above the median EUI of office buildings in the same site"
    is nineteen thresholds, one per site, and which applies depends on the
    building being tested. Read as a single cohort-wide number it filters
    nothing.
    """
    lowered = text or ""
    group = RELATIVE_GROUP_RE.search(lowered)
    if not group:
        return None
    named = next((g for g in group.groups() if g), "").lower().replace(" ", "_")
    dimension = {"site": "site", "city": "city", "country": "country",
                 "building-type": "building_type", "building_type": "building_type",
                 "category": "building_type", "use": "building_type",
                 "peer_group": "building_type"}.get(named.replace("-", "_"))
    if dimension is None:
        return None
    statistic = ("mean" if re.search(r"\bmean\b|\baverage\b", lowered, re.I)
                 and not re.search(r"\bmedian\b", lowered, re.I) else "median")
    multiple = above = None
    match = ABOVE_BY_RE.search(lowered)
    if match:
        pct, direction, twice, times = match.groups()
        if pct is not None:
            multiple = 1 + float(pct) / 100.0
            above = direction.lower() in ("above", "higher")
        elif twice is not None:
            multiple, above = 2.0, True
        elif times is not None:
            multiple, above = float(times), True
    if multiple is None:
        return None
    return {"relative_to": dimension, "relative_statistic": statistic,
            "at_least": multiple if above else None,
            "at_most": None if above else multiple}


def asks_for_conjunction(text: str) -> bool:
    """Does the question want entities satisfying every criterion at once?"""
    return bool(CONJUNCTION_RE.search(text or ""))


def asks_which_are_eligible(text: str) -> bool:
    """Is the question *about* the eligible set, or merely restricted to it?

    The distinction is grammatical and it decides which step answers. "Which
    buildings are eligible for annual EUI analysis" asks for the set, and the
    tool that resolves the set has answered it. "Across all eligible office
    buildings, compare Seasonal Naive and TTM" uses eligibility to restrict a
    cohort and asks for a forecast comparison; treating that as an eligibility
    question would drop the comparison and answer with a count nobody wanted.

    Predicative eligibility -- "are eligible", "qualify for" -- is the ask.
    Attributive eligibility -- "eligible office buildings" -- is a filter.
    """
    lowered = (text or "").lower()
    return bool(re.search(
        r"\b(?:are|is|were|was)\s+(?:\w+\s+){0,2}elig\w+\b"
        r"|\bqualif(?:y|ies|ied)\s+for\b"
        r"|\bcan\s+support\b", lowered))


def asks_about_shape(text: str) -> bool:
    """Whether a comparison is about the shape of a profile, not its size."""
    lowered = (text or "").lower()
    return bool(re.search(SHAPE_SUBJECT, lowered)
                and re.search(SHAPE_CHANGE, lowered))


def resolve_measure(text: str, default: str = "eui") -> str:
    """The measure named by a question, from the vocabulary a tool accepts.

    Intensity is checked last: a question saying "total electricity consumption
    per square metre" is asking for intensity, and one saying "total electricity
    consumption" is not, so the more specific phrase has to win. Ordering the
    alternatives the other way makes every "total" question an EUI question.
    """
    lowered = (text or "").lower()
    # A measure named outright wins over one inferred from a word. The
    # intensity pattern matches "normali[sz]", so "load factor normalised
    # against age and floor area" was read as a question about EUI -- the one
    # measure the question does not mention.
    # Seasonal variation is the spread of a building's *monthly* levels, not
    # of its readings: asked which category varies most by season, the plan
    # returned four rows per building of per-season means -- the ingredients,
    # with the statistic never taken.
    for metric, pattern in (("seasonal_cv",
                             r"\bseasonal\s+variation\b|\bseasonal\s+variability\b"
                             r"|\bvari\w+\s+(?:by|across|between)\s+season"
                             r"|\bmonth[-\s]to[-\s]month\s+vari\w+"),
                            ("load_factor", r"\bload\s*factor\b"),
                            ("peak_to_average_ratio",
                             r"\bpeak[\s-]?to[\s-]?average\b")):
        if re.search(pattern, lowered):
            return metric
    # Consumption per person is its own measure, and nothing named it. "The
    # highest electricity consumption relative to their reported number of
    # usual occupants" fell through to the EUI default, so the rule route
    # ranked by floor area intensity and returned a different ten buildings --
    # right shape, wrong measure, and nothing in the answer saying so.
    if re.search(r"\bper\s+(?:occupant|person|capita|head|student|resident)s?\b"
                 r"|\b(?:relative\s+to|normalis\w+\s+(?:by|for)|divided\s+by|"
                 r"against)\s+(?:their\s+|the\s+)?(?:reported\s+)?"
                 r"(?:number\s+of\s+)?(?:usual\s+)?(?:occupant|person|people|"
                 r"student|resident)s?\b"
                 r"|\boccupant[-\s]normalis\w+\b", lowered):
        return "per_occupant"
    # An intensity divided by the hours the building is actually in use is a
    # third measure, not EUI with a note attached: the two rank the portfolio
    # differently, because a building that never turns off reads as operating
    # 24 hours a day.
    if (re.search(r"\b(?:operating|operational|occupied|in[\s-]?use)\s+hours?\b",
                  lowered)
            and re.search(INTENSITY_WORDS, lowered)):
        return "eui_per_operating_hour"
    # "Normalised baseload" is the floor per square metre, not EUI: the word
    # "normalised" alone matches the intensity pattern below and turned a
    # question about the overnight floor into one about annual consumption.
    if (re.search(r"\bbase[\s-]?load\b", lowered)
            and re.search(INTENSITY_WORDS + r"|\bnormali[sz]", lowered)):
        # "Nighttime baseload" names the overnight floor specifically -- the
        # mean between midnight and 05:00, in W/m2 -- not the 5th percentile
        # of the whole day's demand in kW/m2. They are different numbers
        # about different hours: 28.9 against 0.106 on this cohort, and a
        # question naming the night answered with the other one.
        if re.search(r"\b(?:night|nightt?ime|overnight|nocturnal)\b", lowered):
            return "overnight_baseload_per_area"
        return "baseload_per_area"
    # "Normalised" means per square metre when it qualifies consumption, and
    # "scaled to a common shape" when it qualifies a profile or a load curve.
    # Reading the second as an intensity turned a DTW clustering question into
    # an EUI question. The other intensity words still decide on their own.
    shape_normalised = re.search(
        r"\bnormali[sz]\w*\b[^.?]{0,40}\b(?:profile|curve|shape)s?\b", lowered)
    says_intensity = re.search(
        r"\beui\b|\bintensity\b|\bper\s+(?:square|sq|m2|sqm)\b|kwh/m", lowered)
    if shape_normalised and not says_intensity:
        return default
    if re.search(GEOGRAPHIC_METRICS[3][1], lowered):
        return "eui"
    for metric, pattern in GEOGRAPHIC_METRICS[:3]:
        if re.search(pattern, lowered):
            return metric
    return default


# The measures an index can be built from, and the words that name each one.
# An index is a combination the question composes, so what goes into it is
# read from the question rather than fixed: "using EUI, weather data and
# operating hours" names three, and the tool's default is those three.
INDEX_WORDS = (
    ("eui", r"\beui\b|\bintensity\b|\bper\s+(?:square|sq|m2|sqm)\b|kwh/m"),
    ("weather_sensitivity", r"\bweather\b|\btemperature\b|\bclimat\w*"),
    ("operating_hours", r"\b(?:operating|operational|occupied)\s+hours?\b"
                        r"|\bschedule\b|\bhours\s+of\s+(?:use|operation)\b"),
    ("baseload_per_area", r"\bbase[\s-]?load\b"),
    ("peak_power_density", r"\bpeak\s+(?:demand|power|load)\b"),
    ("load_factor", r"\bload\s*factor\b"),
    # How a building stands against the ones beside it, and how often it ran
    # far above its own normal. Both are criteria a prioritisation question
    # names in words, and neither had a component to name.
    ("peer_relative_eui", r"\bpeer[-\s]relative\b|\bpeer\s+performance\b"
                          r"|\brelative\s+to\s+(?:its|their|同)\s*peers\b"
                          r"|\bcompared\s+with\s+(?:its|their)\s+peers\b"),
    ("anomaly_count", r"\banomalous\s+(?:consumption|hours|readings)\b"
                      r"|\banomal\w+\b"),
)

INDEX_ASKED = (r"\bmulti[\s-]?variable\b|\bcomposite\b|\bcombined\s+(?:index|score)\b"
               r"|\b(?:performance|efficiency)\s+(?:index|score)\b"
               r"|\bindex\s+(?:using|combining|from|of)\b"
               # "Prioritised for investigation when considering A, B, C and D"
               # is a composite by another name: four measures, one ranking.
               r"|\bpriorit\w+\b[^.?]{0,80}\b(?:considering|based\s+on|using)\b"
               r"|\bwhen\s+considering\b[^.?]{0,80},")


def resolve_index_components(text: str) -> list[str] | None:
    """The measures a question wants combined into one index, or None.

    Two are the minimum: one measure combined with nothing is a ranking, and
    rank_buildings already answers that.
    """
    lowered = (text or "").lower()
    if not re.search(INDEX_ASKED, lowered):
        return None
    # "Prioritised for review based on being *both* above the 90th percentile
    # of X and of Y" names two measures and is not a weighted index: it is two
    # thresholds and an intersection, where every building either clears both
    # lines or does not. Scoring it would rank the ones that clear neither.
    if asks_for_conjunction(lowered) and re.search(
            r"\bpercentile\b|\bthreshold\b|\babove\b|\bbelow\b", lowered):
        return None
    named = [name for name, pattern in INDEX_WORDS if re.search(pattern, lowered)]
    return named if len(named) >= 2 else None


VINTAGE_BOUNDS = (
    # "Before 1980" excludes 1980 and "since 1980" includes it. Both catalog
    # bounds are inclusive, so the exclusive readings take the year next to
    # the one named -- written out here rather than left to whoever reads the
    # plan to notice.
    (r"\b(?:before|prior\s+to|earlier\s+than)\s+(\d{4})\b",
     lambda y: ("max_year_built", y - 1, f"built before {y}")),
    (r"\bpre[\s-]?(\d{4})\b",
     lambda y: ("max_year_built", y - 1, f"built before {y}")),
    (r"\b(?:after|later\s+than)\s+(\d{4})\b",
     lambda y: ("min_year_built", y + 1, f"built after {y}")),
    (r"\bpost[\s-]?(\d{4})\b",
     lambda y: ("min_year_built", y + 1, f"built after {y}")),
    (r"\bsince\s+(\d{4})\b",
     lambda y: ("min_year_built", y, f"built since {y}")),
    (r"\b(?:in\s+or\s+before|no\s+later\s+than)\s+(\d{4})\b",
     lambda y: ("max_year_built", y, f"built by {y}")),
)

VINTAGE_WORDS = (r"\bbuilt\b|\bvintage\b|\bage[sd]?\b|\bconstruct\w*\b"
                 r"|\berected\b|\byears?\s+old\b|\bnewer\b|\bolder\b")

# An age stated as a span rather than a year -- "10 years old or newer" --
# still names a year_built cohort, but only once it is read against the year
# the question is about. The two are not interchangeable: the same phrase is
# 2007 in a question about 2017 and 2006 in one about 2016, and a cohort off
# by a year moves whichever buildings were built in it.
RELATIVE_AGE = (
    (r"\b(\d{1,3})\s+years?\s+(?:old\s+)?or\s+(?:newer|younger|less)\b"
     r"|\b(?:newer|younger|less)\s+than\s+(\d{1,3})\s+years?\s*(?:old)?\b"
     r"|\bbuilt\s+(?:in\s+)?the\s+last\s+(\d{1,3})\s+years?\b",
     "min_year_built", 0, "{n} years old or newer"),
    (r"\b(?:older|more)\s+than\s+(\d{1,3})\s+years?\s*(?:old)?\b"
     r"|\b(\d{1,3})\s+years?\s+(?:old\s+)?or\s+(?:older|more)\b",
     "max_year_built", -1, "older than {n} years"),
)

# Where a question names one age cohort and contrasts it with "older" or
# "newer" buildings, the second cohort is the complement of the first and is
# never spelled out. Reading only the stated half answers half the question.
AGE_COMPLEMENT = re.compile(
    r"\b(?:older|newer|younger)\s+(?:\w+\s+){0,2}buildings?\b"
    r"|\bthe\s+(?:older|newer|younger)\s+(?:ones|group|cohort)\b"
    r"|\bthan\s+(?:the\s+)?(?:older|newer|younger)\b", re.I)


def _calendar_year(start: str | None, end: str | None) -> int | None:
    """The calendar year a start/end pair spans, if it spans exactly one.

    A plan says 2017 as ``start="2017-01-01", end="2018-01-01"``; the
    performance tools take ``year=2017``. Anything that is not one whole year
    -- a quarter, eighteen months, an open end -- has no year to hand over and
    returns None rather than a year the caller did not ask for.
    """
    if not start or not end:
        return None
    try:
        first, last = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    except ValueError:
        return None
    if first.month != 1 or first.day != 1:
        return None
    if last == date(first.year + 1, 1, 1) or last == date(first.year, 12, 31):
        return first.year
    return None


def _reference_year(text: str) -> int:
    """The year a relative age is counted back from.

    A question about 2017 that says "10 years old" means built in 2007. Where
    it names no year, the snapshot's latest full year is the one in view.
    """
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text or "")
             if 2000 <= int(y) <= LATEST_FULL_YEAR]
    return max(years) if years else LATEST_FULL_YEAR


# A year written next to a build word is an age, not a period: the catalog
# carries year_built back to the 1800s while the measurements cover two years.
VINTAGE_NAMED = r"\b(?:built|constructed|erected)\s+(?:in\s+)?(\d{4})\b"


def vintage_years(text: str) -> set[int]:
    """The years a question names as building ages rather than as periods.

    The scope gate refuses a question naming a year the snapshot does not
    cover, which is right for "consumption in 2015" and wrong for "buildings
    built before 1980" -- the second asks nothing of the measurement window.
    """
    lowered = (text or "").lower()
    if not re.search(VINTAGE_WORDS, lowered):
        return set()
    # A relative age names no year at all, so it contributes none here; the
    # year it resolves against is one the question already states as a period.
    return {int(match.group(1))
            for pattern in [p for p, _ in VINTAGE_BOUNDS] + [VINTAGE_NAMED]
            for match in re.finditer(pattern, lowered)}


def resolve_vintages(text: str) -> list[tuple[str, str, int]]:
    """The building-age cohorts a question names, in the order it names them.

    "Buildings built before 1980 versus those built after 2010" is two cohorts
    of the catalog, not a period filter on the measurements: 1980 is a vintage,
    and read as a year of data it selects nothing this snapshot holds. A year
    only reads as a vintage where the question says so -- "since 2016" is a
    period in every question that is not about how old a building is.
    """
    lowered = (text or "").lower()
    if not re.search(VINTAGE_WORDS, lowered):
        return []
    found: dict[int, tuple[str, str, int]] = {}
    for pattern, interpret in VINTAGE_BOUNDS:
        for match in re.finditer(pattern, lowered):
            parameter, value, label = interpret(int(match.group(1)))
            found[match.start()] = (label, parameter, value)
    reference = _reference_year(lowered)
    for pattern, parameter, offset, template in RELATIVE_AGE:
        for match in re.finditer(pattern, lowered):
            span = next((g for g in match.groups() if g), None)
            if span is None:
                continue
            years = int(span)
            found[match.start()] = (template.format(n=years), parameter,
                                    reference - years + offset)
    cohorts = [found[position] for position in sorted(found)]
    # One stated cohort contrasted with "older buildings" is two cohorts: the
    # unstated one is everything the first bound excludes.
    if len(cohorts) == 1 and AGE_COMPLEMENT.search(lowered):
        label, parameter, value = cohorts[0]
        if parameter == "min_year_built":
            cohorts.append((f"older than {reference - value} years",
                            "max_year_built", value - 1))
        elif parameter == "max_year_built":
            cohorts.append((f"{reference - value - 1} years old or newer",
                            "min_year_built", value + 1))
    return cohorts


ADJUSTABLE = {"year_built": r"\bage\b|\byear[\s-]?built\b|\bvintage\b",
              "floor_area_m2": r"\bfloor\s*area\b|\bsize\b|\bsqm\b|\bsquare\s*met",
              "occupants": r"\boccupan\w*\b|\bpeople\b|\bhead\s*count\b"}


def resolve_adjustments(text: str) -> list[str]:
    """The building characteristics a question asks a measure to be read against.

    "Normalised against their building age and floor area percentiles" is not
    a division by either one: it asks what the measure looks like once age and
    size are accounted for, which is a residual.
    """
    lowered = (text or "").lower()
    if not re.search(r"\bnormali[sz]\w*\s+(?:against|for|by)\b|\badjust\w*\s+for\b"
                     r"|\bcontrol\w*\s+for\b|\brelative\s+to\s+(?:their|its)\s+"
                     r"(?:\w+\s+){0,2}(?:age|size)\b|\bpercentiles?\b", lowered):
        return []
    return [name for name, pattern in ADJUSTABLE.items()
            if re.search(pattern, lowered)]


ORDINAL_PERCENTILE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:%|percent)?\s*[- ]?percentile\b", re.I)


def resolve_percentile(text: str) -> float | None:
    """The percentile a question names, as a fraction, or None.

    "The 5th percentile of demand" fixes the estimator and its level, and a
    question that goes to the trouble of naming one means it: the default
    overnight mean is a different measure, not a rounding of the same one.
    """
    match = ORDINAL_PERCENTILE.search(text or "")
    if not match:
        return None
    level = int(match.group(1))
    return level / 100 if 0 < level < 100 else None


COMPARISON = re.compile(
    r"\b(?:exceed\w*|above|over|more\s+than|greater\s+than|higher\s+than|"
    r"at\s+least|below|under|less\s+than|lower\s+than|at\s+most)\b"
    r"[\s,]*(?:a\s+)?(\d+(?:\.\d+)?)", re.I)


def resolve_threshold(text: str) -> tuple[float | None, float | None]:
    """A stated bound on a measure: "exceeding 50 W/m2" is at_least=50.

    Only an explicit comparative counts. "The top 10%" and "10 buildings" are
    not thresholds, and reading them as one silently filters a ranking.
    """
    # A number followed by a span of time bounds the *record*, not the
    # measure: "at least 12 months of continuous data" says how long the
    # stream must run, and reading it as at_least=12 filtered a set question
    # by an intensity of 12 kWh/m2 -- the right buildings found and then cut
    # by a threshold nobody stated.
    for match in COMPARISON.finditer(text or ""):
        tail = (text or "")[match.end():match.end() + 24].lower()
        if not re.match(r"\s*(?:%|percent)?\s*(?:months?|days?|weeks?|years?"
                        r"|hours?|minutes?)\b", tail):
            break
    else:
        return None, None
    if not match:
        return None, None
    value = float(match.group(1))
    lower = re.search(r"\b(?:exceed\w*|above|over|more\s+than|greater\s+than"
                      r"|higher\s+than|at\s+least)\b", match.group(0), re.I)
    return (value, None) if lower else (None, value)


CLOCK_WORDS = {"midnight": 0, "noon": 12}

# Two shapes join the ends of a clock window: "2 PM to 6 PM" and
# "between 2 PM and 6 PM". "and" is only admitted after a "between", because
# on its own it joins far more things than times -- "buildings 5 and 10" --
# and even there the am/pm guard below is what keeps a count out.
CLOCK_RANGE = re.compile(
    r"\b(?:between\s+)?(midnight|noon|\d{1,2})(?::00)?\s*(a\.?m\.?|p\.?m\.?)?\s*"
    r"(?:to|until|till|through|-|\u2013)\s*"
    r"(midnight|noon|\d{1,2})(?::00)?\s*(a\.?m\.?|p\.?m\.?)?\b"
    r"|\bbetween\s+(midnight|noon|\d{1,2})(?::00)?\s*(a\.?m\.?|p\.?m\.?)?\s*"
    r"and\s+(midnight|noon|\d{1,2})(?::00)?\s*(a\.?m\.?|p\.?m\.?)?\b", re.I)


def _clock_hour(token: str, meridiem: str | None) -> int | None:
    """One clock time as an hour of the day, or None if it is not one."""
    token = token.strip().lower()
    if token in CLOCK_WORDS:
        return CLOCK_WORDS[token]
    if not token.isdigit() or int(token) > 24:
        return None
    hour, suffix = int(token), (meridiem or "").replace(".", "").lower()
    if suffix.startswith("p") and hour != 12:
        hour += 12
    elif suffix.startswith("a") and hour == 12:
        hour = 0
    return hour % 24


def resolve_clock_window(text: str) -> tuple[int, int] | None:
    """A phrase like "midnight to 4 AM" as the first and last hour it covers.

    The hour named at the end of such a phrase is where the window stops, not
    an hour inside it: "midnight to 4 AM" is 00:00 through 03:59, which is
    hours 0 to 3. Reading it as hours 0 to 4 takes in a fifth hour the
    question excludes, and for the top decile of office baseload ratio that
    one hour changes eight of the twenty-nine buildings.

    At least one side has to be an unambiguous clock time -- "midnight",
    "noon", or an am/pm -- so that "buildings 5 to 10" is not read as a time.
    """
    match = CLOCK_RANGE.search(text or "")
    if not match:
        return None
    groups = match.groups()
    # Whichever of the two alternatives fired supplies the four ends; the
    # other alternative's groups are None.
    opens, open_suffix, closes, close_suffix = (
        groups[:4] if groups[0] is not None else groups[4:])
    if not (open_suffix or close_suffix
            or opens.lower() in CLOCK_WORDS or closes.lower() in CLOCK_WORDS
            or ":" in match.group(0)):
        # A 24-hour "14:00 to 18:00" is as unambiguous as an am/pm, so the
        # colon counts as the marker. Only a whole hour is admitted: the
        # pattern does not match "14:30", which keeps a half-hour window from
        # being silently rounded down to the hour.
        return None
    start, end = _clock_hour(opens, open_suffix), _clock_hour(closes, close_suffix)
    if start is None or end is None or start == end:
        return None
    return start, (end - 1) % 24


# --- what an AI-analytics question asks for ---------------------------------
# These three categories (forecasting, anomaly detection, pattern discovery)
# differ from the rest in one way that matters to planning: the question names
# a *method*, and naming a different one is a wrong answer even when the
# numbers are defensible. "Using a simple 7-day trailing moving average" is not
# a description of the data wanted, it is an instruction, and a planner that
# reads only the entity and the date will run the wrong model and report it
# with confidence. So the method is resolved from the question's own words,
# and where the question names none, the default is stated rather than implied.

# Model phrases, most specific first: a question naming a library estimator
# means that estimator, and one describing a procedure means the model that
# implements it. Order matters -- "naive forecast using a 7-day moving average"
# names the average, not the naive baseline, and the average is matched first.
FORECAST_MODEL_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bmoving\s+average\b|\brolling\s+(?:mean|average)\b"
     r"|\btrailing\s+(?:mean|average)\b|\brunning\s+average\b", "moving_average"),
    # "AutoARIMA" is one word, so \barima\b never matches inside it, and the
    # question that names two models came back naming one. It is also a
    # *different model* from the fixed-order ARIMA below -- it searches the
    # orders rather than being given them -- so it maps to its own entry and
    # is listed first, because the plain pattern would otherwise claim it.
    (r"\bauto[-\s]?arima\b|\bautomatic\s+arima\b", "auto_arima"),
    (r"\bsarima\b|\barima\b|\bbox[-\s]?jenkins\b", "arima"),
    (r"\bexponential\s+smoothing\b|\bholt[-\s]?winters?\b|\bets\b", "ets"),
    (r"\btheta\s+(?:method|model|forecast)\b", "theta"),
    (r"\bfoundation\s+model\b|\bchronos\b|\bzero[-\s]?shot\b|\bpre[-\s]?trained\b"
     r"|\bprobabilistic\b|\bprediction\s+interval\b|\bquantifies?\s+uncertainty\b"
     r"|\bconfidence\s+interval\b|\buncertainty\s+band\b", "chronos"),
    (r"\b(?:air\s+)?temperature\b|\bweather[-\s]?(?:normal\w+|driven|based)\b"
     r"|\bdegree[-\s]?days?\b|\bthermal\b", "weather_regression"),
    (r"\blight\s?gbm\b|\blgbm\b|\blightgbm\b", "lightgbm"),
    (r"\bk[-\s]?nn\b|\bk[-\s]nearest[-\s]neighbou?rs?\b"
     r"|\bnearest[-\s]neighbou?r\s+(?:model|forecast\w*|regress\w*)\b",
     "knn"),
    (r"\btiny\s+time\s+mixers?\b|\bttm\b|\bgranite[-\s]timeseries\b",
     "tiny_time_mixer"),
    (r"\bgradient\s+boost\w*\b|\bboosted\s+trees?\b|\brandom\s+forest\b"
     r"|\bmachine[-\s]?learning\b|\bgbm\b|\bxgboost\b", "gradient_boosting"),
    (r"\blinear\s+regression\b|\bleast[-\s]?squares\b|\bharmonic\s+regression\b",
     "linear"),
    # "Naive" and "Naive" with a diaeresis are the same model. The set writes
    # it the second way throughout, and the pattern matched neither question.
    (r"\bna[iï]ve\b|\bpersistence\b|\bsame\s+weekday\b|\bone\s+week\s+earlier\b"
     r"|\blast\s+week\b|\bsame\s+(?:day|hour)\s+last\s+week\b"
     r"|\bseasonal\s+na[iï]ve\b",
     "seasonal_naive"),
)

# "a simple 7-day trailing moving average" -- the window belongs to the
# average, so the number is only read when it sits beside one of its words.
MOVING_WINDOW = re.compile(
    r"\b(\d{1,3})[-\s]?(?:day|d)\b(?=[^.]{0,40}?"
    r"(?:moving|rolling|trailing|running|average|mean|window))"
    r"|(?:moving|rolling|trailing|running)\s+(?:average|mean)\s+"
    r"(?:over|of|across)\s+(?:the\s+(?:last|past|previous)\s+)?(\d{1,3})\s*(?:day|d)\b",
    re.I)

# How far ahead. "7-day-ahead" and "next 48 hours" are horizons; "for the 24
# hours of 2017-08-01" is a horizon that also pins the day, and the day is read
# by the ordinary date resolver.
HORIZON_HOURS = re.compile(
    r"\bnext\s+(\d{1,4})\s*(hour|day|week|month)s?\b"
    r"|\b(\d{1,4})[-\s]?(hour|day|week|month)[-\s]?ahead\b"
    r"|\bfor\s+the\s+(\d{1,4})\s+hours\s+of\b", re.I)

HORIZON_MULTIPLIER = {"hour": 1, "day": 24, "week": 168, "month": 24 * 30}


def resolve_forecast_model(text: str) -> tuple[str, bool]:
    """Which forecaster the question named, and whether it named one at all.

    The second value is what stops a default from being reported as a choice.
    A question that says only "forecast tomorrow" gets the seasonal naive
    baseline, and the answer should say that this was the planner's decision
    rather than the analyst's.
    """
    lowered = (text or "").lower()
    for pattern, model in FORECAST_MODEL_PHRASES:
        if re.search(pattern, lowered):
            return model, True
    return "seasonal_naive", False


def resolve_forecast_models(text: str) -> list[str]:
    """Every forecaster the question names, in the order it names them.

    "How accurately do LightGBM and ARIMA forecast..." names two, and reading
    only the first answers half the question with no sign that the other half
    was dropped. Order is the question's, because that is the order a reader
    will look for the rows in.
    """
    lowered = (text or "").lower()
    found: list[tuple[int, str]] = []
    for pattern, model in FORECAST_MODEL_PHRASES:
        match = re.search(pattern, lowered)
        if match and model not in {m for _, m in found}:
            found.append((match.start(), model))
    return [model for _, model in sorted(found)]


# "How accurately", "what is its MAPE", "validate", "backtest". A question
# about a model's *error* is not a request for a forecast: it is a request to
# score forecasts that have already been checked against what happened.
ACCURACY_ASKED = re.compile(
    r"\bhow\s+(?:accurate|accurately|well|closely)\b|\baccurac\w+\b"
    # MAE by name. The list held mape, rmse, cvrmse and mase and not the
    # commonest of them, so "which model has the lower MAE" and "which
    # buildings have the lowest and highest MAE" were not read as accuracy
    # questions at all: both were answered with a single forecast.
    r"|\bmape\b|\bmae\b|\brmse\b|\bcv\s?rmse\b|\bcvrmse\b|\bmase\b"
    r"|\bmean\s+absolute\s+error\b"
    r"|\bmean\s+absolute\s+(?:percentage\s+)?error\b"
    r"|\bforecast\w*\s+error\b|\berror\s+(?:rate|metric)\b"
    r"|\bback[-\s]?test\w*\b|\bvalidat\w+\b|\bskill\s+score\b",
    re.I)

# Which accuracy measures to report, when the question names any.
ACCURACY_METRICS = (
    ("mape", r"\bmape\b|\bmean\s+absolute\s+percentage\s+error\b"),
    ("cvrmse", r"\bcv\s?rmse\b|\bcvrmse\b|\bcoefficient\s+of\s+variation\b"),
    ("rmse", r"\brmse\b|\broot\s+mean\s+squared?\s+error\b"),
    ("mae", r"\bmae\b|\bmean\s+absolute\s+error\b"),
    ("mase", r"\bmase\b"),
    ("bias", r"\bbias\b|\bsystematic\s+error\b"),
)


def asks_forecast_accuracy(text: str) -> bool:
    """Does the question ask how well a forecast did, rather than for one?"""
    return bool(ACCURACY_ASKED.search(text or ""))


def resolve_accuracy_metrics(text: str) -> list[str]:
    """The error measures the question names, most specific first.

    Named measures lead, because "what are their respective MAPE values" wants
    MAPE in the first column and everything else as context. Where none is
    named the usual three are reported.
    """
    lowered = (text or "").lower()
    named = [metric for metric, pattern in ACCURACY_METRICS
             if re.search(pattern, lowered)]
    rest = [metric for metric, _ in ACCURACY_METRICS if metric not in named]
    return (named + rest)[:4] if named else ["mape", "mae", "rmse"]


# "What improvement in MAPE is obtained by adding air temperature to the
# LightGBM model" -- the answer is the change between two variants, not two
# numbers the reader subtracts. The phrasing always names the thing added and
# the thing it is added to.
IMPROVEMENT_ASKED = re.compile(
    r"\bwhat\s+(?:improvement|gain|benefit|difference)\b"
    r"|\bhow\s+much\s+(?:does|do|would)\b[^.?]{0,40}\bimprov\w*"
    r"|\bimprovement\s+(?:in|to|of)\b"
    r"|\b(?:improv\w*|gain|benefit|effect|impact)\b[^.?]{0,30}"
    r"\b(?:by|from|of)\s+(?:adding|including|introducing|using)\b"
    r"|\b(?:adding|including|introducing)\b[^.?]{0,40}\bimprov\w*"
    r"|\bwith\s+(?:and|versus|vs\.?)\s+without\b"
    r"|\bablation\b", re.I)

# What is being added, and which registry name that turns a model into.
ADDED_FEATURE = (
    ("weather", r"\b(?:air\s+)?temperature\b|\bweather\b|\bthermal\b"
                r"|\bdegree[-\s]?days?\b"),
)


def asks_improvement(text: str) -> bool:
    """Does the question ask for the change between two variants?"""
    return bool(IMPROVEMENT_ASKED.search(text or ""))


def resolve_ablation(text: str) -> tuple[str, str] | None:
    """The baseline model and the same model with the named feature added.

    Only pairs the registry actually holds are returned. Answering "what does
    adding temperature give ARIMA" with some other model's weather variant
    would be a different experiment reported as this one, so an unsupported
    pair is no pair at all and the caller falls back to a plain comparison.
    """
    from ..tools import forecast_models
    lowered = (text or "").lower()
    if not any(re.search(pattern, lowered) for _, pattern in ADDED_FEATURE):
        return None
    for model in resolve_forecast_models(text) or []:
        variant = forecast_models.WITH_WEATHER.get(model)
        if variant:
            return model, variant
    return None


# The weather a question hands a forecaster, named as the dataset spells it or
# as a person would.
WEATHER_FEATURE_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bair\s?temperature\b|\bairtemperature\b|\boutdoor\s+temperature\b"
     r"|\btemperature\b", "air_temperature"),
    (r"\bdew\s?temperature\b|\bdewtemperature\b|\bdew\s?point\b",
     "dew_temperature"),
    (r"\bcloud\s?coverage\b|\bcloudcoverage\b|\bcloud\s?cover\b",
     "cloud_coverage"),
)


def resolve_weather_variables(text: str) -> list[str] | None:
    """Which weather variables the question names, if any."""
    lowered = (text or "").lower()
    found = [name for pattern, name in WEATHER_FEATURE_PHRASES
             if re.search(pattern, lowered)]
    return found or None


def asks_pooled_metadata(text: str) -> bool:
    """Does the question ask what building metadata adds to a forecast?

    Metadata does not vary hour to hour, so it can only be a feature of a
    model fitted across buildings. A question asking whether it helps is
    therefore asking for a pooled fit, whether or not it says so.
    """
    lowered = (text or "").lower()
    return bool(
        re.search(r"\b(?:building\s+)?meta[\s-]?data\b", lowered)
        and re.search(r"\bincorporat\w+|\badding\b|\binclud\w+|\bwith\b"
                      r"|\bimprov\w+|\bfeature", lowered)
        and re.search(r"\bforecast\w*|\bpredict\w*|\bmae\b|\brmse\b"
                      r"|\bmape\b", lowered))


def asks_detector_validation(text: str) -> bool:
    """Does the question ask whether a detector is *better*, not what it found?

    "Does adding change features improve detection" cannot be answered by
    counting flags -- more is not better and fewer is not worse. It needs
    labels, and the only labels available are ones we plant.
    """
    lowered = (text or "").lower()
    return bool(
        re.search(r"\bimprov\w+|\bbetter\b|\bevaluate\s+whether\b"
                  r"|\bvalidat\w+|\bfalse[-\s](?:positive|negative)"
                  r"|\bprecision\b|\brecall\b|\bf1\b"
                  r"|\bpseudo[-\s]?label|\binjected?\b|\bsynthetic\b",
                  lowered)
        and re.search(r"\banomal\w*|\bdetect\w*|\boutlier\w*|\blof\b",
                      lowered))


def asks_sensitivity(text: str) -> bool:
    """Is the question about how much the answer depends on the settings?"""
    return bool(re.search(
        r"\bhow\s+sensitiv\w*|\bsensitivit\w+\b|\bvary\s+with\s+the\b"
        r"|\bsweep\b|\bacross\s+thresholds?\b"
        r"|\bwhen\s+the\s+\w+\s+threshold\s+is\s+(?:increased|raised|"
        r"changed|varied)", text or "", re.I))


def asks_cluster_stability(text: str) -> bool:
    """Is the question about whether a clustering holds up?"""
    lowered = (text or "").lower()
    return bool(re.search(r"\bstab\w+|\bconsisten\w+|\bhold\s+up\b"
                          r"|\brobust\w*", lowered)
                and re.search(r"\bcluster\w*|\bgroup\w*|\bsegment\w*",
                              lowered))


def asks_explanation(text: str) -> bool:
    """Does the question ask what explains consumption, rather than what it is?"""
    lowered = (text or "").lower()
    return bool(re.search(
        r"\bexplain\w*\b|\bexplanator\w+|\bstrongest\s+explanation\b"
        r"|\bwhich\s+combination\s+of\b|\bdrivers?\s+of\b"
        r"|\bregression\s+coefficients?\b|\bvar(?:y|ies)\s+(?:with|across)\b"
        r"|\bwhat\s+accounts\s+for\b", lowered))


#: "starting at 00:00 on 8 May 2016", "starting on 1 March 2016", "from
#: 2016-05-08". The origin a forecast is issued from, when the question fixes
#: one. A question that fixes an origin is asking for a single forecast; a
#: question that gives only a window ("over March and April") is asking for a
#: rolling origin, and the two are different measurements.
FORECAST_ORIGIN = re.compile(
    r"\b(?:start\w*|beginning|from|as\s+of|origin)\b[^.?]{0,30}?"
    r"(?:(\d{1,2})[:.](\d{2})\s*(?:on\s+)?)?"
    r"(?:(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})"
    r"|([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})"
    r"|(\d{4})-(\d{2})-(\d{2}))", re.I)


def resolve_forecast_origin(text: str) -> str | None:
    """The single date a forecast is issued from, as YYYY-MM-DD, or None."""
    match = FORECAST_ORIGIN.search(text or "")
    if not match:
        return None
    # "from 1 March 2016 **to** 30 April 2016" is a window, and a window is
    # what the rolling origin is for. Only a date with no closing date after
    # it fixes an origin.
    after = (text or "")[match.end():match.end() + 40].lower()
    if re.match(r"\s*(?:to|through|until|till|-|\u2013)\s*"
                r"(?:\d{1,2}\s+[a-z]{3,9}|[a-z]{3,9}\s+\d{1,2}|\d{4}-\d{2})",
                after):
        return None
    _, _, d1, m1, y1, m2, d2, y2, y3, m3, d3 = match.groups()
    months = {name[:3].lower(): n for name, n in MONTHS.items()}
    try:
        if y3:
            return f"{int(y3):04d}-{int(m3):02d}-{int(d3):02d}"
        if y1:
            month = months.get(str(m1)[:3].lower())
            return None if not month else f"{int(y1):04d}-{month:02d}-{int(d1):02d}"
        if y2:
            month = months.get(str(m2)[:3].lower())
            return None if not month else f"{int(y2):04d}-{month:02d}-{int(d2):02d}"
    except (TypeError, ValueError):
        return None
    return None


#: "with a 24-hour season", "season length 24", "m=24", "using a weekly
#: season". The season a seasonal model repeats from, when the question says.
SEASON_LENGTH = re.compile(
    r"\b(\d{1,3})[-\s]?hour\s+season\b|\bseason\s*(?:length|=|of)\s*(\d{1,3})\b"
    r"|\b(?:m|sp)\s*=\s*(\d{1,3})\b|\b(daily|weekly)\s+season\b", re.I)


#: Whether the question asks for normalised profiles. "un-normalised" and
#: "raw" contain the word and mean its opposite: Q46 says "cluster the raw,
#: un-normalised profiles" and a plain search for "normalis" read that as a
#: request for exactly what it rules out.
WANTS_NORMALISED = re.compile(
    r"(?<!un-)(?<!un)(?<!not )(?<!non-)\b(?:normali[sz]\w*|standardi[sz]\w*"
    r"|z[-\s]?normali[sz]\w*)\b", re.I)
RULES_OUT_NORMALISED = re.compile(
    r"\bun[-\s]?normali[sz]\w*\b|\braw\b|\bnon[-\s]?normali[sz]\w*\b"
    r"|\bnot\s+normali[sz]\w*\b", re.I)


def asks_normalised(text: str) -> bool:
    """Whether the question asks for the profiles to be normalised."""
    lowered = text or ""
    if RULES_OUT_NORMALISED.search(lowered):
        return False
    return bool(WANTS_NORMALISED.search(lowered))


#: "At what hour of the day did X have its highest average consumption" is a
#: question about the shape of the day, and the plan answered it with peak
#: demand -- the single largest reading, 195.4 kW, where the answer is hour
#: 11 at a mean of 123.6. A peak is one hour of one day; the highest average
#: hour is a property of the year.
ASKS_PEAK_HOUR = re.compile(
    r"\b(?:at\s+)?(?:what|which)\s+hour\b[^.?]{0,60}?"
    r"\b(?:highest|greatest|largest|maximum|peak|lowest|minimum)\b"
    r"|\b(?:highest|lowest)\s+average\s+hourly\b"
    r"|\bhour\s+of\s+(?:the\s+)?day\b[^.?]{0,40}?\b(?:highest|lowest|peak)\b",
    re.I)

#: A question that says "cluster" wants a clustering. Asked to cluster one
#: building's daily profiles the plan fetched the readings and drew a chart,
#: which is neither the count per cluster nor a refusal.
ASKS_CLUSTERING = re.compile(
    r"\bcluster(?:ing|s|ed)?\b|\bk[-\s]?means\b|\bsegment(?:ation)?\b"
    r"|\barchetypes?\b", re.I)


def resolve_season_length(text: str) -> int | None:
    """The seasonal period a question names, in hours, or None."""
    match = SEASON_LENGTH.search(text or "")
    if not match:
        return None
    hours, named, short, word = match.groups()
    if word:
        return 24 if word.lower() == "daily" else 168
    value = hours or named or short
    return int(value) if value else None


def resolve_horizon_hours(text: str, default: int = 24) -> int:
    """The forecast horizon in hours, bounded to something scoreable."""
    match = HORIZON_HOURS.search(text or "")
    if not match:
        # "for 2017-08-01", "on 2017-08-01" -- one named day is 24 hours.
        return default
    if match.group(5):                                  # "the 24 hours of ..."
        return max(1, min(int(match.group(5)), 24 * 90))
    count = int(match.group(1) or match.group(3))
    unit = (match.group(2) or match.group(4) or "hour").lower()
    return max(1, min(count * HORIZON_MULTIPLIER.get(unit, 1), 24 * 90))


def resolve_moving_window_days(text: str, default: int = 7) -> int:
    """The trailing window a moving-average question names, in days."""
    match = MOVING_WINDOW.search(text or "")
    if not match:
        return default
    return max(1, min(int(match.group(1) or match.group(2)), 365))


# --- anomaly questions -------------------------------------------------------
# The method words separate three genuinely different tests, and running the
# wrong one answers a different question: a multiple of the typical day, a
# standard-deviation distance from it, and a departure of the day's *shape*
# from the building's usual shape. The first two compare daily totals; only the
# third can see a day whose total is ordinary and whose profile is not.
ANOMALY_METHOD_PHRASES: tuple[tuple[str, str], ...] = (
    # The modified z-score is matched before the plain one, because its name
    # contains it and the coarser reading would claim both. It is a different
    # test: median and median absolute deviation rather than mean and standard
    # deviation, so the outliers being looked for do not move the statistics
    # they are measured against.
    (r"\bmodified\s+z[-\s]?scores?\b|\bmedian\s+absolute\s+deviation\b"
     r"|\bmad\b|\brobust\s+z[-\s]?scores?\b", "modified_zscore"),
    # A question about *change* rather than level.
    (r"\b(?:absolute\s+)?change\s+from\s+the\s+(?:preceding|previous|prior)"
     r"\s+(?:hour|reading|observation)\b|\bhour[-\s]to[-\s]hour\s+change\b"
     r"|\bstep\s+change\s+between\s+(?:hours|readings)\b"
     r"|\bfirst\s+difference\b|\bdelta\s+from\s+the\s+previous\b",
     "first_difference"),
    (r"\blocal\s+outlier\s+factor\b|\blof\b|\bdensity[-\s]based\b"
     r"|\bdistance[-\s]based\s+.{0,24}anomal\w*", "local_outlier_factor"),
    (r"\b(?:more|greater|larger)\s+than\s+\w+\s+times\b|\b(?:twice|double|triple)\b"
     r"|\b\d+(?:\.\d+)?\s*(?:x|times)\s+(?:its|their|the)\b"
     r"|\bmultiple\s+of\b|\bexceeds?\s+\w*\s*(?:twice|double)\b", "multiple_of_typical"),
    (r"\bstandard\s+deviations?\b|\bsigma\b|\bz[-\s]?scores?\b|\bsds?\s+(?:from|above)\b",
     "zscore"),
    (r"\bload\s+shape\b|\bprofile\s+(?:deviat\w+|shape)\b|\bshape\s+deviat\w+\b"
     r"|\bdaily\s+profile\b|\birregular\s+load\b", "daily_profile"),
    (r"\binterquartile\b|\biqr\b", "iqr"),
    (r"\bisolation\s+forest\b|\bunsupervised\s+anomaly\b", "isolation_forest"),
)

# "more than 2 standard deviations", "twice its typical", "3x the usual".
ANOMALY_THRESHOLD = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:standard\s+deviations?|sigma|sds?)\b"
    r"|\b(\d+(?:\.\d+)?)\s*(?:x|times)\s+(?:its|their|the)\b"
    r"|\b(twice|double)\b|\b(triple|thrice)\b", re.I)


def resolve_anomaly_method(text: str) -> tuple[str, bool]:
    """Which anomaly test the question asked for, and whether it said.

    A standard-deviation question is asked at two different grains and the
    tool has a method for each: `zscore` scores hourly readings and answers
    "which hours", `daily_profile` scores daily totals and answers "which
    days". The words are identical -- "more than two standard deviations from
    its typical" -- so the grain has to come from what the question asks for,
    and getting it wrong returns 40 hours where a reader expected a list of
    dates.
    """
    lowered = (text or "").lower()
    for pattern, method in ANOMALY_METHOD_PHRASES:
        if re.search(pattern, lowered):
            if method == "zscore" and re.search(
                    r"\bdays?\b|\bdaily\b|\bdates?\b", lowered):
                return "daily_profile", True
            return method, True
    return "daily_profile", False


def resolve_anomaly_threshold(text: str, method: str) -> float:
    """The line the question drew, in the units the chosen method uses.

    A multiple and a standard-deviation count are both bare numbers in the
    question and mean entirely different things, so the threshold is only read
    once the method is known: "twice its typical daily average" is 2.0 as a
    multiple and would be a nonsensical 2-sigma test.
    """
    match = ANOMALY_THRESHOLD.search(text or "")
    if match:
        sigma, multiple, twice, triple = match.groups()
        if method == "multiple_of_typical":
            if multiple:
                return float(multiple)
            if twice:
                return 2.0
            if triple:
                return 3.0
        elif sigma:
            return float(sigma)
    return {"multiple_of_typical": 2.0,
            # The conventional line for a MAD-based score (Iglewicz & Hoaglin).
            "modified_zscore": 3.5,
            # "The IQR method" means Tukey's fence, and Tukey's fence is 1.5 --
            # 3.0 is his *outer* fence, for extreme outliers only. The 3.0 here
            # was the z-score default falling through, and it is not a near
            # miss: on the London offices in 2017 it finds 40 outlier readings
            # on 13 days where the 1.5 fence finds 2,268 on 201.
            "iqr": 1.5,
            # LOF is read on its own scale, where 1 means "as dense as its
            # neighbours"; a 3-sigma default would flag nothing.
            "local_outlier_factor": 1.5}.get(method, 3.0)


# "Gaps exceeding one week", "a gap of at least 48 hours", "breaks longer than
# three days". The unit is part of the number, and the comparison word decides
# the edge: "exceeding" and "longer than" are strictly greater, "at least" and
# "or more" include the value.
_COUNT = r"\d+(?:\.\d+)?|" + "|".join(WORD_NUMBERS) + r"|a|an"
_EDGE = (r"exceeding|longer\s+than|more\s+than|over|greater\s+than"
         r"|at\s+least|of\s+at\s+least|or\s+more")
GAP_PHRASE = re.compile(
    rf"\b(?:gaps?|breaks?|outages?|interruptions?)\b[^.?]{{0,60}}?"
    rf"\b({_EDGE})\b\s*(?:({_COUNT})\s*)?(hours?|days?|weeks?|months?)\b"
    rf"|\b({_EDGE})\b\s*(?:({_COUNT})\s*)?(hours?|days?|weeks?|months?)\b"
    rf"[^.?]{{0,40}}?\b(?:gaps?|breaks?|outages?)\b", re.I)

GAP_UNIT_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 24 * 30}


def gap_bound(text: str) -> tuple[int, str] | None:
    """A stated gap length in hours, and whether the edge includes it.

    "One week" carries no digit, so a bare unit means one of it: reading only
    the number would miss the phrasing every question in the list actually
    uses.
    """
    match = GAP_PHRASE.search(text or "")
    if not match:
        return None
    groups = [g for g in match.groups() if g]
    comparison = next((g.lower() for g in groups
                       if re.fullmatch(r"exceeding|longer\s+than|more\s+than"
                                       r"|over|greater\s+than|at\s+least"
                                       r"|of\s+at\s+least|or\s+more", g, re.I)),
                      "at least")
    unit = next((g.lower().rstrip("s") for g in groups
                 if g.lower().rstrip("s") in GAP_UNIT_HOURS), None)
    if unit is None:
        return None
    # "One week" and "a week" carry no digit and mean one of it.
    count = next((float(g) if re.fullmatch(r"\d+(?:\.\d+)?", g)
                  else float(WORD_NUMBERS[g.lower()])
                  for g in groups
                  if re.fullmatch(r"\d+(?:\.\d+)?", g)
                  or g.lower() in WORD_NUMBERS), 1.0)
    hours = int(round(count * GAP_UNIT_HOURS[unit]))
    strict = bool(re.fullmatch(r"exceeding|longer\s+than|more\s+than|over"
                               r"|greater\s+than", comparison, re.I))
    return hours, ("exceeding" if strict else "at_least")


def resolve_day_type(text: str) -> str:
    """Whether the question is about weekends, weekdays, or every day.

    A scan restricted this way restricts its baseline too: "unusual weekend
    consumption" is a weekend that is unusual among this building's other
    weekends, not one that differs from its weekdays -- which every weekend
    does. A "weekday morning ramp" is not a restriction of this kind, so the
    phrase has to be about the days being examined rather than about a time
    within them; the guard is that no clock word sits beside it.
    """
    lowered = (text or "").lower()
    if re.search(r"\bweekends?\b|\bsaturdays?\b|\bsundays?\b", lowered) \
            and not re.search(r"\bweekdays?\b", lowered):
        return "weekend"
    if re.search(r"\bweekdays?\b|\bworking\s+days?\b|\bbusiness\s+days?\b",
                 lowered) and not re.search(r"\bweekends?\b", lowered):
        return "weekday"
    return "all"


# Which clustering algorithm the question names. Hierarchical linkages are
# listed before the generic word so "average linkage" is not read as
# "hierarchical" and then silently run as Ward.
CLUSTER_METHOD_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bward(?:'s)?\s*(?:linkage|method|criterion)?\b", "ward"),
    (r"\baverage\s+linkage\b|\bupgma\b", "average"),
    (r"\bcomplete\s+linkage\b|\bfurthest\s+neighbou?r\b", "complete"),
    (r"\bsingle\s+linkage\b|\bnearest\s+neighbou?r\s+linkage\b", "single"),
    (r"\bhierarchical\b|\bagglomerative\b|\bdendrogram\b|\blinkage\b",
     "hierarchical"),
    # "TimeSeriesKMeans" is one word and k-means by another name; without it
    # the question that names it read as naming no method at all.
    (r"\btime\s?series\s?k[-\s]?means\b|\bts[-\s]?kmeans\b", "kmeans"),
    (r"\bk[-\s]?means\b|\bcentroid\b", "kmeans"),
)


#: "on which day", "which date" -- the calendar date, not the day of the week.
#: "day_of_week" answers Thursday where the question wants 22 June.
CALENDAR_DAY_RE = re.compile(
    r"\bon\s+(?:which|what)\s+(?:day|date)\b|\bwhich\s+(?:day|date)\s+(?:did|had|was)\b"
    r"|\bwhat\s+date\b|\bwhich\s+calendar\s+day\b", re.I)


def asks_calendar_day(text: str) -> bool:
    """Does the question want a dated day rather than a weekday?"""
    lowered = text or ""
    if not CALENDAR_DAY_RE.search(lowered):
        return False
    # "On which day of the week" is the other grouping and says so.
    return not re.search(r"\bday\s+of\s+the\s+week\b|\bweekday\b", lowered, re.I)


#: "the next 7 days of daily consumption ... the previous 30 days of daily
#: data" -- a daily model, not an hourly one summed afterwards.
DAILY_GRAIN_RE = re.compile(
    r"\bdaily\s+(?:electricity\s+)?consumption\b[^.?]{0,80}\bdaily\s+data\b"
    r"|\bof\s+daily\s+data\b|\bdaily\s+series\b|\bdaily\s+grain\b", re.I)


def asks_daily_grain(text: str) -> bool:
    """Is the forecast fitted on daily values rather than hourly ones?"""
    return bool(DAILY_GRAIN_RE.search(text or ""))


def resolve_cluster_methods(text: str) -> list[str]:
    """Every clustering algorithm the question names, in order.

    "Compare K-Means and Hierarchical Clustering" names two and the single
    resolver returns whichever pattern matched first, so the comparison came
    back as one method against nothing.
    """
    lowered = (text or "").lower()
    found: list[tuple[int, str]] = []
    for pattern, method in CLUSTER_METHOD_PHRASES:
        match = re.search(pattern, lowered)
        if match and method not in {m for _, m in found}:
            found.append((match.start(), method))
    return [m for _, m in sorted(found)]

CLUSTER_DISTANCE_PHRASES: tuple[tuple[str, str], ...] = (
    # Dynamic time warping, which the tool supports and this list did not
    # name: a question asking for DTW was clustered under Euclidean, which
    # is the one distance DTW exists to avoid.
    (r"\bdtw\b|\bdynamic\s+time\s+warp\w*\b", "dtw"),
    (r"\bcosine\b|\bangular\b", "cosine"),
    (r"\bmanhattan\b|\bcityblock\b|\bl1\b|\btaxicab\b", "manhattan"),
    (r"\beuclidean\b|\bl2\b", "euclidean"),
)

# "Report the silhouette score", "compare Davies-Bouldin across k". All three
# are always computed; naming one says which to lead with.
CLUSTER_SCORE_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bsilhouette\b", "silhouette"),
    (r"\bdavies[-\s]?bouldin\b|\bdb\s+index\b", "davies_bouldin"),
    (r"\bcalinski[-\s]?harabasz\b|\bvariance[-\s]?ratio\b", "calinski_harabasz"),
)


#: Group dimensions a similarity question can be asked about. "Which building
#: types are most similar" is a different question from "which buildings are
#: most like this one", and only the second needs a reference building.
SIMILARITY_GROUPS = (
    (r"\bbuilding\s+subtypes?\b|\bsubtypes?\b", "building_subtype"),
    (r"\bbuilding\s+(?:types?|categor(?:y|ies)|uses?)\b"
     r"|\bspace\s+usages?\b|\btypes?\s+of\s+buildings?\b", "building_type"),
    (r"\bsubindustr(?:y|ies)\b", "subindustry"),
    (r"\bindustr(?:y|ies)\b", "industry"),
    (r"\bcities\b|\bcitys?\b", "city"),
    (r"\bsites?\b(?!\s+\w*\s*(?:id|name))", "site"),
)


def resolve_similarity_group(text: str) -> str | None:
    """The group a similarity question compares, if it compares groups at all."""
    lowered = (text or "").lower()
    # A named reference building means the question is building-to-buildings,
    # whatever nouns surround it.
    if re.search(r"\b[A-Z][a-z]+_[a-z]+_[A-Z][a-z]+\b", text or ""):
        return None
    for pattern, group in SIMILARITY_GROUPS:
        if re.search(pattern, lowered):
            return group
    return None


#: "Which two cities in DIFFERENT COUNTRIES have the most similar profiles" is
#: not the same question as "which two cities". The most similar pair is very
#: often two cities in one country -- they share a climate and a calendar --
#: so a plan that drops the constraint answers with the pair the question
#: explicitly rules out, and reads as though it had honoured it.
REQUIRE_DIFFERENT_RE = (
    (r"\b(?:in|from|across|of|spanning)\s+(?:two\s+|different|separate|"
     r"distinct|other)\s*(?:different\s+)?countries\b"
     r"|\bdifferent\s+countries\b|\bnot\s+in\s+the\s+same\s+country\b", "country"),
    (r"\bdifferent\s+cities\b|\bnot\s+in\s+the\s+same\s+city\b", "city"),
    (r"\bdifferent\s+sites\b|\bnot\s+(?:at|on)\s+the\s+same\s+site\b", "site"),
)


#: Geography, coarsest first. A constraint is only meaningful on a group that
#: sits inside it: "two cities in different countries" narrows, "two building
#: types in different countries" asks for something no pair can satisfy and
#: would turn a real question into an empty result.
GEOGRAPHIC_RANK = {"country": 3, "city": 2, "site": 1}


def resolve_require_different(text: str, group_by: str | None = None) -> str:
    """The attribute a question demands the two groups differ on.

    Returns "none" where it demands nothing, which is the tool's default and
    lets any pair win, and also where the groups being compared are not
    geographic -- the constraint has nothing to bite on there.
    """
    lowered = (text or "").lower()
    for pattern, attribute in REQUIRE_DIFFERENT_RE:
        if re.search(pattern, lowered):
            if group_by is None:
                return attribute
            if GEOGRAPHIC_RANK.get(attribute, 0) >= GEOGRAPHIC_RANK.get(group_by, 99):
                return attribute
            return "none"
    return "none"


#: "What share of consumption falls in the top 5% of hours" names a measure the
#: catalogue computes directly. Without this it was read as a weather question
#: -- the plan ended on mean air temperature during those hours, which is a
#: real quantity and not the one asked for.
TOP_SHARE_RE = re.compile(
    r"\b(?:proportion|share|fraction|percentage|percent|how\s+much)\b"
    r"[^.?]{0,80}?\btop\s*(\d{1,2})\s*%", re.I)


def resolve_top_hours_share(text: str) -> float | None:
    """The N in "the top N% of hours", when consumption share is asked for."""
    match = TOP_SHARE_RE.search(text or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if 0 < value < 100 else None


#: Two or more named similarity measures means the question is ABOUT the
#: measures. Clustering under one of them and reporting a silhouette answers
#: a different question with a real number, which is the hardest kind of wrong
#: answer to notice.
MEASURE_WORDS = {
    "pearson": r"\bpearson\b|\bcorrelation\b",
    "euclidean": r"\beuclid\w*",
    "cosine": r"\bcosine\b",
}


def resolve_similarity_measures(text: str) -> list[str]:
    """The distance measures a question asks to compare, if it compares any."""
    lowered = (text or "").lower()
    found = [name for name, pattern in MEASURE_WORDS.items()
             if re.search(pattern, lowered)]
    # One measure is a choice of method; two or more is a comparison.
    return found if len(found) >= 2 else []


def resolve_cluster_method(text: str) -> tuple[str, bool]:
    """The clustering algorithm the question named, and whether it named one."""
    lowered = (text or "").lower()
    for pattern, method in CLUSTER_METHOD_PHRASES:
        if re.search(pattern, lowered):
            return method, True
    return "kmeans", False


def resolve_cluster_distance(text: str) -> str:
    """The distance clusters are formed under; Euclidean unless said."""
    lowered = (text or "").lower()
    for pattern, metric in CLUSTER_DISTANCE_PHRASES:
        if re.search(pattern, lowered):
            return metric
    return "euclidean"


def resolve_cluster_scores(text: str) -> list[str]:
    """Which validity score the question leads with, if it names one."""
    lowered = (text or "").lower()
    return [name for pattern, name in CLUSTER_SCORE_PHRASES
            if re.search(pattern, lowered)]


#: "How many clusters are there" asks the data, not the caller. Answering it
#: with the default k and reporting that number back is circular -- the reply
#: is the argument, dressed as a finding.
ASKS_HOW_MANY_CLUSTERS = re.compile(
    r"\bhow\s+many\s+(?:[-\w]+\s+){0,5}?"
    r"(?:clusters?|groups?|archetypes?|segments?)\b"
    r"|\b(?:optimal|best|ideal|right|appropriate)\s+(?:number\s+of\s+)?"
    r"(?:clusters?|k)\b"
    r"|\bnumber\s+of\s+(?:distinct\s+)?(?:clusters?|groups?)\b", re.I)


def resolve_n_clusters(text: str, default: int = 4) -> int | str | None:
    """How many groups the question asked for, or None to let the data decide.

    "Four archetypes", "into 3 groups", "k=5". A bare number elsewhere in the
    sentence is not a cluster count, so the number has to sit beside a word
    that means one. When the question *asks* how many there are and names no
    number, this returns ``None`` and the tool picks k by silhouette -- which
    is the only way the count can be an answer rather than an echo.
    """
    lowered = (text or "").lower()
    words = "|".join(WORD_NUMBERS)
    match = re.search(
        rf"\b(\d{{1,2}}|{words})\s+(?:distinct\s+|representative\s+|"
        rf"different\s+)?(?:clusters?|groups?|archetypes?|segments?|"
        rf"categor(?:y|ies)|classes)\b"
        rf"|\b(?:into|in\s+to)\s+(\d{{1,2}}|{words})\b"
        rf"|\bk\s*=\s*(\d{{1,2}})\b", lowered)
    if not match:
        return "auto" if ASKS_HOW_MANY_CLUSTERS.search(lowered) else default
    token = next((g for g in match.groups() if g), None)
    if token is None:
        return "auto" if ASKS_HOW_MANY_CLUSTERS.search(lowered) else default
    count = int(token) if token.isdigit() else WORD_NUMBERS.get(token, default)
    return max(2, min(count, 20))


def asks_for_change_point(text: str) -> bool:
    """A question about a persistent shift in level, not a one-off day."""
    lowered = (text or "").lower()
    # "Shifts over 96 hours" is a level shift with its window stated, and it
    # matched none of these spellings -- so a question about sustained change
    # was answered with a daily-outlier test, which finds a single unusual day
    # and cannot see a step at all. A bare "shift" is admitted only when it is
    # about consumption over a span, so "shift work" and "shifted to gas" do
    # not drag ordinary questions into change-point detection.
    return bool(re.search(
        r"\bstructural\s+(?:change|break)\b|\bstep[-\s]?change\b|\blevel\s+shift\b"
        r"|\bregime\s+(?:change|shift)\b|\bbreakpoint\b|\bchange[-\s]?points?\b"
        r"|\bbaseline\s+\w*\s*(?:has\s+)?(?:under)?gone\b"
        r"|\bshifts?\s+(?:of|over|across|spanning)\s+\d+\s*(?:h|hour|hours|day|days)\b"
        r"|\bsustained\s+(?:increase|decrease|change|shift)\b", lowered))


class DeterministicPlanner:
    """Rule-based planning. No model, fully reproducible, the floor baseline."""

    name = "deterministic"

    COMPARABLE = {"demand": "load_factor", "eui": "eui", "rank": "eui",
                  "consumption": "total_kwh", "per_occupant": "per_occupant"}

    # The routes written for a known question shape, in the order they are
    # tried. Named, because which one answered is worth recording: a hybrid
    # planner asks whether any of them matched before handing the question to
    # a model, and a reader of the results wants to know which did.
    def _routes(self):
        return (("explain_consumption", self._explain_consumption),
                ("compare_building_types", self._compare_building_types),
                ("size_band", self._size_band_comparison),
                ("metadata_completeness", self._metadata_completeness),
                ("performance_index", self._performance_index),
                ("compare_vintages", self._compare_vintages),
                ("normalise_by_operating_hours", self._normalise_by_operating_hours),
                ("adjusted_ranking", self._adjusted_ranking),
                ("compare_named_days", self._compare_named_days),
                ("compare_semesters", self._compare_semesters),
                ("peak_hour_of_profile", self._peak_hour_of_profile),
                ("calendar_period", self._pick_calendar_period),
                ("size_deciles", self._compare_size_deciles),
                ("two_cohorts", self._compare_two_cohorts))

    def plan(self, intent: Intent, known_facts=None) -> Plan:
        """Plan the question, then mirror the branch for any second cohort."""
        for name, route in self._routes():
            compared = route(intent)
            if compared is not None:
                compared.route = name
                return compared
        built = self._plan(intent, known_facts)
        built.route = (f"tag:{intent.metric}" if intent.recognised_shape
                       else "unrecognised")
        return self._mirror_cohorts(built, intent)

    DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday")

    def _explain_consumption(self, intent: Intent) -> Plan | None:
        """"What explains consumption" is a fit, not a ranking.

        "Which combination of building characteristics and weather variables
        provides the strongest explanation" asks for coefficients and the
        error left over. Answering it with a ranked table of consumption says
        what the differences *are* and nothing about what accounts for them.
        """
        asked = (intent.question or "").lower()
        if not asks_explanation(asked):
            return None
        # ...but only where the question names things to explain *with*. "How
        # does EUI vary across building types" is a grouping, which the
        # ordinary routes answer better.
        wants = [name for name, pattern in (
            ("floor_area_m2", r"\bfloor\s*area\b|\bsize\b|\bsqm\b|\bm2\b"),
            ("year_built", r"\byear\s*built\b|\bvintage\b|\bage\b"),
            ("occupants", r"\boccupan\w*\b"),
            ("number_of_floors", r"\bfloors?\b|\bstoreys?\b"),
            ("building_type", r"\bbuilding\s*types?\b|\bcharacteristics?\b"
                              r"|\busage\b|\bcategor\w+"),
            ("mean_temperature", r"\btemperature\b|\bweather\b"),
            ("heating_degree_days", r"\bheating\b|\bweather\b|\bdegree\s*days?\b"),
            ("cooling_degree_days", r"\bcooling\b|\bweather\b|\bdegree\s*days?\b"),
        ) if re.search(pattern, asked)]
        if len(wants) < 2:
            return None
        entities = intent.entities or {}
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "find_buildings",
            {"country": entities.get("country"), "city": entities.get("city"),
             "site_id": entities.get("site_id"),
             "building_type": entities.get("building_type"),
             "limit": 5000},
            resolution="graph_only", purpose="the cohort to explain"))
        plan.add(PlanStep(
            "s2", "explain_consumption",
            {"building_ids": "$s1.column.building_id",
             "target": resolve_measure(intent.question, default="eui"),
             "features": wants,
             "year": (intent.time_range or {}).get("year")},
            depends_on=["s1"],
            purpose="coefficients and the error they leave"))
        plan.answer_step = "s2"
        return plan.fill_agents()

    def _compare_building_types(self, intent: Intent) -> Plan | None:
        """Two building types named in one comparison is a cohort comparison.

        "Compare the load factor for education against office buildings" is
        answered by the two distributions and the ratio between them. Resolving
        both cohorts and stopping -- which is what the generic branch did --
        answers with a list of 307 offices.
        """
        entities = intent.entities or {}
        types = entities.get("building_type_all") or []
        asked = (intent.question or "").lower()
        if len(types) != 2 or not re.search(
                r"\bcompare[ds]?\b|\bversus\b|\bvs\.?\b|\bagainst\b"
                r"|\bbetween\b", asked):
            return None
        metric = resolve_measure(intent.question, default="eui")
        requires = ["occupancy_normalisation" if metric == "per_occupant"
                    else "eui_analysis" if metric in ("eui", "peak_per_area")
                    else "consumption_analysis"]
        plan = Plan(question=intent.question)
        for step, building_type in zip(("s1", "s2"), types):
            plan.add(PlanStep(step, "find_buildings",
                              {"building_type": building_type,
                               "country": entities.get("country"),
                               "requires": requires, "limit": 5000},
                              resolution="graph_only",
                              purpose=f"the {building_type} cohort"))
        plan.add(PlanStep(
            "s3", "compare_building_groups",
            {"group_a": "$s1.column.building_id",
             "group_b": "$s2.column.building_id", "metric": metric,
             "label_a": types[0], "label_b": types[1]},
            depends_on=["s1", "s2"],
            purpose=f"{types[0]} against {types[1]} on {metric}, with the "
                    f"ratio between their medians"))
        plan.answer_step = "s3"
        return plan.fill_agents()

    def _size_band_comparison(self, intent: Intent) -> Plan | None:
        """"Compared to others of similar floor area" is a band, not a cohort.

        A 500 m2 school and a 50,000 m2 one are not peers, and ranking them
        together answers about size as much as about intensity. Each building
        is placed among the others within a quarter of its own floor area.
        """
        asked = (intent.question or "").lower()
        if not re.search(r"\b(?:similar|comparable|same|like)\s+(?:sized?\s+)?"
                         r"(?:floor\s*area|size|buildings?|others?|peers?)\b"
                         r"|\bothers?\s+of\s+(?:similar|comparable)\b"
                         r"|\bfor\s+(?:their|its)\s+size\b", asked):
            return None
        share = re.search(r"top[\s-]?(\d+(?:\.\d+)?)\s*%", asked)
        bottom = re.search(r"bottom[\s-]?(\d+(?:\.\d+)?)\s*%", asked)
        entities = intent.entities or {}
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "find_buildings",
            {"building_type": entities.get("building_type"),
             "country": entities.get("country"), "city": entities.get("city"),
             "requires": ["eui_analysis"], "limit": 5000},
            resolution="graph_only", purpose="the cohort to band by size"))
        plan.add(PlanStep(
            "s2", "identify_peer_outliers",
            {"building_ids": "$s1.column.building_id",
             "metric": resolve_measure(intent.question),
             "method": "percentile", "peers": "size_band",
             "threshold": float((share or bottom).group(1)) if (share or bottom)
                          else 10.0,
             "side": "low" if bottom else "high"},
            depends_on=["s1"],
            purpose="each building against those of roughly its own size"))
        plan.answer_step = "s2"
        return plan.fill_agents()

    def _metadata_completeness(self, intent: Intent) -> Plan | None:
        """Missing metadata is measured, not listed.

        "A matrix of missing metadata attributes across all global regions" is
        one call: how many buildings carry each attribute, one row per region.
        Listing the buildings and their columns leaves the reader to do the
        counting, and answers a question about coverage with a catalog dump.
        """
        asked = (intent.question or "").lower()
        if not re.search(r"\bmetadata\b|\bmeta[\s-]data\b", asked):
            return None
        # The completeness word has to be *about* the metadata, which in
        # practice means next to it. "Using the dataset metadata to first
        # identify a building with at least 3 years of continuous consumption
        # history, detect whether ... across its available history" was claimed
        # by this route on the strength of an "available" sixty characters
        # away that described the measurements: the metadata is how the
        # building is chosen, not what the question asks about.
        near = re.search(r"\bmeta[\s-]?data\b", asked)
        window = asked[max(0, near.start() - 40):near.end() + 40] if near else asked
        if not re.search(r"\bcomplete\w*|\bmissing\b|\bavailab\w*|\bcoverage\b"
                         r"|\bmatrix\b|\bfitness\b", window):
            return None
        group = {"sites": "site", "cities": "site", "countries": "country",
                 "building_types": "building_type"}.get(intent.group_by or "", "none")
        fields = [name for name, pattern in
                  (("floor_area", r"\bfloor\s*area\b|\bsqm\b|\bsquare\s*met"),
                   ("year_built", r"\byear\s*built\b|\bvintage\b|\bage\b"),
                   ("occupants", r"\boccupan\w*\b"),
                   ("number_of_floors", r"\bfloors?\s*count\b|\bstoreys?\b"),
                   ("building_type", r"\bbuilding\s*types?\b|\busage\b"))
                  if re.search(pattern, asked)]
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "profile_metadata_completeness",
            {"fields": fields or None, "group_by": group,
             "building_type": (intent.entities or {}).get("building_type"),
             "country": (intent.entities or {}).get("country")},
            resolution="graph_only",
            purpose=("how complete the metadata is"
                     + (f", one row per {group}" if group != "none" else ""))))
        plan.answer_step = "s1"
        return plan.fill_agents()

    def _performance_index(self, intent: Intent) -> Plan | None:
        """An index is several measures combined, not one measure ranked.

        The components come from the question, the weights are equal and
        stated, and each component is a z-score across the cohort being
        ranked. Ranking on any one of them answers a different question.
        """
        components = resolve_index_components(intent.question)
        if not components:
            return None
        entities = intent.entities or {}
        asked = (intent.question or "").lower()
        plan = Plan(question=intent.question)
        explicit = entities.get("building_ids")
        if explicit:
            ids: object = explicit
            depends: list[str] = []
        else:
            plan.add(PlanStep(
                "s1", "find_buildings",
                {"country": entities.get("country"), "city": entities.get("city"),
                 "site_id": entities.get("site_id"),
                 "building_type": entities.get("building_type"),
                 "requires": ["eui_analysis"], "limit": 5000},
                resolution="graph_only",
                purpose="the cohort the index is standardised across"))
            ids = "$s1.column.building_id"
            depends = ["s1"]
        step = "s2" if depends else "s1"
        plan.add(PlanStep(
            step, "construct_performance_index",
            {"building_ids": ids, "components": components,
             # An index of efficiency is read from its worst end unless the
             # question asks for the best: the buildings worth looking at are
             # the ones the index cannot explain away.
             "order": "asc" if re.search(r"\bmost\s+efficient\b|\bbest\b"
                                         r"|\blowest\b", asked) else "desc",
             "limit": entities.get("limit", 20)},
            depends_on=depends,
            purpose="z-score each component across the cohort, average them "
                    "with equal stated weights, and rank"))
        plan.answer_step = step
        return plan.fill_agents()

    def _compare_vintages(self, intent: Intent) -> Plan | None:
        """Two building ages named in one question is a cohort comparison.

        The vintages resolve from the catalog, one find_buildings each, and
        the two sets go to compare_building_groups, which reports each
        distribution and the ratio between their medians. A year in this
        question is an age, not a period: filtering the measurements to 1980
        selects nothing the snapshot holds.
        """
        vintages = resolve_vintages(intent.question)
        if len(vintages) != 2:
            return None
        entities = intent.entities or {}
        metric = resolve_measure(intent.question)
        plan = Plan(question=intent.question)
        # The capability the measure needs, asked of the catalog once, so the
        # two cohorts are screened by the same rule before either is measured.
        requires = ["occupancy_normalisation" if metric == "per_occupant"
                    else "eui_analysis"]
        for step, (label, parameter, value) in zip(("s1", "s2"), vintages):
            plan.add(PlanStep(
                step, "find_buildings",
                {parameter: value, "country": entities.get("country"),
                 "city": entities.get("city"),
                 "building_type": entities.get("building_type"),
                 "requires": requires, "limit": 5000},
                resolution="graph_only", purpose=f"the {label} cohort"))
        # The year and the statistic the question names travel with the step
        # that answers it. compare_building_groups defaults an intensity to
        # the latest full year, which is right for most of these questions and
        # silently wrong for one about 2016.
        compare = {"group_a": "$s1.column.building_id",
                   "group_b": "$s2.column.building_id", "metric": metric,
                   "label_a": vintages[0][0], "label_b": vintages[1][0]}
        asked_year = _calendar_year((intent.time_range or {}).get("start"),
                                    (intent.time_range or {}).get("end"))
        if asked_year:
            compare["year"] = int(asked_year)
        statistic = resolve_named_statistic(intent.question)
        if statistic in ("median", "mean", "total"):
            compare["statistic"] = statistic
        plan.add(PlanStep(
            "s3", "compare_building_groups", compare,
            depends_on=["s1", "s2"],
            purpose=f"{vintages[0][0]} against {vintages[1][0]} on {metric}, "
                    f"with the ratio between their medians"))
        plan.answer_step = "s3"
        return plan.fill_agents()

    def _normalise_by_operating_hours(self, intent: Intent) -> Plan | None:
        """"EUI normalised by inferred operating hours" is its own measure.

        The hours are inferred from each building's own daily profile and are
        the weak link in the number, so they are reported beside it. Asked of
        a portfolio the answer is the distribution and its median, not a top
        ten: nothing in the question selects.
        """
        if resolve_measure(intent.question, default="") != "eui_per_operating_hour":
            return None
        asked = (intent.question or "").lower()
        entities = intent.entities or {}
        plan = Plan(question=intent.question)
        explicit = entities.get("building_ids")
        # A named cohort ("office buildings in Bristol") is resolved from the
        # catalog first, exactly as every other measure route resolves it: a
        # route that reads the measure and forgets the filter answers over the
        # whole portfolio.
        if explicit:
            ids: object = explicit
            depends: list[str] = []
        else:
            plan.add(PlanStep(
                "s1", "find_buildings",
                {"country": entities.get("country"), "city": entities.get("city"),
                 "site_id": entities.get("site_id"),
                 "building_type": entities.get("building_type"),
                 "requires": ["eui_analysis"], "limit": 5000},
                resolution="graph_only",
                purpose="resolve the eligible set before touching measurements"))
            ids = "$s1.column.building_id"
            depends = ["s1"]
        step = "s2" if depends else "s1"
        if re.search(r"\bhighest\b|\blowest\b|\btop\b|\bbottom\b|\bworst\b"
                     r"|\bbest\b|\bmost\b|\bleast\b|\brank\b", asked):
            plan.add(PlanStep(
                step, "rank_buildings",
                {"building_ids": ids, "metric": "eui_per_operating_hour",
                 "limit": entities.get("limit", 10),
                 "order": "asc" if re.search(r"\blowest|\bleast|\bbest\b", asked)
                          else "desc"},
                depends_on=depends,
                purpose="intensity per hour of use, ordered"))
        else:
            plan.add(PlanStep(
                step, "calculate_normalized_performance",
                {"building_ids": ids, "metric": "eui_per_operating_hour"},
                depends_on=depends,
                purpose="EUI divided by the hours a day each building is "
                        "apparently in use, over the whole cohort"))
        plan.answer_step = step
        return plan.fill_agents()

    def _adjusted_ranking(self, intent: Intent) -> Plan | None:
        """"Load factor normalised against age and floor area" is a residual.

        Dividing by either characteristic answers a different question, and
        aggregating the raw measure per site answers none of it: the point is
        what is left once age and size have been accounted for.
        """
        adjust = resolve_adjustments(intent.question)
        if len(adjust) < 2:
            return None
        asked = (intent.question or "").lower()
        across = next((dimension for dimension, pattern in
                       (("site", r"\bsites?\b"), ("city", r"\bcit(?:y|ies)\b"),
                        ("country", r"\bcountr(?:y|ies)\b"),
                        ("building_type", r"\bbuilding\s+types?\b"))
                       if re.search(pattern, asked)), "building")
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "calculate_normalized_performance",
            {"building_ids": (intent.entities or {}).get("building_ids"),
             "metric": resolve_measure(intent.question, default="load_factor"),
             "adjust_for": adjust, "across": across},
            purpose=f"what the measure leaves once {' and '.join(adjust)} "
                    f"are accounted for, per {across}"))
        plan.answer_step = "s1"
        return plan.fill_agents()

    def _compare_named_days(self, intent: Intent) -> Plan | None:
        """Two days of the week named in one question is a day-pair comparison."""
        asked = (intent.question or "").lower()
        named = [d for d in self.DAY_NAMES if re.search(rf"\b{d}\b", asked)]
        # Keep the order the question used, so the ratio reads the way it was
        # asked rather than alphabetically.
        named = sorted(set(named), key=asked.index)
        if len(named) != 2:
            return None
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "compare_temporal_profiles",
            {"building_ids": (intent.entities or {}).get("building_ids"),
             "comparison": "named_days",
             "day_a": named[0], "day_b": named[1]},
            purpose=f"{named[0]} against {named[1]}, with the ratio between them"))
        plan.answer_step = "s1"
        return plan.fill_agents()

    def _compare_semesters(self, intent: Intent) -> Plan | None:
        """Two semesters named in one question is a term-pair comparison.

        The cohort is resolved first because a semester question is normally
        about a building type -- "education buildings" -- rather than about
        one named building.
        """
        named = resolve_semesters(intent.question)
        if len(named) != 2:
            return None
        asked = (intent.question or "").lower()
        measure = ("peak_hour" if re.search(r"\bpeak\s+(?:timing|hour|time)\b"
                                            r"|\bwhen\b.*\bpeaks?\b", asked)
                   else "mean_demand")
        (a_name, a_months), (b_name, b_months) = named
        plan = Plan(question=intent.question)
        cohort = (intent.entities or {}).get("building_ids")
        depends: list[str] = []
        if not cohort:
            plan.add(PlanStep("s1", "find_buildings",
                              {"building_type": (intent.entities or {}).get(
                                  "building_type"), "limit": 5000},
                              purpose="the cohort the semesters are compared over"))
            cohort, depends = "$s1.column.building_id", ["s1"]
        plan.add(PlanStep(
            "s2", "compare_temporal_profiles",
            {"building_ids": cohort, "comparison": "named_months",
             "measure": measure, "months_a": a_months, "months_b": b_months,
             "label_a": a_name, "label_b": b_name, "across": "cohort"},
            depends_on=depends,
            purpose=f"{a_name} semester against {b_name}, as one figure "
                    f"for the cohort"))
        plan.answer_step = "s2"
        return plan.fill_agents()

    def _peak_hour_of_profile(self, intent: Intent) -> Plan | None:
        """"At what hour does it typically peak" is the profile's top hour."""
        asked = (intent.question or "").lower()
        if not re.search(r"\b(?:at\s+)?what\s+(?:local\s+)?hour\b"
                         r"|\bwhat\s+time\s+of\s+day\b", asked):
            return None
        if not re.search(r"\bpeak\w*|\bhighest|\bmaximum", asked):
            return None
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "calculate_load_profile",
            {"building_ids": (intent.entities or {}).get("building_ids"),
             "profile": "hourly", "peak_only": True},
            purpose="the hour of the highest mean in the daily profile"))
        plan.answer_step = "s1"
        return plan.fill_agents()

    def _pick_calendar_period(self, intent: Intent) -> Plan | None:
        """"Which month had the highest use" is a calendar grouping.

        The word "highest" otherwise makes it a ranking of buildings, where
        the question names one building and asks which of its months stands
        out.
        """
        asked = (intent.question or "").lower()
        dimension = ("month" if re.search(r"\bwhich\s+month\b|\bwhat\s+month\b",
                                          asked)
                     else "hour_of_day" if re.search(
                         r"\bwhich\s+hour\b|\bwhat\s+hour\b", asked)
                     else "day_of_week" if re.search(
                         r"\bwhich\s+day\s+of\s+(?:the\s+)?week\b", asked)
                     else None)
        if not dimension or not re.search(
                r"\bhighest|\blowest|\bmost\b|\bleast\b|\bpeak\b|\bmaximum|"
                r"\bminimum", asked):
            return None
        period = intent.time_range or {}
        ids = (intent.entities or {}).get("building_ids")
        plan = Plan(question=intent.question)
        plan.add(PlanStep(
            "s1", "calculate_consumption_statistics",
            {"building_ids": ids, "start": period.get("start"),
             "end": period.get("end"), "metrics": ["total"],
             "across": dimension, "order_by": "total",
             "order": "asc" if re.search(r"\blowest|\bleast\b", asked) else "desc",
             "limit": 1},
            purpose=f"the {dimension.replace('_', ' ')} at the top of the total"))
        plan.answer_step = "s1"
        return plan.fill_agents()

    def _compare_size_deciles(self, intent: Intent) -> Plan | None:
        """"Top 10% largest against the bottom 50%" is two decile cohorts.

        Read as a top-ten ranking it returns ten buildings, which answers
        neither half of it.
        """
        asked = (intent.question or "").lower()
        top = re.search(r"\b(?:top|largest|biggest|highest)\s+(\d+)\s*%", asked)
        bottom = re.search(r"\b(?:bottom|smallest|lowest)\s+(\d+)\s*%", asked)
        if not (top and bottom) or not re.search(
                r"\bsize|\blargest|\bsmallest|\bsqm\b|\bsq\s*m\b|\bfloor\s+area",
                asked):
            return None
        # A percentage of the portfolio is that many tenths of it, rounded to
        # the decile the phrase names.
        top_decile = max(1, 11 - max(1, round(int(top.group(1)) / 10)))
        bottom_decile = max(1, round(int(bottom.group(1)) / 10))
        plan = Plan(question=intent.question)
        plan.add(PlanStep("s1", "find_buildings",
                          {"floor_area_decile_at_least": top_decile,
                           "limit": 5000}, resolution="graph_only",
                          purpose=f"the largest {top.group(1)}% by floor area"))
        plan.add(PlanStep("s2", "find_buildings",
                          {"floor_area_decile_at_most": bottom_decile,
                           "limit": 5000}, resolution="graph_only",
                          purpose=f"the smallest {bottom.group(1)}% by floor area"))
        plan.add(PlanStep(
            "s3", "compare_building_groups",
            {"group_a": "$s1.column.building_id",
             "group_b": "$s2.column.building_id",
             "metric": "total_kwh",
             "label_a": f"largest {top.group(1)}% by floor area",
             "label_b": f"smallest {bottom.group(1)}% by floor area",
             "min_coverage": 0.0,
             # total_kwh is not an annual measure, so nothing supplies the
             # year for it: without this the "annual" total is the sum over
             # the whole eighteen-month record. Where the question says
             # "annual" and names no year, the house reading of "annual" is
             # the latest full year, which is what the router prompt tells a
             # model to pass and what this has to agree with.
             "year": (_calendar_year((intent.time_range or {}).get("start"),
                                     (intent.time_range or {}).get("end"))
                      or (LATEST_FULL_YEAR
                          if re.search(r"\bannual\w*\b|\bper\s+year\b", asked)
                          else None))},
            depends_on=["s1", "s2"],
            purpose="each group's total and the ratio between them"))
        plan.answer_step = "s3"
        return plan.fill_agents()

    def _compare_two_cohorts(self, intent: Intent) -> Plan | None:
        """Two named cohorts on one metric is a comparison, not two answers.

        Mirroring ran the same analysis twice and left the reader to divide one
        table by the other; compare_building_groups returns both distributions
        and the ratio between their medians, which is what "compare A with B"
        asks for.
        """
        cohorts = (intent.entities or {}).get("building_type_all") or []
        metric = self.COMPARABLE.get(intent.metric or "")
        if len(cohorts) != 2 or not metric or not re.search(
                r"\bcompare\b|\bversus\b|\bvs\.?\b|\bagainst\b",
                (intent.question or "").lower()):
            return None
        plan = Plan(question=intent.question)
        for index, cohort in enumerate(cohorts, start=1):
            plan.add(PlanStep(f"s{index}", "find_buildings",
                              {"building_type": cohort, "limit": 5000},
                              resolution="graph_only",
                              purpose=f"the {cohort} cohort"))
        plan.add(PlanStep(
            "s3", "compare_building_groups",
            {"group_a": "$s1.column.building_id",
             "group_b": "$s2.column.building_id",
             "metric": metric, "label_a": cohorts[0], "label_b": cohorts[1],
             "year": (intent.time_range or {}).get("year"),
             # A comparison of shape is about every building of each kind, not
             # only the well-covered ones.
             "min_coverage": 0.0},
            depends_on=["s1", "s2"],
            purpose="both distributions and the ratio between their medians"))
        plan.answer_step = "s3"
        return plan.fill_agents()

    @staticmethod
    def _mirror_cohorts(plan: Plan, intent: Intent) -> Plan:
        """Run the same analysis again for each further cohort a question names.

        "Compare office buildings with healthcare buildings" produced one branch
        filtered to Office and compared it with nothing. The rules build a plan
        for a cohort, so the general repair is to build it once per cohort
        rather than to teach every branch about comparison: clone the steps,
        swap the type, and rename the references so the clone reads its own
        results and not the original's.
        """
        # A question can name several of any dimension: two building types to
        # compare, or a region that covers several countries. Whichever the
        # intent resolved as plural is the one to mirror on.
        # Only a dimension the question is *comparing*. A region names several
        # countries and means their union, which is one query with a list --
        # mirroring it makes two branches and reports whichever ran last, which
        # answered "education buildings in Europe" with Ireland alone.
        entities = intent.entities or {}
        field = next((f for f in ("building_type",)
                      if len(entities.get(f"{f}_all") or []) > 1), None)
        if field is None or not plan.steps:
            return plan
        cohorts = entities[f"{field}_all"]
        primary = entities.get(field)
        if not primary:
            return plan
        filtered = any(step.args.get(field) == primary for step in plan.steps)
        if not filtered:
            return plan

        import copy
        for index, cohort in enumerate(cohorts[1:], start=2):
            rename = {step.id: f"{step.id}c{index}" for step in plan.steps}

            def redirect(value):
                if isinstance(value, str) and value.startswith("$"):
                    for old_id, new_id in rename.items():
                        if value == f"${old_id}" or value.startswith(f"${old_id}."):
                            return "$" + new_id + value[len(old_id) + 1:]
                    return value
                if isinstance(value, list):
                    return [redirect(item) for item in value]
                if isinstance(value, dict):
                    return {key: redirect(item) for key, item in value.items()}
                return value

            for step in list(plan.steps[:len(rename)]):
                clone = copy.deepcopy(step)
                clone.id = rename[step.id]
                clone.depends_on = [rename.get(d, d) for d in step.depends_on]
                clone.args = {k: redirect(v) for k, v in clone.args.items()}
                if clone.args.get(field) == primary:
                    clone.args[field] = cohort
                clone.purpose = f"{step.purpose} ({cohort})".strip()
                plan.add(clone)
        plan.notes.append(
            "compared cohorts: " + ", ".join(str(c) for c in cohorts))
        return plan

    def _plan(self, intent: Intent, known_facts=None) -> Plan:
        plan = Plan(question=intent.question)
        entities, period = intent.entities, intent.time_range
        tag = intent.metric or "consumption"
        limit = entities.get("limit", 10)
        year = period.get("year")
        start, end = period.get("start"), period.get("end")
        # Grouping asked for by name: "segmented by subindustry" wants one row
        # per subindustry and got one per building.
        GROUPS = {"sites": "site", "cities": "city", "countries": "country",
                  "building_types": "building_type",
                  "subindustry": "subindustry",
                  "building_subtype": "building_subtype",
                  "subtype": "building_subtype",
                  "industry": "industry"}

        # A season with no year carries months instead of a window, and every
        # step that filters by period has to honour it or the answer covers the
        # whole record: "the daily load factor over the summer" was computed
        # from 28,876 building-days where the season holds 7,360.
        season_months = period.get("months") if period.get("spans_all_years") else None

        # Step 1 -- resolve the entity set, catalog-only wherever possible.
        explicit = entities.get("building_ids")
        if tag == "discovery" and not explicit:
            lowered = intent.question.lower()
            # The subject of the question is the noun being counted, not every
            # noun in the sentence. "How many buildings are in this dataset"
            # asks about buildings; "dataset" is the scope. Matching the scope
            # word first answered it with the one-row datasets catalog.
            # "per site" / "by country" / "each city" asks for a tally grouped
            # by that entity, which the catalog answers in one row per group.
            # Without this, "how many buildings per site" listed 1,636 buildings
            # and left the reader to tally them.
            if intent.group_by:
                entity = intent.group_by
                noun = entity
                plan.add(PlanStep(
                    "s1", "search_catalog",
                    {"entity": entity, "country": entities.get("country"),
                     "city": entities.get("city"),
                     "site_id": entities.get("site_id"),
                     "building_type": entities.get("building_type")},
                    resolution="graph_only",
                    purpose=f"one row per {noun}, counted from the catalog"))
                return plan.fill_agents()

            # Route on the subject the intent already resolved, rather than
            # re-deriving it from the text with a second, different rule. Two
            # rules that disagree is how "types of buildings" reached the
            # buildings branch while the intent said building_types.
            if intent.subject == "buildings":
                plan.add(PlanStep(
                    "s1", "find_buildings",
                    {"country": entities.get("country"),
                     "city": entities.get("city"),
                     "site_id": entities.get("site_id"),
                     "building_type": entities.get("building_type"),
                     "requires": intent.requirements or None,
                     "limit": 100000},
                    resolution="graph_only",
                    purpose="count or list buildings from the catalog"))
                return plan.fill_agents()
            entity = intent.subject or (
                "cities" if re.search(r"\bcit(?:y|ies)\b", lowered)
                else "countries" if re.search(r"\bcountr(?:y|ies)\b", lowered)
                else "sites" if re.search(r"\bsites?\b", lowered)
                else "datasets" if re.search(r"\bdatasets?\b", lowered)
                else "building_types")
            plan.add(PlanStep("s1", "search_catalog",
                              {"entity": entity,
                               "country": entities.get("country"),
                               "city": entities.get("city"),
                               "site_id": entities.get("site_id"),
                               "building_type": entities.get("building_type")},
                              resolution="graph_only",
                              purpose="answer from the catalog without scanning meters"))
            return plan.fill_agents()

        if explicit:
            ids: object = explicit
            plan.notes.append(f"{len(explicit)} building(s) named explicitly")
        else:
            # A peak per square metre needs a meter and a floor area, not a
            # whole year of record: eui_analysis also demands "enough data to
            # divide a year by", which is right for a total and wrong for an
            # extreme. Asking it of the education cohort dropped 110 of 604
            # buildings from a question about observed peaks.
            peak_intensity = bool(
                re.search(r"\bpeak\b|\bmaximum\b|\bmax\b", intent.question.lower())
                and re.search(r"\b[kmw]?w(?:h)?\s*(?:/|per\s*)\s*m\s*(?:2|²)"
                              r"|per\s+(?:square|m2|sqm)", intent.question.lower()))
            requires = {"eui": [] if peak_intensity else ["eui_analysis"],
                        "rank": ["eui_analysis"],
                        "per_occupant": ["occupancy_normalisation"],
                        "benchmark": ["peer_benchmark"],
                        "weather_relationship": ["weather_analysis"],
                        "temperature_response": ["weather_analysis"],
                        "weather_normalize": ["weather_analysis"],
                        "forecast": ["forecasting"],
                        "compare_geo": ["spatial_analysis"],
                        "peak_timing": ["spatial_analysis"]}.get(tag, [])
            # A data-quality question is *about* poor coverage, so the resolve
            # step must not apply a coverage floor that removes its own answers.
            floor = 0.0 if tag == "quality" else None
            # This step resolves the cohort the analysis then runs over, so its
            # limit is a cap on the answer. 500 is the tool default and the
            # portfolio is 1,636: "which buildings have more than 3 consecutive
            # days of zero readings" was answered with the 13 such buildings
            # among the first 500 by name, out of 49. The planning rules warn
            # the model about exactly this and the deterministic planner was
            # doing it itself.
            plan.add(PlanStep(
                "s1", "find_buildings",
                {"country": entities.get("country"), "city": entities.get("city"),
                 "site_id": entities.get("site_id"),
                 "building_type": entities.get("building_type"),
                 "requires": requires or None,
                 "limit": len(self.building_ids) + 1 if getattr(
                     self, "building_ids", None) else 5000,
                 "min_coverage": floor},
                resolution="graph_only",
                purpose="resolve the eligible set before touching measurements"))
            ids = "$s1.column.building_id"

        prefix = [] if explicit else ["s1"]
        # When buildings are named explicitly there is no resolve step, so the
        # analysis is s1 rather than leaving a gap at s1.
        base = 1 if explicit else 2
        nxt = (lambda n: f"s{n - 2 + base}")

        # Step 2+ -- the analysis itself.
        if tag == "entity_path":
            named = explicit or []
            places = [v for k, v in entities.items()
                      if k in ("city", "country", "site_id", "building_type")]
            operands = (named + places)[:2]
            if len(operands) < 2:
                plan.notes.append("a relationship question needs two named entities")
                plan.add(PlanStep("s1", "search_catalog", {"entity": "sites"},
                                  resolution="graph_only",
                                  purpose="no pair of entities resolved"))
                return plan.fill_agents()
            plan.steps.clear()
            plan.add(PlanStep("s1", "find_entity_paths",
                              {"source": operands[0], "target": operands[1],
                               "max_depth": 3},
                              resolution="graph_only",
                              purpose="how the two entities connect, by traversal"))
            return plan.fill_agents()

        if tag == "known_facts":
            plan.steps.clear()
            plan.add(PlanStep("s1", "lookup_derived_facts",
                              {"building_ids": explicit or None},
                              resolution="cached_fact",
                              purpose="read previously computed findings "
                                      "instead of recomputing"))
            return plan.fill_agents()

        if tag == "semantic_export":
            target = next((t for t in ("brick", "haystack", "bedes")
                           if t in intent.question.lower()), "brick")
            plan.steps.clear()
            plan.add(PlanStep("s1", "export_semantic_model",
                              {"target": target, "building_ids": explicit or None},
                              resolution="graph_only",
                              purpose=f"project the canonical model into {target}"))
            return plan.fill_agents()

        if tag == "per_occupant":
            plan.add(PlanStep(nxt(2), "calculate_normalized_performance",
                              {"building_ids": ids, "metric": "per_occupant",
                               "year": year},
                              depends_on=prefix,
                              purpose="occupancy-normalised intensity; refuses "
                                      "where occupancy metadata is absent"))
        elif tag in ("eui", "rank") and intent.subject in (
                "building_types", "cities", "countries", "sites"):
            # "Which category has the highest X" ranks categories, not the
            # buildings inside them: ranking buildings answered it with
            # Eagle_health_Lucinda, which is not a category at all.
            dimension = {"building_types": "building_type", "cities": "city",
                         "countries": "country", "sites": "site"}[intent.subject]
            asked = intent.question.lower()
            statistic = ("stddev" if re.search(
                r"\bvariab|\bdeviation|\bstd\b|\bspread\b", asked)
                else "mean" if re.search(r"\baverage|\bmean\b", asked)
                else "median" if re.search(r"\bmedian\b", asked)
                else "max" if re.search(r"\bpeak|\bmaximum|\bhighest\s+demand\b",
                                        asked)
                else "total")
            plan.add(PlanStep(
                nxt(2), "calculate_consumption_statistics",
                {"building_ids": ids, "across": dimension,
                 "metrics": [statistic], "order_by": statistic,
                 "order": "asc" if re.search(r"\blowest|\bleast\b", asked)
                          else "desc",
                 "limit": 1},
                depends_on=prefix,
                purpose=f"the {dimension} at the top of {statistic}"))
        elif tag in ("eui", "rank"):
            asked = intent.question.lower()
            at_least, at_most = resolve_threshold(intent.question)
            if at_least is not None or at_most is not None:
                # "Which buildings exceed 50 W/m2" asks for a set, not an
                # order. Ranking it and taking ten answers a different
                # question, and the filter has to be the tool's because a
                # large result is handed back as an artifact.
                peak = bool(re.search(r"\bpeak\b|\bmaximum\b|\bmax\b", asked))
                watts = bool(re.search(r"\bw\s*(?:/|per\s*)\s*m\s*(?:2|²)", asked))
                metric = ("peak_power_density" if peak and watts
                          else "peak_per_area" if peak else "eui")
                plan.add(PlanStep(nxt(2), "calculate_normalized_performance",
                                  {"building_ids": ids, "metric": metric,
                                   "year": year if year else LATEST_FULL_YEAR,
                                   "at_least": at_least, "at_most": at_most},
                                  depends_on=prefix,
                                  purpose=f"the buildings whose {metric} is in "
                                          f"the range the question named"))
                return plan.fill_agents()
            # An intensity is per year. "Use per square metre" with no period
            # named is the latest full year, the same reading "annual" gets:
            # pooled over the snapshot's eighteen months it is not an EUI at
            # all, just a bigger number, and it reorders four of the top ten
            # offices.
            plan.add(PlanStep(nxt(2), "rank_buildings",
                              {"building_ids": ids, "metric": "eui",
                               "year": year if year else LATEST_FULL_YEAR,
                               "limit": limit,
                               "order": "asc" if re.search(r"\blowest|least|best\b",
                                                           intent.question.lower())
                                        else "desc"},
                              depends_on=prefix,
                              purpose="rank on area-normalised intensity"))
        elif tag == "benchmark":
            target = (explicit or [None])[0]
            if target:
                measure = resolve_measure(intent.question)
                # "Among all office buildings" names the cohort, and a peer
                # group matched on type *and* country is a narrower one: the
                # percentile it reports is against the building's compatriots
                # rather than against the buildings the question asked about.
                named_cohort = (entities.get("building_type")
                                and re.search(r"\b(?:among|across|out\s+of|"
                                              r"compared\s+(?:to|with))\b",
                                              intent.question.lower()))
                if named_cohort:
                    plan.add(PlanStep(
                        "s1", "find_buildings",
                        {"building_type": entities.get("building_type"),
                         "country": entities.get("country"),
                         "requires": ["consumption_analysis"], "limit": 5000},
                        resolution="graph_only",
                        purpose="the cohort the question compares against"))
                    peer_reference = "$s1.column.building_id"
                else:
                    plan.add(PlanStep("s1", "construct_peer_group",
                                      {"building_id": target,
                                       "match_on": ["building_type", "country"]},
                                      purpose="build a comparable cohort"))
                    peer_reference = "$s1.data.peer_group"
                plan.add(PlanStep("s2", "benchmark_building",
                                  {"building_id": target,
                                   "peer_ids": peer_reference,
                                   "metric": measure,
                                   "year": year or LATEST_FULL_YEAR,
                                   "min_coverage": 0.0},
                                  depends_on=["s1"],
                                  purpose="position the target within its cohort"))
            else:
                plan.add(PlanStep("s2", "identify_peer_outliers",
                                  {"building_ids": ids, "metric": "eui", "year": year},
                                  depends_on=prefix,
                                  purpose="find cohort members that differ materially"))
        elif tag == "quality":
            # "below 90%" is a filter, not a report: profile_data_quality
            # describes every stream, detect_data_issues returns the ones that
            # breach a stated threshold.
            lowered_q = intent.question.lower()
            # A stretch of dark days is its own check, and the share of zeros
            # over the whole record answers a different question with a much
            # larger number. The edge travels with the phrasing: "more than 3"
            # is a run of 4, "at least 3" is a run of 3.
            zero_run = re.search(
                r"(\d+)\s*(?:or more\s+)?consecutive\s+days?|"
                r"consecutive\s+days?[\w\s]{0,20}?(\d+)", lowered_q)
            threshold = re.search(r"(?:below|under|less than|<)\s*(\d{1,3})\s*%",
                                  lowered_q)
            if zero_run and re.search(r"\bzero|\bno\s+(?:readings?|consumption)",
                                      lowered_q):
                days = int(next(g for g in zero_run.groups() if g))
                if re.search(r"\b(?:more than|over|exceeding|longer than)\b",
                             lowered_q):
                    days += 1
                plan.add(PlanStep(nxt(2), "detect_data_issues",
                                  {"building_ids": ids, "checks": ["zero_run"],
                                   "consecutive_zero_days_at_least": days},
                                  depends_on=prefix,
                                  purpose="buildings dark for a run of whole days"))
            elif gap_bound(lowered_q):
                # "Gaps exceeding one week" is a filter with a number in it,
                # and the rule route was describing every stream instead:
                # 1,578 rows where 610 buildings have such a gap. The edge
                # travels with the phrasing exactly as it does for a zero run
                # -- "exceeding" is strictly longer, "at least" includes the
                # value -- and three streams sit exactly on a one-week gap.
                hours, bound = gap_bound(lowered_q)
                plan.add(PlanStep(nxt(2), "detect_data_issues",
                                  {"building_ids": ids, "checks": ["gaps"],
                                   "gap_hours": hours, "gap_bound": bound},
                                  depends_on=prefix,
                                  purpose=f"streams whose longest gap is "
                                          f"{bound.replace('_', ' ')} {hours}h"))
            elif threshold:
                # "Below 90% coverage" is a ceiling on coverage and
                # `max_coverage` is exactly that. "Less than 2% *missing*" is
                # the same sentence inverted, and passing 0.02 here asked for
                # streams with under 2% coverage -- the ten emptiest meters
                # rather than the 1,183 sound ones. The cohort step carries
                # the floor, measured against the hours the year should hold.
                share = int(threshold.group(1)) / 100
                if re.search(r"\bmissing\b|\bgaps?\b|\babsent\b",
                             lowered_q):
                    for step in plan.steps:
                        if step.tool == "find_buildings":
                            step.args["min_coverage"] = round(1 - share, 6)
                            if intent.time_range.get("year"):
                                step.args["coverage_year"] = int(
                                    intent.time_range["year"])
                            break
                    plan.answer_step = plan.steps[0].id
                    return plan.fill_agents()
                plan.add(PlanStep(nxt(2), "detect_data_issues",
                                  {"building_ids": ids, "checks": ["coverage"],
                                   "max_coverage": share},
                                  depends_on=prefix,
                                  purpose="streams breaching the stated coverage floor"))
            else:
                plan.add(PlanStep(nxt(2), "profile_data_quality",
                                  {"building_ids": ids}, depends_on=prefix,
                                  resolution="graph_only",
                                  purpose="quality metrics are precomputed per stream"))
            if re.search(r"\b(most|worst|highest|lowest|top|least)\b", lowered_q):
                plan.notes.append("ranked on coverage; profile_data_quality already "
                                  "returns rows ordered by coverage ascending")
        elif tag == "load_profile":
            plan.add(PlanStep(nxt(2), "calculate_load_profile",
                              {"building_ids": ids, "profile": "hourly",
                               "start": start, "end": end},
                              depends_on=prefix, purpose="typical daily load shape"))
        elif tag == "compare_temporal":
            asked = intent.question.lower()
            comparison = ("summer_winter" if re.search(r"summer|winter", asked)
                          else "weekday_weekend")
            args = {"building_ids": ids, "comparison": comparison,
                    "start": start, "end": end}
            # "Does the profile change" asks about the rhythm, not the size.
            # The ratio of means says a building drew more; it cannot say the
            # shape moved, and those are different answers.
            shape = asks_about_shape(asked)
            if shape:
                args["measure"] = "profile_shape"
            plan.add(PlanStep(nxt(2), "compare_temporal_profiles", args,
                              depends_on=prefix, purpose=f"contrast {comparison}"))
        elif tag == "baseload":
            # The default night is 00:00-05:59, a convention rather than an
            # answer to any particular question. Where the question names its
            # own window -- "after-hours (midnight to 4 AM)" -- that window is
            # the one to use, and a top-decile ask is a rank bucket over the
            # cohort, which the tool can do itself.
            args = {"building_ids": ids, "method": "night_mean",
                    "start": start, "end": end}
            level = resolve_percentile(intent.question)
            if level is not None:
                args["method"], args["percentile"] = "percentile", level
            window = resolve_clock_window(intent.question)
            if window:
                args["night_start_hour"], args["night_last_hour"] = window
            asked = intent.question.lower()
            if re.search(r"\btop[-\s]?decile\b|\btop\s+10\s*%", asked):
                args["decile_at_least"] = 10
            # "Unusually high night consumption" asks which buildings stand
            # out, not what every building's ratio is. Answering it with the
            # full cohort table leaves the reader to eyeball 123 rows and call
            # it an answer.
            if re.search(r"\bunusual\w*|\babnormal\w*|\banomalous\b|"
                         r"\bexcessiv\w*|\bsuspicious\w*", asked):
                args["select"] = ("unusual_low"
                                  if re.search(r"\blow\b|\bleast\b", asked)
                                  else "unusual_high")
            plan.add(PlanStep(nxt(2), "calculate_baseload", args,
                              depends_on=prefix, purpose="overnight baseload"))
        elif tag == "operating":
            asked = intent.question.lower()
            args = {"building_ids": ids, "start": start, "end": end}
            if re.search(r"\bweekday\b|\bweekdays\b|\bworking\s+day", asked):
                args["days"] = "weekday"
            elif re.search(r"\bweekend\b", asked):
                args["days"] = "weekend"
            # A question that states the working day is not asking the tool to
            # infer one: "outside typical business hours (7am-7pm, weekdays)"
            # fixes the window, and everything else -- evenings, nights, the
            # whole weekend -- is what it wants measured.
            stated = (resolve_clock_window(intent.question)
                      if re.search(r"business\s+hours|working\s+hours"
                                   r"|outside\b", asked) else None)
            if stated:
                args["business_start_hour"], args["business_last_hour"] = stated
                args["business_days"] = ("weekend" if re.search(r"\bweekends?\b", asked)
                                         else "weekday")
                args.pop("days", None)     # the weekend has to be counted, not dropped
                args["report"] = "after_hours"
            # "What time does it start" wants the opening hour, not the length
            # of the day; "when does it shut down" wants the closing one.
            elif re.search(r"\bstart\w*\b|\bramp\w*\s+up|\bopens?\b"
                           r"|\bwhat\s+time\b.*\bmorning\b", asked):
                args["report"] = "start"
            elif re.search(r"\bshut\s*down\b|\bcloses?\b|\bend\s+of\s+the\s+day\b",
                           asked):
                args["report"] = "end"
            plan.add(PlanStep(nxt(2), "analyze_operating_pattern", args,
                              depends_on=prefix,
                              purpose="inferred schedule and after-hours share"))
        elif tag == "demand":
            # A question naming one building wants that building's metrics; a
            # question asking *which* building wants one row, ordered.
            asked = intent.question.lower()
            pick = ("min_demand" if re.search(r"\bminimum|\blowest|\bmin\b", asked)
                    else "peak_demand" if re.search(
                        r"\bmaximum|\bhighest|\bpeak\b|\bmax\b", asked)
                    else None)
            # "Which building" is one row and "which buildings" is a list, and
            # both are ranked questions. Ordering only for the singular left
            # the plural form returning every office building unordered, with
            # the ten the question asked for somewhere inside it.
            ranked = re.search(r"\bhighest|\blowest|\bmost\b|\bleast\b|"
                               r"\btop\b|\bbottom\b|\bmaximum|\bminimum", asked)
            if re.search(r"\bpeak[- ]to[- ]average|\bpeak[- ]to[- ]mean", asked):
                pick = "peak_to_average_ratio"
            season = {"months": season_months} if season_months else {}
            ordering = ({"order_by": pick,
                         "order": ("asc" if pick == "min_demand"
                                   or re.search(r"\blowest|\bleast\b", asked)
                                   else "desc"),
                         "limit": (1 if intent.output_format == "entity"
                                   else limit)}
                        if pick and (ranked or intent.output_format == "entity")
                        else {})
            plan.add(PlanStep(nxt(2), "calculate_demand_metrics",
                              {"building_ids": ids, "start": start, "end": end,
                               **season, **ordering},
                              depends_on=prefix,
                              purpose=("the one building at that extreme"
                                       if ordering else
                                       "peak, load factor, peak-to-average")))
        elif tag == "year_over_year":
            plan.add(PlanStep(nxt(2), "compare_temporal_profiles",
                              {"building_ids": ids, "comparison": "year_year"},
                              depends_on=prefix,
                              purpose="the same building across both years"))
        elif tag == "weather_value":
            # A weather variable asked about for its own value needs the weather
            # series, not the electricity one, and needs no building cohort at
            # all: it is a property of the site.
            plan.steps.clear()
            variable = next(
                (v for v, pattern in WEATHER_VARIABLES
                 if re.search(pattern, intent.question.lower())),
                "air_temperature")
            plan.add(PlanStep(
                "s1", "calculate_weather_statistics",
                {"site_ids": [entities["site_id"]] if entities.get("site_id") else None,
                 "variables": [variable], "start": start, "end": end},
                resolution="compute",
                purpose=f"{variable.replace('_', ' ')} statistics for the site"))
            return plan.fill_agents()
        elif tag == "weather_during_peak":
            plan.steps.clear()
            share = re.search(r"top\s+(\d+(?:\.\d+)?)\s*%", intent.question.lower())
            plan.add(PlanStep(
                "s1", "calculate_weather_statistics",
                {"building_ids": explicit, "during": "top_demand_hours",
                 "top_share_pct": float(share.group(1)) if share else 5.0,
                 "variables": ["air_temperature"],
                 "months": period.get("months"), "start": start, "end": end},
                purpose="the weather over the hours the building was busiest"))
            plan.answer_step = "s1"
            return plan.fill_agents()
        elif tag == "weather_relationship":
            asked = intent.question.lower()
            # Three different questions wear the same tag. "What is the
            # correlation" asks for a correlation; "did consumption spike above
            # 35C" asks for a contrast across a line, which a correlation over
            # the whole range cannot answer; anything else is the regression.
            contrast = re.search(r"(?:above|over|exceed\w*|higher\s+than|>)\s*"
                                 r"(\d+(?:\.\d+)?)\s*(?:°\s*)?c\b", asked)
            if contrast:
                plan.add(PlanStep(
                    nxt(2), "analyze_weather_relationship",
                    {"building_ids": ids, "method": "threshold_contrast",
                     "variable": "air_temperature",
                     "threshold": float(contrast.group(1)),
                     "months": period.get("months"), "start": start, "end": end},
                    depends_on=prefix,
                    purpose=f"consumption above {contrast.group(1)}C against "
                            f"the rest of the window"))
            else:
                plan.add(PlanStep(
                    nxt(2), "analyze_weather_relationship",
                    {"building_ids": ids,
                     "method": ("correlation"
                                if re.search(r"\bcorrelat\w*", asked)
                                else "regression"),
                     "start": start, "end": end},
                    depends_on=prefix,
                    purpose="temperature-electricity relationship"))
        elif tag == "temperature_response":
            # "The cooling change-point temperature for site Robin" is one
            # model over the site's pooled daily consumption. Fitting each of
            # its buildings and handing back the list answers a different
            # question, and the median of those is not the site's balance
            # point.
            site = entities.get("site_id")
            if site and intent.subject in ("sites", None):
                plan.steps.clear()
                plan.add(PlanStep("s1", "fit_temperature_response",
                                  {"site_ids": [site], "across": "site",
                                   "model": "cooling", "start": start, "end": end},
                                  purpose=f"the balance point for site {site}"))
                plan.answer_step = "s1"
                return plan.fill_agents()
            plan.add(PlanStep(nxt(2), "fit_temperature_response",
                              {"building_ids": ids, "model": "cooling",
                               "start": start, "end": end},
                              depends_on=prefix,
                              purpose="change-point balance temperature and slope"))
        elif tag == "weather_normalize":
            plan.add(PlanStep(nxt(2), "weather_normalize_consumption",
                              {"building_ids": ids,
                               "baseline_start": "2016-01-01", "baseline_end": "2017-01-01",
                               "target_start": start or "2017-06-01",
                               "target_end": end or "2017-09-01"},
                              depends_on=prefix,
                              purpose="separate weather from behaviour"))
        elif tag == "compare_geo":
            plan.steps.clear()          # catalog-only: no entity resolution needed
            level = ("country" if re.search(r"\bcountr(?:y|ies)\b",
                                            intent.question.lower())
                     else "site" if re.search(r"\bsites?\b", intent.question.lower())
                     else "city")
            plan.add(PlanStep("s1", "aggregate_geographically",
                              {"level": level,
                               "metric": resolve_measure(intent.question),
                               "building_type": entities.get("building_type"),
                               "year": year},
                              resolution="graph_only",
                              purpose=f"roll up to {level} level"))
            return plan.fill_agents()
        elif tag == "peak_timing" and explicit:
            plan.steps.clear()
            plan.add(PlanStep("s1", "analyze_operating_pattern",
                              {"building_ids": explicit, "start": start, "end": end},
                              purpose="typical peak hour from the mean daily "
                                      "profile, in local time"))
            return plan.fill_agents()
        elif tag == "peak_timing":
            plan.steps.clear()          # catalog-only
            plan.add(PlanStep("s1", "analyze_spatiotemporal_patterns",
                              {"level": "city", "pattern": "peak_timing",
                               "building_type": entities.get("building_type")},
                              resolution="graph_only",
                              purpose="peak hour by geography, in local time"))
            return plan.fill_agents()
        elif tag == "forecast":
            target = (explicit or [None])[0]
            asked = intent.question
            # A *scoring* question works over a cohort: "how accurately does
            # this model forecast Public services in the UK" is seven
            # buildings, and the one-target guard below turned it into a
            # data-availability listing. Only an actual forecast needs a
            # single subject, because a forecast is a series and a cohort of
            # them is not an answer.
            if not target and not (asks_forecast_accuracy(asked)
                                   or asks_improvement(asked)):
                plan.add(PlanStep("s2", "get_data_availability",
                                  {"building_ids": ids}, depends_on=prefix,
                                  resolution="graph_only",
                                  purpose="pick a forecastable building"))
                plan.notes.append("no building named; forecasting needs one target")
                return plan.fill_agents()
            # "How accurately do LightGBM and ARIMA forecast this building
            # over March and April, and what are their MAPE values" is not a
            # request for a forecast. A forecast is issued once and runs
            # forward; scoring a two-month window with a single 1,464-hour
            # horizon measures how fast a model decays. What is asked for is a
            # rolling backtest, one row per model, on identical hours.
            if asks_forecast_accuracy(asked) or asks_improvement(asked):
                # An ablation is two variants of one model differing in one
                # feature, and the order is the question's: baseline first.
                pair = resolve_ablation(asked) if asks_improvement(asked) else None
                named = list(pair) if pair else resolve_forecast_models(asked)
                named = named or resolve_forecast_models(asked)
                # "Does adding metadata help" is two arms of one pooled model,
                # because metadata is constant within a building and can only
                # be a feature of a fit that spans them.
                pooled = asks_pooled_metadata(asked)
                if pooled and named:
                    named = [m for base in named
                             if base in ("lightgbm", "knn", "gradient_boosting",
                                         "linear")
                             for m in (base, f"{base}_metadata")] or named
                args = {"models": named or ["seasonal_naive"],
                        "start": start, "end": end,
                        "horizon_hours": resolve_horizon_hours(asked, default=24),
                        "metrics": resolve_accuracy_metrics(asked),
                        "weather_variables": resolve_weather_variables(asked)
                                             if any(m.endswith("_weather")
                                                    for m in (named or []))
                                             or "weather" in asked.lower()
                                             else None,
                        "pooled": pooled,
                        "report": "improvement" if (pair or pooled) else "table"}
                # One named building is a building; a cohort is whatever the
                # discovery step found. "Public services in the UK" is seven
                # buildings, and scoring one of them answers a question about
                # one building.
                if len(explicit or []) == 1 and not intent.group_by:
                    args["building_id"] = explicit[0]
                    plan.add(PlanStep("s1", "backtest_forecast_models", args,
                                      purpose="score over the stated window, "
                                              "rolling origin"))
                else:
                    args["building_ids"] = ids
                    plan.add(PlanStep(nxt(2), "backtest_forecast_models", args,
                                      depends_on=prefix,
                                      purpose="score over the stated window, "
                                              "rolling origin, one row per model"))
                if not named:
                    plan.notes.append(
                        "the question names no model, so the seasonal naive "
                        "baseline is scored and reported as the planner's choice")
                if pair:
                    plan.notes.append(
                        f"an ablation: {pair[0]} against {pair[1]}, identical "
                        f"but for the weather features, scored on the same hours")
                return plan.fill_agents()
            horizon = resolve_horizon_hours(asked)
            # The question names the method, and running a different one is a
            # wrong answer even when its numbers are reasonable: "using a
            # simple 7-day trailing moving average" is an instruction. Every
            # forecast here used the seasonal-naive baseline regardless, so
            # Q104 was answered by repeating last Tuesday and Q105 by ignoring
            # the temperature it was told to train on.
            model, named = resolve_forecast_model(asked)
            args = {"building_id": target, "horizon_hours": horizon,
                    "model": model}
            if model == "moving_average":
                args["window_days"] = resolve_moving_window_days(asked)
            # `cutoff` is exclusive: the model sees data strictly before it.
            # "after 2017-07-31" therefore means the whole of the 31st is
            # history and the horizon starts on 1 August.
            cutoff = start
            if cutoff and re.search(r"\bafter\b", intent.question.lower()):
                cutoff = str((pd.Timestamp(cutoff) + pd.Timedelta(days=1)).date())
            args["cutoff"] = cutoff
            if not named:
                plan.notes.append(
                    "the question names no forecasting method, so the seasonal "
                    "naive baseline is used and reported as the planner's choice")
            plan.add(PlanStep("s1", "forecast_timeseries", args,
                              purpose=f"forecast from a frozen cutoff "
                                      f"(exclusive) using {model}"))
            # Scoring is only meaningful when the horizon falls inside the
            # snapshot. Forecasting past its end has no ground truth, and a
            # refusal there would mask an otherwise valid forecast.
            if cutoff:
                plan.add(PlanStep("s2", "evaluate_forecast",
                                  {"forecast_handle": "$s1.handle",
                                   "building_id": target},
                                  depends_on=["s1"],
                                  purpose="score against the observed continuation"))
            else:
                plan.notes.append("no cutoff given, so the horizon falls beyond the "
                                  "snapshot; forecast returned without scoring")
        elif tag == "anomaly":
            # A question about spikes is not a question about deviation. Asked
            # for suspicious spikes, a two-sided test answered with 57 readings
            # far below the building's normal level.
            asked = intent.question.lower()
            # ...and a question about a sustained SHIFT is neither. A step up
            # that persists shows as no anomalies at all once the new level
            # becomes the norm, so "shifts over 96 hours" answered by a daily
            # outlier test returns nothing and reads as "there were none".
            # The change-point route already existed; this branch simply never
            # asked for it.
            if asks_for_change_point(asked):
                # The tool takes no window: it segments the whole series,
                # which is what a change point means -- a date is only a shift
                # relative to what came before it, and clipping the history
                # would hide the very comparison being asked for.
                plan.add(PlanStep(nxt(2), "detect_change_points",
                                  {"building_ids": ids},
                                  depends_on=prefix,
                                  purpose="dates where the level shifts"))
                plan.notes.append(
                    "the question asks about a sustained shift, not an odd "
                    "day, so change-point detection answers it")
                return plan
            # "More than two standard deviations *from* its typical average"
            # is two-sided -- a day far below is as much a deviation as a day
            # far above -- while "more than twice its typical" and "above" name
            # one side. The preposition is the whole difference, and reading
            # every "more than" as high drops half the answer.
            deviates_from = re.search(
                r"\bdeviat\w*\b|\bdiffer\w*\b"
                r"|\b(?:from|away\s+from)\s+(?:its|their|the)\s+"
                r"(?:typical|usual|normal|average|mean|baseline|expected)\b",
                asked)
            if deviates_from:
                which = "both"
            elif re.search(r"\bspikes?\b|\bsurges?\b|\bunusually\s+high\b"
                           r"|\bexcessive\b|\btoo\s+high\b|\bexceed\w*\b"
                           r"|\b(?:more|greater)\s+than\b|\babove\b", asked):
                which = "high"
            elif re.search(r"\bdips?\b|\bdrops?\b|\bunusually\s+low\b"
                           r"|\bdropouts?\b", asked):
                which = "low"
            else:
                which = "both"
            # Which test, and where the line falls, are both in the question.
            # Every anomaly question ran the same 3-sigma shape test before
            # this: "exceed twice its typical daily average" is a multiple, not
            # a profile deviation, and "more than 2 standard deviations" is a
            # different line on a different statistic again.
            method, named = resolve_anomaly_method(asked)
            threshold = resolve_anomaly_threshold(asked, method)
            if not named:
                plan.notes.append(
                    "the question names no anomaly test, so the load-shape "
                    "deviation test is used at 3 sigma")
            purpose = {
                "multiple_of_typical":
                    f"days over {threshold:g}x the building's own typical day",
                "zscore":
                    f"days over {threshold:g} standard deviations from the "
                    f"building's own daily mean",
                "daily_profile": "days whose load shape departs from the "
                                 "building's own usual shape",
            }.get(method, "days departing from the building's own history")
            if asks_detector_validation(asked):
                # Whether one detector beats another needs labels, and the
                # only labels available are planted ones. Counting flags
                # answers a different question: more is not better.
                change_asked = bool(re.search(
                    r"\bvalue[-\s]change|\bchange\s+features?\b"
                    r"|\bdifference\s+features?\b|\bdelta\b", asked))
                configurations = (
                    [{"method": method, "feature_set": "level"},
                     {"method": method, "feature_set": "level_and_change"}]
                    if change_asked and method == "local_outlier_factor"
                    else [{"method": method, "threshold": threshold},
                          {"method": "modified_zscore", "threshold": 3.5},
                          {"method": "first_difference", "threshold": 3.0}])
                plan.add(PlanStep(nxt(2), "validate_anomaly_detection",
                                  {"building_ids": ids,
                                   "configurations": configurations,
                                   "start": start, "end": end},
                                  depends_on=prefix,
                                  purpose="score the detectors against planted "
                                          "anomalies, which is the only way "
                                          "one can be shown to beat another"))
                return plan.fill_agents()
            if asks_sensitivity(asked):
                # "How sensitive is the rate to the threshold" is answered by
                # running the detector over a grid, not by running it once at
                # one of the settings being asked about.
                plan.add(PlanStep(nxt(2), "sweep_anomaly_sensitivity",
                                  {"building_ids": ids, "method": method,
                                   "start": start, "end": end},
                                  depends_on=prefix,
                                  purpose=f"how far the {method} rate moves "
                                          f"across its settings"))
                return plan.fill_agents()
            day_type = resolve_day_type(asked)
            if day_type != "all":
                purpose += f", restricted to {day_type}s"
            clock = resolve_clock_window(asked)
            if clock:
                purpose += f", hours {clock[0]:02d}-{clock[1]:02d} only"
            plan.add(PlanStep(nxt(2), "detect_anomalies",
                              {"building_ids": ids, "method": method,
                               "threshold": threshold, "direction": which,
                               "days": day_type,
                               "hour_from": clock[0] if clock else None,
                               "hour_to": clock[1] if clock else None,
                               "start": start, "end": end},
                              depends_on=prefix, purpose=purpose))
        elif tag == "change_point":
            # "Identify a building with at least 3 years of continuous history,
            # then detect..." states a precondition, and the precondition is
            # the answer: the snapshot spans two calendar years, so nothing
            # qualifies and the analysis has nothing to run on. Dropping the
            # requirement runs the detector on a cohort the question excluded
            # and reports change points as though the question were answerable.
            span = re.search(r"\bat\s+least\s+(\d+|one|two|three|four|five)\s+"
                             r"years?\b[^.?]{0,40}\b(?:continuous|uninterrupted|"
                             r"complete|unbroken)?\s*\b(?:consumption\s+)?"
                             r"(?:history|record|data)\b", intent.question.lower())
            if span and prefix:
                token = span.group(1)
                years = int(token) if token.isdigit() else WORD_NUMBERS.get(token, 0)
                if years:
                    for step in plan.steps:
                        if step.tool == "find_buildings":
                            step.args["min_months_of_data"] = years * 12
                            step.args["gap_at_most_hours"] = 24
                            plan.notes.append(
                                f"the question requires {years} years of "
                                f"continuous history; the cohort is screened "
                                f"for it before anything is detected")
            plan.add(PlanStep(nxt(2), "detect_change_points", {"building_ids": ids},
                              depends_on=prefix, purpose="persistent level shifts"))
        elif tag == "cluster":
            # A question about a *shift* needs the same building in the data
            # more than once; clustering one shape per building can only show
            # differences between buildings.
            asked = intent.question.lower()
            seasonal = bool(re.search(r"\bseasonal\b|\bbetween\s+seasons\b"
                                      r"|\bshifts?\b|\bchange\w*\b", asked)
                            and re.search(r"\bseasonal\b|\bseasons?\b", asked))
            # The word "cluster" in the sentence makes this a cluster
            # question by tag, but naming three distance measures makes it a
            # question about the measures. Comparing them is the ask; the
            # clustering is the context it is asked in.
            measures = resolve_similarity_measures(asked)
            if measures:
                plan.add(PlanStep(nxt(2), "compare_similarity_measures",
                                  {"building_ids": ids, "measures": measures},
                                  depends_on=prefix,
                                  purpose=f"whether {', '.join(measures)} pick "
                                          f"the same neighbour"))
                return plan.fill_agents()
            method, method_named = resolve_cluster_method(asked)
            distance = resolve_cluster_distance(asked)
            # k-means minimises squared Euclidean distance to a centroid and
            # is undefined under cosine or correlation. Asked to cluster UK
            # education buildings under cosine, the plan paired the default
            # algorithm with the named distance and the tool refused -- a
            # correct refusal to an incoherent request that nobody made. The
            # question named the distance, not the algorithm, so the distance
            # is what to keep: hierarchical linkage accepts it.
            if distance != "euclidean" and method == "kmeans" and not method_named:
                method = "average"
                plan.notes.append(
                    f"k-means cannot form clusters under {distance}, so "
                    f"average-linkage hierarchical clustering is used -- the "
                    f"question named the distance, not the algorithm")
            # One named building means the samples are its days; a cohort
            # means the samples are the buildings.
            one_building = len(ids) == 1 if isinstance(ids, list) else False
            args = {"building_ids": ids,
                    "scope": ("within_building" if one_building
                              else "across_buildings"),
                    "n_clusters": resolve_n_clusters(asked),
                    "method": method,
                    "distance": distance,
                    # What normalising is *for* is making buildings of
                    # different size comparable. Across a cohort it is
                    # required -- otherwise the clusters are a ranking by
                    # floor area. Within one building every day comes off the
                    # same meter, so dividing each day by its own mean throws
                    # away the level, which is half of what separates a
                    # working day from a quiet one. Clustering
                    # Eagle_office_Lamont's 330 days raw gives 95/104/131;
                    # normalised it gives 84/105/141, and the question that
                    # wants the second says "normalized".
                    "normalize": ("mean" if (one_building is False
                                             or asks_normalised(asked))
                                  else "none")}
            # Only when k-means is what actually ran: the distance branch
            # above may have switched to hierarchical linkage, and saying both
            # "average linkage was used" and "k-means is used" leaves the
            # reader to guess which of the two notes is stale.
            if not method_named and method == "kmeans":
                plan.notes.append(
                    "the question names no clustering algorithm, so k-means is "
                    "used and reported as the planner's choice")
            if re.search(r"\bweekly\b|\bweek[- ]?long\b|\bday\s+of\s+week\b", asked):
                args["profile"] = "weekly"
            if seasonal:
                args["profile"], args["by_season"] = "weekly", True
            if asks_cluster_stability(asked):
                plan.add(PlanStep(nxt(2), "compare_cluster_stability",
                                  {"building_ids": ids,
                                   "n_clusters": args["n_clusters"],
                                   "method": args["method"]},
                                  depends_on=prefix,
                                  purpose="cluster weekday and weekend shapes "
                                          "separately and compare them"))
                return plan.fill_agents()
            plan.add(PlanStep(nxt(2), "cluster_timeseries", args,
                              depends_on=prefix,
                              purpose="group by load shape, one profile per "
                                      "building per season" if seasonal
                                      else "group by load shape"))
        elif tag == "similar":
            # "Which building TYPES are most alike" is a question about
            # groups, and every route here wants a reference building. With
            # none named the alternatives are to give up, or -- because
            # "min-max normalized" also reads as a normalisation question --
            # to answer with median EUI, a different quantity in different
            # units.
            asked = intent.question.lower()
            measures = resolve_similarity_measures(asked)
            if measures:
                plan.add(PlanStep(nxt(2), "compare_similarity_measures",
                                  {"building_ids": ids, "measures": measures},
                                  depends_on=prefix,
                                  purpose=f"whether {', '.join(measures)} pick "
                                          f"the same neighbour"))
                return plan.fill_agents()
            group = resolve_similarity_group(asked)
            if group:
                plan.add(PlanStep(nxt(2), "correlate_group_profiles",
                                  {"building_ids": ids, "group_by": group,
                                   # "Most similar" is a distance question.
                                   # Correlation asks whether two groups rise
                                   # and fall together; distance asks whether
                                   # they sit on top of each other, and two
                                   # cities can move in step while drawing
                                   # quite different amounts. Asked which two
                                   # cities in different countries have the
                                   # most similar weekday office profiles,
                                   # correlation answered Cornell/London where
                                   # the distance answer is Dublin/Tempe.
                                   "metric": ("spearman" if "spearman" in asked
                                              else "pearson" if re.search(
                                                  r"\bcorrelat\w+", asked)
                                              else "euclidean" if re.search(
                                                  r"\bmost\s+similar\b|\bclosest\b"
                                                  r"|\bmost\s+alike\b", asked)
                                              else "pearson"),
                                   "normalize": ("mean" if re.search(
                                       r"\bmean[-\s]normali", asked) else "minmax"),
                                   # "Weekday load profiles" is a different
                                   # shape from the week's: the weekend
                                   # flattens the working day out of the mean.
                                   "days": resolve_day_type(asked),
                                   # "Load profile" is the 24-hour shape.
                                   "shape": ("daily_profile"
                                             if re.search(r"\bprofiles?\b", asked)
                                             else "series"),
                                   # "Two cities in different countries" rules
                                   # out the pair a free search usually finds.
                                   "require_different":
                                       resolve_require_different(asked, group),
                                   **{k: v for k, v in
                                      (("start", (intent.time_range or {}).get("start")),
                                       ("end", (intent.time_range or {}).get("end")))
                                      if v}},
                                  depends_on=prefix,
                                  purpose=f"which {group.replace('_', ' ')}s "
                                          f"move together"))
                return plan.fill_agents()
            target = (explicit or [None])[0]
            if not target:
                plan.notes.append("similarity needs a named reference building")
                plan.add(PlanStep(nxt(2), "get_data_availability",
                                  {"building_ids": ids}, depends_on=prefix,
                                  resolution="graph_only", purpose="no target named"))
                return plan.fill_agents()
            # "Which five OFFICE buildings are most similar to X" searches
            # the offices, not the portfolio. Without the cohort the answer
            # came back with health buildings in it.
            cohort = (intent.entities or {}).get("building_type")
            if cohort:
                plan.add(PlanStep("s0", "find_buildings",
                                  {"building_type": cohort, "limit": 5000},
                                  resolution="graph_only",
                                  purpose=f"the {cohort} cohort to search"))
            plan.add(PlanStep("s1", "find_similar_timeseries",
                              {"building_id": target, "k": limit,
                               **({"candidate_ids": "$s0.column.building_id"}
                                  if cohort else {}),
                               # "Most similar profile" is a distance. A
                               # correlation asks whether two buildings rise
                               # and fall together, which two buildings of
                               # very different size do.
                               "metric": ("correlation"
                                          if re.search(r"\bcorrelat\w+", asked)
                                          else "euclidean")},
                              depends_on=(["s0"] if cohort else []),
                              purpose="nearest load shapes"))
        elif tag == "prioritize" and known_facts is not None \
                and known_facts.covers(PRIORITY_METHODS, minimum=5):
            # The evidence this question needs has already been computed, so
            # read it instead of re-running four specialists.
            from ..knowledge.probe import method_column
            a, b, c = (nxt(n) for n in range(2, 5))
            plan.add(PlanStep(a, "lookup_derived_facts",
                              {"building_ids": ids, "as_evidence": True},
                              depends_on=prefix, resolution="cached_fact",
                              purpose="read previously computed evidence"))
            plan.add(PlanStep(b, "prioritize_entities",
                              {"evidence_handle": f"${a}.handle",
                               "criteria": {method_column(PRIORITY_METHODS[0]): 0.6,
                                            method_column(PRIORITY_METHODS[1]): 0.4},
                               "limit": limit},
                              depends_on=[a], resolution="cached_fact",
                              purpose="apply the same policy to cached evidence"))
            plan.add(PlanStep(c, "generate_decision_support",
                              {"priorities_handle": f"${b}.handle",
                               "question": intent.question, "limit": limit},
                              depends_on=[b],
                              purpose="recommendation with evidence and limits"))
            plan.notes.append(
                "evidence served from the knowledge layer; four specialist "
                "computations avoided")
        elif tag == "prioritize":
            a, b, c, d, e = (nxt(n) for n in range(2, 7))
            plan.add(PlanStep(a, "calculate_normalized_performance",
                              {"building_ids": ids, "metric": "eui", "year": year},
                              depends_on=prefix, purpose="intensity evidence"))
            plan.add(PlanStep(b, "calculate_baseload",
                              {"building_ids": ids, "method": "night_mean"},
                              depends_on=prefix, purpose="baseload evidence"))
            plan.add(PlanStep(c, "combine_evidence",
                              {"sources": [f"${a}.handle", f"${b}.handle"],
                               "labels": ["perf", "base"]},
                              depends_on=[a, b],
                              purpose="fan-in from several specialists"))
            plan.add(PlanStep(d, "prioritize_entities",
                              {"evidence_handle": f"${c}.handle",
                               "criteria": {"perf.eui": 0.6, "base.baseload_ratio": 0.4},
                               "limit": limit},
                              depends_on=[c], purpose="apply a stated policy"))
            plan.add(PlanStep(e, "generate_decision_support",
                              {"priorities_handle": f"${d}.handle",
                               "question": intent.question, "limit": limit},
                              depends_on=[d],
                              purpose="recommendation with evidence and limits"))
        else:  # consumption statistics
            # Answer the statistic that was asked for rather than returning the
            # whole summary table when the question wants one number.
            lowered = intent.question.lower()
            wanted = [name for pattern, name in (
                (r"\btotal|overall|sum\b", "total"),
                (r"\baverage|mean\b", "mean"),
                (r"\bmedian\b", "median"),
                (r"\bmaximum|max\b", "max"),
                (r"\bminimum|min\b", "min"),
                (r"\b95th|p95\b", "p95"),
                (r"\bvariab|deviation|spread\b", "stddev"),
            ) if re.search(pattern, lowered)]
            # "Which building recorded the minimum demand on 1 January" names
            # one building, and the summary answered with all 1,529 for the
            # reader to scan. When the question asks which building holds an
            # extreme of demand, ask for the extreme: one ranked row is the
            # answer, and reading it off an unordered table has named the wrong
            # building before.
            extreme = ("min_demand" if re.search(r"\bminimum|\blowest|\bmin\b",
                                                 lowered)
                       else "peak_demand" if re.search(
                           r"\bmaximum|\bhighest|\bpeak\b|\bmax\b", lowered)
                       else None)
            if intent.output_format == "entity" and extreme:
                plan.add(PlanStep(
                    nxt(2), "calculate_demand_metrics",
                    {"building_ids": ids, "start": start, "end": end,
                     "order_by": extreme,
                     "order": "asc" if extreme == "min_demand" else "desc",
                     "limit": 1},
                    depends_on=prefix,
                    purpose=f"the one building at the {extreme} of the period"))
            else:
                plan.add(PlanStep(nxt(2), "calculate_consumption_statistics",
                                  {"building_ids": ids, "start": start, "end": end,
                                   "metrics": wanted or None,
                                   **({"across": GROUPS[intent.group_by]}
                                      if intent.group_by in GROUPS else {}),
                                   **({"grain": "daily"} if re.search(
                                       r"\bday[- ]to[- ]day\b|\bdaily\b", lowered)
                                      else {})},
                                  depends_on=prefix,
                                  purpose=("requested statistic only" if wanted
                                           else "summary statistics")))
        return plan.fill_agents()


PLANNER_SYSTEM = """You are the Planning agent of EnergyX, a building-electricity \
analytics system. Convert the user's question into a JSON plan of tool calls.

Rules:
- Respond with JSON only: {"steps": [...], "notes": [...]}
- Each step: {"id", "tool", "args", "depends_on", "resolution", "purpose"}
- resolution is one of graph_only, cached_fact, hybrid, compute. Use graph_only \
for catalog/metadata questions that need no measurements.
- Refer to earlier results with "$<step>.handle" for a whole table, or \
"$<step>.<column>" to pass one column onward (e.g. "$1.building_id" hands the \
building_id column of step 1 to the next tool). A list parameter needs a column \
reference, never a handle.
- Never invent building ids, cities, years or enum values; use only the snapshot \
facts and the parameter enums given below.
- Resolve the eligible building set first with find_buildings, then analyse.
- Use only the listed tools and their declared parameters.

Answer from the catalog, not from meter readings, whenever the question is about
what the data *contains* rather than what it *shows*:
- "what is in this dataset", "what does it cover", scope, date range, resolution \
-> search_catalog(entity="datasets")
- "how many / list buildings", optionally filtered by site, city, country, type \
-> find_buildings
- a tally grouped "per site" / "by country" / "for each city" \
-> search_catalog(entity="sites"|"countries"|"cities"|"building_types")
- "which sites / cities / countries / building types exist" \
-> search_catalog with that entity
A threshold has an edge: "exceeding", "more than" and "over" are strictly
greater, while "at least" and "or more" include the value. Set the bound
parameter where a tool offers one.
"Assess metadata completeness" is a measurement, not a listing: use
profile_metadata_completeness, which reports the share recorded per field.
Returning the buildings and their columns leaves the reader to count.
"Complete metadata" names no fields and they differ a great deal in coverage,
so say which with has_metadata rather than guessing what "complete" meant.
A region -- Europe, North America -- is several countries and means their union.
The resolved intent carries them under country_all; pass them to find_buildings
as `countries`, one query, not one query per country.
"Continuous" and "uninterrupted" are two conditions, not one. A stream can run
for twelve months and be missing a fortnight in the middle, so a question asking
for continuous data needs the span *and* a limit on the break: pass
min_months_of_data together with min_coverage and gap_at_most_hours. Span alone
matched 1,511 buildings where 628 actually qualify.

A question asking to find or list *all* of something needs a limit above the
default 500, or the answer is a truncated set presented as a complete one.
To analyse every building, omit building_ids entirely. Never pass an empty
list: that is what a step whose upstream matched nothing produces, and it now
returns EMPTY rather than silently analysing the whole portfolio.
A comparison naming two cohorts -- "offices versus schools", "education vs
office buildings" -- carries both in the resolved intent under a *_all key.
Plan a step for each cohort and compare the results. A plan that filters on
only one of them answers half the question and reads as though it answered
all of it.
- "what years / what period / what date range does this cover" \
-> search_catalog(entity="years"), which returns one row per calendar year
These cost nothing and need no measurements: mark them resolution="graph_only".
Never answer a counting or listing question by computing statistics over meter
data -- that scans millions of rows to produce a number the catalog already has.

fetch_electricity, fetch_weather and fetch_contextual_data return the raw series
for a later step to work on. They are never the last step of a plan: their first
row is one period, not a summary, and presenting it as the answer to "what was
the total in 2017" reports a single day. A question asking for one number
over a period ends in a tool that aggregates:
A question that asks to plot, chart, graph, visualise or "show the shape of"
something ends with create_visualization, taking the handle of the step that
produced the data: compute first, then draw. Without that step the question is
answered with a table of numbers and no figure.

generate_report, create_visualization and combine_evidence assemble a deliverable
from work already done. They are never the answer to a question a single analytical
tool answers: asked "how many education buildings are available at each site", a
plan ending in generate_report returned a Markdown document containing two handles
instead of the counts. Reach for them only when the question asks for a
report, a written summary or a figure.

`end` is EXCLUSIVE everywhere. A full calendar year is start="2017-01-01",
end="2018-01-01". Writing end="2017-12-31" silently drops 31 December -- 49,522
kWh on a large building, which looks like a plausible total.
- one building across two years, or "did X use more in A than in B" ->
  compare_temporal_profiles(comparison="year_year"); do not fetch both years
  and leave the comparison to the reader
- a weather variable's own value (average temperature, highest wind speed) ->
  calculate_weather_statistics, never the electricity tools
- total, mean, median, min, max, variability -> calculate_consumption_statistics
- peak demand, load factor, peak timing      -> calculate_demand_metrics
- EUI or any area/occupant normalisation     -> calculate_normalized_performance"""


class PlannerRefused(RuntimeError):
    """The model could not produce a usable plan and no substitute is allowed.

    Raised only in strict mode. The ordinary behaviour is to fall back to the
    rule-based planner, which keeps the system answering; strict mode is for
    runs that exist to measure the model, where a substituted plan would be
    recorded as if the model had produced it.

    It carries the planner's ``meta`` because the tokens were still spent: a
    refusal reported with an empty token count is indistinguishable from a
    question the model was never asked, which is the confusion strict mode
    exists to end.
    """

    def __init__(self, message: str, meta: dict | None = None):
        super().__init__(message)
        self.meta = dict(meta or {})


class LLMPlanner:
    """Ask a model for a plan; validate it; fall back when it is unusable."""

    name = "llm"

    def _retry(self, prompt: str, problems: list[str], meta: dict):
        """Re-ask once with the validation errors quoted back."""
        correction = (prompt + "\n\nYour previous plan was rejected:\n"
                      + "\n".join(f"- {p}" for p in problems[:6])
                      + "\nEmit a corrected JSON plan using only the tool names "
                        "listed above, exactly as spelled.")
        try:
            response = self.llm.complete(correction, system=PLANNER_SYSTEM)
            meta["prompt_tokens"] = meta.get("prompt_tokens", 0) + response.prompt_tokens
            meta["completion_tokens"] = (meta.get("completion_tokens", 0)
                                         + response.completion_tokens)
            payload = response.json()
            plan = Plan(question=meta.get("question", ""),
                        notes=list(payload.get("notes", [])),
                        steps=[PlanStep(
                            id=str(s["id"]), tool=str(s["tool"]),
                            args=s.get("args", {}),
                            depends_on=s.get("depends_on", []),
                            resolution=s.get("resolution", "compute"),
                            purpose=s.get("purpose", "")) for s in payload["steps"]]
                        ).fill_agents()
            plan.notes.extend(plan.normalise())
            if plan.validate():
                return None
            meta.update(fell_back=False, corrected=True)
            return plan
        except Exception as exc:                # the fallback is the safety net
            # ...but a safety net that swallows the reason makes a degraded
            # environment indistinguishable from a code regression. Two grid
            # runs scored three points lower than their predecessor and the
            # only visible difference was `fell_back: 51` with no cause
            # recorded, which cost an hour of guessing. The reason is cheap to
            # keep and is the first thing anyone will want.
            meta.update(fell_back=True,
                        fallback_reason=f"{type(exc).__name__}: {exc}"[:200])
            return None

    def __init__(self, llm, fallback: DeterministicPlanner | None = None,
                 max_tools: int | None = None, strict: bool = False):
        self.llm = llm
        self.fallback = fallback or DeterministicPlanner()
        # Strict means the model plans or nothing does. The fallback exists so
        # a transient model failure still yields an answer, but that safety net
        # is invisible in the result: the row reads as a rule-based route with
        # no tokens, which is indistinguishable from a question the model was
        # never asked. When the point of a run is to measure the model, a
        # silent substitution is worse than an error, so strict re-raises.
        self.strict = strict
        # No cap by default. This was 48 -- the tool count at the time -- and
        # silently truncated once the registry grew to 54, hiding the whole
        # weather/spatial agent from the single-mode planner. A cap that drops
        # tools without saying so makes a planner look incapable when it was
        # never shown the tool.
        self.max_tools = max_tools
        self.last_response = None

    def _catalogue(self, mode: str, agent: str | None = None) -> str:
        """Every runtime tool, each tagged with its owning agent.

        Planning is a *core* agent: in both modes it sees the whole registry,
        because its job is to decide which specialist does what. The per-agent
        restriction is an execution-time constraint that Orchestration enforces
, not a planning-time blindfold -- a planner that cannot see the
        weather tools cannot route a step to the weather agent.
        """
        from ..tools import ensure_loaded
        ensure_loaded()          # an empty catalogue makes the model invent tools
        specs = (registry.for_agent(agent) if agent else registry.runtime_tools())
        if not specs:
            raise RuntimeError("tool registry is empty; cannot plan")
        if self.max_tools is not None and len(specs) > self.max_tools:
            dropped = [spec.name for spec in specs[self.max_tools:]]
            raise RuntimeError(
                f"max_tools={self.max_tools} would hide {len(dropped)} tool(s) "
                f"from the planner: {', '.join(dropped)}. Raise the cap or "
                f"scope the catalogue by agent instead of truncating it.")
        return "\n".join(self._compact(spec) for spec in specs)

    @staticmethod
    def _compact(spec) -> str:
        """One line per tool: signature and enums only.

        A JSON-schema dump of 48 tools is 22 KB, which crowds out the response
        itself. Names, types, enums and required flags are what the model
        actually needs to emit a valid call.
        """
        parts = []
        properties = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        for name, schema in properties.items():
            kind = schema.get("type", "string")
            if schema.get("enum"):
                kind = "|".join(schema["enum"])
            elif kind == "array":
                item = schema.get("items", {})
                # Show the enum inside a list too. "metrics:[string]" told the
                # planner nothing, so it guessed "total_kwh" -- the name the
                # aggregate tables use -- and the tool refused outright.
                kind = ("[" + "|".join(item["enum"]) + "]" if item.get("enum")
                        else f"[{item.get('type', 'string')}]")
            parts.append(f"{name}:{kind}" + ("*" if name in required else ""))
        return (f"{spec.name}({', '.join(parts)}) [{spec.agent}] "
                f"- {spec.summary}")

    def plan(self, intent: Intent, mode: str = "single", context: str = "",
             grounding: str = "", known_facts=None) -> tuple[Plan, dict]:
        if known_facts is not None and known_facts.total_facts:
            # Caching is transparent: the ordinary tools serve a warm result
            # from the knowledge store themselves. Telling the planner to call
            # lookup_derived_facts instead made it route a baseload question to
            # a fact lookup with an invented finding_type, which returned
            # nothing. That tool is for *reading the record*, not for
            # getting an answer cheaply.
            grounding = (grounding + "\n- already computed for some buildings ("
                         + known_facts.summary() + "). Plan the normal tool "
                         "anyway: it reads these automatically when they apply. "
                         "Use lookup_derived_facts only when the question asks "
                         "what has previously been recorded.")
        prompt = ((f"{grounding}\n\n" if grounding else "")
                  + f"Question: {intent.question}\n"
                  f"Resolved intent: {json.dumps(intent.to_dict(), default=str)}\n"
                  f"{context}\n"
                  f"Available tools (* = required):\n{self._catalogue(mode)}\n"
                  # The rules are in the system prompt, with a long tool
                  # catalogue between them and the point of writing. Repeating
                  # them after it costs three hundred tokens and puts them
                  # where the next token is written.
                  + f"\nThe rules again, in short:{planning_rules(intent.question)}\n\n"
                  "Return the JSON plan.")
        meta = {"planner": "llm", "question": intent.question}
        try:
            response = self.llm.complete(prompt, system=PLANNER_SYSTEM)
            self.last_response = response
            meta.update(prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        duration_s=response.duration_s,
                        provider=response.provider, model=response.model)
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                # Malformed JSON is a correctable mistake, and the machinery to
                # correct it is already here -- it just was not reachable from
                # this branch. A plan naming a tool that does not exist gets
                # re-asked with the error quoted back; a plan that is not valid
                # JSON went straight to a refusal and lost the question, on
                # every attempt, for the same character position. The model
                # cannot fix what it is not told about.
                retry = self._retry(
                    prompt, [f"your reply was not valid JSON: {exc}. "
                             f"Emit one JSON object and nothing else -- no "
                             f"prose, no trailing commas, and escape any "
                             f"quote inside a string"], meta)
                if retry is not None:
                    meta["recovered_json"] = True
                    return retry, meta
                raise
            plan = Plan(question=intent.question,
                        notes=list(payload.get("notes", [])),
                        steps=[PlanStep(
                            id=str(s["id"]), tool=str(s["tool"]),
                            args=s.get("args", {}),
                            depends_on=s.get("depends_on", []),
                            resolution=s.get("resolution", "compute"),
                            purpose=s.get("purpose", "")) for s in payload["steps"]]
                        ).fill_agents()
            repairs = plan.normalise()
            if repairs:
                plan.notes.extend(repairs)
                meta["repairs"] = repairs
            problems = plan.validate()
            if problems:
                # A plan numbered from 2 while its dependencies count from 1
                # is right about the analysis and wrong about the labels.
                # Reseat the references before treating the plan as invalid,
                # so an off-by-one does not cost the question.
                reseated = plan.reseat_dangling()
                if reseated:
                    plan.notes.extend(reseated)
                    meta.setdefault("repairs", []).extend(reseated)
                    problems = plan.validate()
            if problems:
                # One more attempt, with the errors handed back. A model that
                # names a tool that does not exist can usually pick a real one
                # when told which names are real; falling straight back to
                # keyword routing never gave it the chance.
                retry = self._retry(prompt, problems, meta)
                if retry is not None:
                    return retry, meta
                meta.update(fell_back=True, reason=f"invalid plan: {problems[:3]}")
                if self.strict:
                    raise PlannerRefused(f"invalid plan: {problems[:3]}", meta)
                return self.fallback.plan(intent), meta
            meta["fell_back"] = False
            return plan, meta
        except PlannerRefused:
            raise
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            meta.update(fell_back=True, reason=f"{type(exc).__name__}: {exc}")
            if self.strict:
                raise PlannerRefused(f"{type(exc).__name__}: {exc}", meta) from exc
            return self.fallback.plan(intent), meta


# --- hierarchical planning for multi-agent mode -----------------------------

def _safe_id(raw: str) -> str:
    """Step ids must match the reference grammar: [A-Za-z0-9_]+.

    A specialist happily returns "2-1" or "3.rank", and a reference to either is
    unparseable -- "$1.1-search.building_type" splits at the first dot and looks
    for a step called "1".
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(raw)).strip("_")
    return cleaned or "s"


def _remap_refs(value, resolve):
    """Rewrite "$step..." references after a sub-goal's steps were renamed."""
    if isinstance(value, str) and value.startswith("$"):
        body = value[1:]
        head, sep, tail = body.partition(".")
        return "$" + resolve(head) + sep + tail
    if isinstance(value, list):
        return [_remap_refs(v, resolve) for v in value]
    if isinstance(value, dict):
        return {k: _remap_refs(v, resolve) for k, v in value.items()}
    return value




AGENT_ROLES = {
    "data_discovery": "find datasets, sites and buildings; fetch electricity, "
                      "weather and joined data",
    # "Filling gaps with a model" names a model, and the router sent it to
    # ai_analytics, whose specialist owns forecasting and duly forecast --
    # predicting past the end of a series that needed its middle filled. Which
    # agent owns a task is decided by what the task does to the data, not by
    # whether a model is mentioned, so both roles say where the line falls.
    "data_quality": "coverage, gaps, duplicates, resampling, timezone, and "
                    "imputation -- filling gaps inside a series, including "
                    "with a weather-informed or other fitted model",
    "consumption": "consumption statistics, demand metrics, load profiles, "
                   "baseload in kW, operating patterns, trends; nothing per "
                   "square metre or per occupant",
    "performance": "area- and occupant-normalised performance, peer groups, "
                   "benchmarking, ranking, outliers; any measure divided by "
                   "floor area or occupants, including baseload",
    "weather_spatial": "weather relationships, change-point models, weather "
                       "normalisation, geographic rollups and comparisons",
    "ai_analytics": "forecasting beyond the end of a series, anomaly and "
                    "change detection, clustering, similarity, "
                    "representations; never filling gaps inside a series",
    "reporting": "combine evidence, prioritise, recommend, visualise, report",
    "knowledge": "query the knowledge graph, traverse relations, read cached "
                 "findings, export to Brick/Haystack/BEDES",
}

ROUTER_SYSTEM = """You are the Planning agent of EnergyX. Decompose the user's \
question into sub-goals and assign each to ONE specialist agent.

Rules:
- Respond with JSON only: {"steps": [...], "notes": [...]}
- Each step: {"id", "agent", "subgoal", "depends_on", "resolution"}
- Do NOT name tools. The specialist chooses its own tools.
- "subgoal" is one sentence saying what that specialist must produce.
- resolution is one of graph_only, cached_fact, hybrid, compute.
- Resolve the eligible building set first (data_discovery), then analyse.
- Use at most 5 steps.

A question about what the data *contains* -- scope, coverage, date range,
resolution, how many or which buildings / sites / cities / countries / building
types -- is answered wholly by data_discovery from the catalog, in ONE step,
resolution="graph_only". Do not send it to consumption, performance or quality:
computing statistics over meter readings to answer "what is in this dataset"
scans millions of rows for a number the catalog already holds.

Counting or listing what exists is data_discovery. Measuring how complete or how
sound it is, is data_quality: "assess metadata completeness", "how much is
missing", "which meters have gaps" are measurements over the catalog, not
listings of it, and data_discovery has no tool that computes a share.

Do not route a counting or listing question to the reporting agent. It assembles a
deliverable from completed work; it does not compute an answer.

A sub-goal that fetches raw series is never the last one: fetch_* returns periods,
not summaries. A question wanting one number over a period must end with a
consumption or performance sub-goal that aggregates.

Dividing a measure by floor area or by occupants is performance's, whatever the
measure is called. A *normalised* baseload is a baseload per square metre, and
the consumption agent cannot produce one -- its baseload is in kW, and a
sub-goal sent there comes back unnormalised, which is a different quantity and
compares two cohorts on the wrong footing.

Do not add a data_quality sub-goal to a question that does not ask about data
quality. Every analysis tool here applies its own coverage floor and reports
what it dropped, so profiling the meters first answers nothing the analysis
does not already handle -- and the profile, not the analysis, is then what the
run has to show for itself.

Filling gaps in a series needs two sub-goals in this order: data_discovery to
fetch the hourly series, then data_quality to put it on a regular grid and
impute. Only data_discovery can fetch, so a lone data_quality sub-goal asked to
impute has no series to work on and reaches for whatever earlier step it can
see -- a quality profile, which carries no readings."""

# Rules that hold however a plan is written, so a single planner and a
# specialist cannot drift apart. Guidance added to one prompt and not the other
# is how multi-agent truncated a 1,572-building answer at 1,000 while single
# returned all of them: the rule existed, in the prompt the specialist does not
# read.
SHARED_PLANNING_RULES = """
- Never invent building ids, years or enum values.
- To analyse every building, omit building_ids. Never pass an empty list: that
  is what a step whose upstream matched nothing produces, and it returns EMPTY
  rather than silently analysing the whole portfolio.
- A question asking to find or list *all* of something needs a limit above the
  population, not a round number. The default is 500 and the portfolio is 1,636,
  so a limit of 500 or 1000 returns a truncated set that reads as a complete one.
- "Continuous" and "uninterrupted" are three conditions, and the values are
  the definition: min_months_of_data=12, gap_at_most_hours=24 and
  min_coverage=0.99. A day is the line for a break, because a single missing
  hour is not a discontinuity, and 99% of the hours have to be there so that
  many small absences do not add up to a year that was never continuous.
  gap_at_most_hours=0 asks for a perfect record and finds 113 buildings where
  628 have a continuous year.
- "At least 12 consecutive months" is consecutive_months_coverage, NOT
  complete_year_coverage. The second means the calendar's twelve months and
  the first means any twelve in a row, so a building complete from July 2016
  through June 2017 satisfies the first and not the second; at a 95% floor the
  two differ by 28 buildings. Use complete_year_coverage only where the
  question says "calendar year", and pass the floor the question states rather
  than the default.
- "A complete year" wanted *for* something -- benchmarking, rating, a baseline
  -- is two conditions with two numbers: min_complete_days_in_year=365 and
  min_varying_days_in_year=350. The first alone keeps stuck meters, and the
  second written as 15 instead of 350 keeps them too: 327 buildings where 269
  have a year that is both complete and alive.
- A comparison naming two cohorts carries both in the resolved intent under a
  *_all key. Plan a step for each and compare the results.
- A region names several countries under country_all and means their union:
  pass them to find_buildings as `countries`, in one step.
- "Assess metadata completeness" is measured by profile_metadata_completeness,
  not by listing buildings and their columns.
- Imputation is three steps, not one: fetch_electricity, then
  resample_timeseries to put the series on a regular grid, then
  impute_missing_values. A fetched series has no empty values -- its gaps are
  rows that are not there -- so filling it straight away fills nothing and
  reports no gaps.
- "After-hours, midnight to 4 AM" is a window: calculate_baseload takes
  night_start_hour=0 and night_last_hour=3. The last hour is part of the
  window, so 4 would run to 04:59 and include an hour the question excludes.
  Unset is 00:00-05:59, a convention rather than an answer: the two windows put
  29 offices each in the top decile of baseload ratio and only 21 are the same
  buildings.
- A baseload the question defines by its hours is the level in those hours:
  method="night_mean" with the window. percentile is the low tail of the
  sample and daily_min_mean is the daily trough, and neither is what "the
  baseload during after-hours" names. Pick a percentile only where the
  question asks for one -- an unrequested percentile=0.1 changes 26 of the 29
  offices in the top decile.
- "Load factor normalised against building age and floor area percentiles" is
  a residual, not a division: calculate_normalized_performance with
  adjust_for=["year_built", "floor_area_m2"] and across="site", which ranks
  each building within its group on those characteristics, regresses the
  measure on the ranks and reports what is left. Dividing by age or by area
  answers a different question, and aggregating the raw measure per site
  answers none of it.
- "EUI normalised by inferred operating hours" is metric=
  "eui_per_operating_hour", not two steps: the tool infers each building's
  operating hours from its own daily profile -- the hours clearing the
  overnight floor by a quarter of the day's range, the same rule
  analyze_operating_pattern applies -- and divides annual EUI by those hours
  times 365. It reports operating_hours_per_day beside the quotient, because
  a building that never turns off reads as operating 24 hours a day and its
  normalised intensity falls for that reason and not an operational one.
  Asked of a portfolio nothing in the question selects, so the answer is the
  distribution and the median the tool names in the headline, not a top ten.
- "A multi-variable normalised performance index using EUI, weather and
  operating hours" is construct_performance_index with
  components=["eui", "weather_sensitivity", "operating_hours"], not a ranking
  on one of them and not three separate rankings. It z-scores each component
  across the cohort it ranks, signs them -- consumption per square metre
  counts against a building, weather sensitivity and operating hours count
  for it, because both are reasons to expect more energy -- and averages them
  with equal weights, which it reports rather than tunes.
- "What percentile does X fall into for total consumption among all office
  buildings" is benchmark_building: find_buildings for the cohort named, then
  benchmark_building with that cohort as peer_ids and metric="total_kwh" --
  the measure the question names, not the default. It reports the percentile
  out of a hundred, with the peer median and the z-score beside it.
- "A top-10% EUI compared to others of similar floor area" is
  identify_peer_outliers with peers="size_band", method="percentile",
  threshold=10 and side="high". Each building is placed among the others
  within a factor of 1.25 of its floor area; ranking the whole cohort answers
  about size as much as about intensity, and "top 10%" is one tail, so
  reporting both ends answers a question about the heaviest with a list that
  also holds the lightest.
- A weather threshold is a contrast, not a correlation: "did consumption spike
  during the heatwave above 35C" is analyze_weather_relationship with
  method="threshold_contrast" and threshold=35, which reports mean daily
  consumption on the days that passed the line against the other days in the
  window, and the ratio between them. A correlation measures the whole range
  and a building that only responds above 35C has a weak one.
- "The average outdoor temperature during the top 5% of demand hours" selects
  hours by consumption and then reads the weather over them:
  calculate_weather_statistics with during="top_demand_hours" and the
  building named. The site's average over the same window answers a different
  question -- the weather while nobody was drawing power is in it.
- "The cooling change-point temperature for site Robin" is one fit over the
  site's pooled daily consumption: fit_temperature_response with across="site"
  and site_ids. Fitting each of its buildings hands back a list, and the
  median of those is not the site's balance point.
- "Buildings built before 1980 versus those built after 2010" is two catalog
  cohorts, not a period: find_buildings with max_year_built=1979 and again
  with min_year_built=2011 -- both bounds are inclusive, so an exclusive
  "before" or "after" takes the year next to the one named -- then
  compare_building_groups, which reports each distribution and the ratio
  between their medians. Filtering the measurements to 1980 selects nothing:
  the snapshot is 2016-2017.
- "Buildings 10 years old or newer, against the older ones" is the same two
  cohorts written as an age, so the year has to be worked out before the
  filter can be: 10 years old in a question about 2017 is min_year_built=2007,
  and "the older ones" is the complement, max_year_built=2006. Read against
  2016 the same phrase is 2006/2005, so the year the question is about is what
  the age counts back from.
- "Which two cities in DIFFERENT COUNTRIES have the most similar profiles" is
  correlate_group_profiles(group_by="city", require_different="country"). The
  most similar pair is usually two cities in one country -- shared climate,
  shared calendar -- so leaving the constraint off returns the pair the
  question rules out. days="weekday" likewise: a weekday profile and an
  all-week one are different shapes.
- "Normalised baseload" is metric="baseload_per_area": the 5th percentile of
  hourly demand divided by floor area, which is the floor a building never
  drops below, per square metre. It is a level rather than a total, so no
  coverage floor applies and no year is implied -- the whole record is the
  sample. The word "normalised" on its own does not make a question about
  EUI.
- "Which buildings have a peak demand exceeding 50 W/m2" asks for a set, not
  an order: calculate_normalized_performance with metric="peak_power_density"
  and at_least=50. Ranking and taking the top ten answers a different
  question. peak_power_density is W/m2 and peak_per_area is the same quantity
  in kW/m2 -- 50 against the kW figure matches nothing and looks like an empty
  answer rather than a unit slip.
- A coverage floor is for totals, not for peaks. A part-year understates a
  year's consumption, but a peak is an observed extreme that a short record
  can only understate, so the floor defaults to 0 for peak metrics. Screening
  education buildings at 0.9 drops 167 of 589 and 29 of the 144 answers.
- An EUI is a per-year figure: kWh per square metre per year. "Which offices
  have the highest electricity use per square metre", with no period named,
  means the latest full year, 2017 -- pass year=2017. Pooling the snapshot's
  eighteen months gives a larger number that is not an intensity and reorders
  four of the top ten.
- Rank an intensity only over the buildings it applies to:
  find_buildings(requires=["eui_analysis"]). Ranking all 307 offices instead
  of the 236 with a usable floor area puts two buildings in the top ten whose
  denominator cannot carry the division.
- "Does the daily profile change between winter and summer" asks whether the
  rhythm moved, not whether there was more of it: compare_temporal_profiles
  with measure="profile_shape", which correlates the two 24-hour profiles
  after dividing each by its own mean. The ratio of means answers a different
  question -- a building that doubles its load without changing its rhythm has
  a ratio of 2 and a correlation of 1.
- "What share of consumption falls outside business hours (7am-7pm, weekdays)"
  states the working day rather than asking for one to be inferred:
  analyze_operating_pattern with business_start_hour=7, business_last_hour=18,
  business_days="weekday" and report="after_hours", which reports the
  percentage. Letting the tool infer the schedule instead answers a different
  question -- for Bear_education_Alfredo the inferred day opens at 08:00 and
  runs all week, giving 45.0% where the stated window gives 52.8%.
- business_days is not days. days narrows what the tool looks at; the business
  window narrows what counts as open inside it. An after-hours share has to
  see the weekend in order to count it, so never pass days="weekday" for one.
- "What time does demand start ramping up" is analyze_operating_pattern with
  report="start": the first hour above the overnight floor. It is not
  ramp_hour, which is the hour demand climbs fastest -- for Fox_office_Alice
  that is 04:00 while the building does not cross its occupied threshold until
  06:00, and the question asks when it starts, not where it is steepest.
- "Each weekday morning" means days="weekday" on that tool. A building that
  opens at 07:00 on Tuesday does not open on Sunday, and pooling the week
  averages the two into a ramp that happens on neither day.
- Every building in this snapshot is a commercial, non-residential one -- that
  is what the dataset is. "All commercial buildings" therefore names the whole
  portfolio, not a building_type: there is no "Commercial" type to filter on
  and asking for one matches nothing.
- Clustering shows a *shift* only when the same building appears more than
  once. cluster_timeseries takes by_season=True for that, which gives each
  building one profile per season and flags the buildings whose seasons do not
  land in the same cluster. One profile per building can only show how
  buildings differ from each other, which is not what "shifts in occupancy
  behaviour" asks.
- "Weekly load profile" is profile="weekly" on that tool: a 168-hour week, so a
  building that runs on Saturdays is a different shape from one that does not.
  The 24-hour daily profile averages the weekend into the week.
- A semester is not a season. "The fall and spring semesters" are the academic
  terms -- September to December and January to May -- not the meteorological
  seasons, which would put the boundary two months out and split the summer
  between them. compare_temporal_profiles takes comparison="named_months" with
  months_a=[9,10,11,12] and months_b=[1,2,3,4,5].
- "How peak timing shifts" asks when demand peaks, not how much there is:
  measure="peak_hour" on that tool, which reports the shift in hours. The
  default measure compares mean demand and answers with a ratio, which is a
  different question.
- "The apparent operating schedule" is analyze_operating_pattern, which infers
  the open hours. calculate_load_profile draws the 24-hour curve and leaves the
  reading of it to whoever looks at the chart, which is not an answer to "what
  is the schedule".
- lookup_derived_facts reads what an earlier run wrote. A question about what
  a building does is answered by computing it, whatever is in the store --
  "what is the operating schedule of X" is analyze_operating_pattern, not a
  finding lookup.
- A question that names a percentile means it: "the stable baseload (e.g., 5th
  percentile of demand)" is calculate_baseload with method="percentile" and
  percentile=0.05. The default estimator is the overnight mean, a different
  measure rather than a rounding of the same one.
- "The baseload for all healthcare facilities" is one figure over their pooled
  readings: calculate_baseload takes across="cohort". A row per building
  answers "which of them", which this does not ask.
- "Top-decile baseload ratio" is decile_at_least=10 on the same tool, and that
  is the whole of it. The decile ranks buildings against each other; it is not
  a percentile of one building's demand and not a threshold on the ratio.
  Writing percentile=0.9 alongside it measures each building's own 90th-busiest
  hour, which is a peak rather than a baseload and answers nothing the question
  asked.
- "How does Monday compare to Sunday" names two days:
  compare_temporal_profiles with comparison="named_days", day_a and day_b.
  weekday_weekend pools Monday with the working week and Sunday with Saturday,
  which answers a different question than the one asked.
- "At what hour does X typically peak" is the daily profile's highest hour:
  calculate_load_profile with peak_only=True. Not the hour of the single
  highest reading, which is one moment in two years, and not the mode of each
  day's peak hour -- a meter that sits flat through hours 1 to 23 makes
  midnight the daily maximum by default, and for one building here that is 362
  days out of 731.
- "Which month had the highest use", "which hour is the peakiest", "which day
  of the week" are calendar groupings: calculate_consumption_statistics takes
  across="month", "year", "hour_of_day" or "day_of_week", with order_by and
  limit=1 for the answer. Fetching the series and reading the peak off it
  answers the same question with more steps and no ordering.
- "What share of consumption falls in the top 5% of hours" is
  calculate_demand_metrics with top_hours_share_pct=5, which reports
  share_in_top_hours. It is a concentration measure, not a ranking: a flat
  building lands near 0.05 because every hour is alike, and 0.12 means those
  hours carry two and a half times their share.
- "The top 10% largest buildings" and "the bottom half" are deciles of floor
  area, not floor areas: find_buildings takes floor_area_decile_at_least=10 and
  floor_area_decile_at_most=5. Guessing a square-metre threshold instead picks
  a different cohort, and one building either side of the line moved a
  consumption ratio by two percent.
- "How does the total consumption of A compare to B" is the ratio of the sums,
  which compare_building_groups reports as total_ratio beside the median one.
  A group of 163 large buildings against 820 small ones differs in count as
  well as in size, and the medians do not say that.
- "Compare A with B on X" is compare_building_groups, which returns both
  distributions and their median ratio in one call. It compares on load_factor
  and peak_to_average_ratio as well as eui and totals -- those two need no
  floor area or occupancy, so they compare cohorts sharing no metadata. Pass
  min_coverage=0 unless the question asks for an analysable subset: the 0.9
  default drops a third of the buildings from each side.
- "Day-to-day variability" is the spread of daily totals, not of hourly
  readings: calculate_consumption_statistics takes grain="daily". The median
  standard deviation across industrial buildings is 616 at daily grain and 44
  at hourly, and only one of those answers how much a day differs from the
  next.
- Segmenting industrial buildings by subindustry is across="subindustry". It is
  recorded for 7 of the 13, so say how many buildings each segment holds: a
  figure for one building is not a segment-level finding.
- "Which building type / city / country has the highest X" is a grouped
  question: calculate_consumption_statistics takes across="building_type",
  "city", "country" or "site", with order_by and limit=1 for the winner. There
  is no need to fetch readings or loop over cohorts to compare groups.
- "For site Peacock", "for all Healthcare buildings", "across the portfolio"
  are all one number about a group: pass across="cohort". Naming a cohort is
  not the same as asking for a row per member, and "the daily load factor for
  site Peacock" came back as a ranked list of its buildings.
- Months and a window are alternatives, not companions. "Over the summer" with
  no year is months=[6,7,8] and no start or end; adding a 2017 window on top
  silently drops the other summer the question covers.
- "The mean demand for all Healthcare buildings" is one number about the
  group; "the mean demand of each Healthcare building" is one per building.
  calculate_consumption_statistics takes across="cohort" for the first and
  defaults to the second. Pooling and averaging the per-building means are
  different numbers -- 308.163 against 300.893 for that cohort -- so the
  wording decides it, not the default.
- "Over the summer", "in winter", "during December" name months, not a window:
  a season across a two-year record is two disjoint ranges that start and end
  cannot express. calculate_demand_metrics takes months -- summer is [6,7,8]
  and winter [12,1,2] for these sites, all of which are northern hemisphere.
- "The daily load factor" is each day's mean over that day's own peak, averaged
  over the days: daily_load_factor. load_factor is the window's mean over the
  window's peak, and over a season that is a different number entirely -- 0.81
  against 0.14 for site Peacock in summer, because one hot afternoon sets the
  denominator for three months.
- A question that names no period covers the whole record. Leave start and end
  out rather than filling in a year: the resolved intent says so, and narrowing
  "which office buildings have the highest peak-to-average ratio" to 2017
  changed four of the ten buildings in the answer. Only set them when the
  question names a period or the resolved intent carries one.
- "Which building has the highest/lowest X" is one building and "which
  buildings" is a list of them. Ask the tool for either: calculate_demand_metrics
  takes order_by with limit -- 1 for the singular, 10 for the plural unless the
  question names a number. Returning every building and leaving the reader to
  find the extreme answers a different question, and reading the extreme off an
  unordered table has named the wrong building before.
- A spike is one-sided. "Spikes", "surges", "unusually high" are
  direction="high" on detect_anomalies; "dips", "drops", "unusually low" are
  direction="low". The default flags both signs, which answered "are there
  suspicious spikes" with 57 readings far below the building's normal level.
- Forecasting is not imputation. forecast_timeseries predicts beyond the end of
  a series; impute_missing_values fills holes inside one. A question asking to
  fill gaps is never answered by a forecast.
- "What do missing values do to <an annual figure>" is measured by computing
  that figure, not by profiling the gaps. calculate_normalized_performance with
  coverage_adjustment="compare" reports the figure, the same figure scaled to
  the whole year, and the difference. profile_data_quality describes the gaps
  and never says what they cost.
- A question about the effect of missing data must not be asked of a cohort
  screened for having little of it. Pass min_coverage=0 and let the capability
  filter choose the population, or the answer is drawn from the buildings the
  question was not about.
- gap_longer_than_hours selects the broken streams and gap_at_most_hours the sound ones.
  "Gaps exceeding a week" is gap_longer_than_hours=168; gap_at_most_hours=168 returns the
  968 streams whose gaps are shorter, which is the opposite set.
- A threshold has an edge. "Exceeding", "more than", "over" and "above" are
  strictly greater; "at least", "minimum" and "or more" include the value.
  Three streams sit exactly on a one-week gap, so the wrong edge is a silent
  three-building error. Where a tool offers a bound parameter, set it.
- "Complete metadata" names no fields, and they differ: floor area is on all
  1,636 buildings, year built on 817, occupants on 230. Name the fields with
  has_metadata rather than assuming which ones "complete" meant.
"""

# Rules that only matter to some questions, attached only to those.
# The shared block reached 26,000 characters and 76 bullets, sent with
# every question, and each addition shifted plans for questions that had
# nothing to do with it: a clustering rule cost Q016, Q045 and Q056,
# none of which clusters anything. Scoping them is cheaper than a repair
# per drifted question, and it is the honest structure -- a rule about
# forecasting is not a rule about the portfolio.
FORECASTING_RULES = """
- The forecasters are named on `model`, and the registry holds fourteen. Beyond
  the ones listed below: "KNN" or "k-nearest neighbours" is model="knn" and
  "Tiny Time Mixers" or "TTM" is model="tiny_time_mixer", a pretrained
  forecaster like chronos. A `_weather` suffix on a model name adds the site's
  weather to it -- knn_weather, lightgbm_weather -- and
  `weather_variables=["air_temperature","dew_temperature","cloud_coverage"]`
  says which. Cloud coverage is recorded on 48% of hours, so asking for it
  roughly halves the sample a model can fit on; ask for it only when the
  question names it.
- A question about what *building metadata* adds to a forecast needs
  pooled=True on backtest_forecast_models, and the arms are named
  "lightgbm" and "lightgbm_metadata" (or knn, gradient_boosting, linear).
  Metadata does not vary hour to hour, so in a per-building model it is a
  constant column and the comparison would return identical forecasts; only a
  model fitted across buildings can use it. Add the "_identity" arm --
  "lightgbm_identity" -- when the question is whether metadata beats simply
  knowing which building it is, which is usually what it is really asking.
- A cohort backtest reports a per-building breakdown as well as the pooled
  figure, so "how does accuracy vary across these buildings" needs no extra
  step: ask for the cohort with building_ids and the spread comes back with
  the mean.- A question that asks for a forecast is answered by forecast_timeseries, not
  by fetching the history and leaving the reader to extrapolate it. "Provide a
  naive forecast for X on 2017-08-01 using the same weekday one week earlier"
  is one step: forecast_timeseries(building_id=X, model="seasonal_naive",
  cutoff="2017-08-01", horizon_hours=24). Fetching 2017-07-25 to 2017-08-01
  returns the week it would have copied and forecasts nothing.
- The forecasting question names its model, and naming a different one is a
  wrong answer whatever its numbers look like. The words map onto `model`:
    "naive", "persistence", "the same weekday one week earlier"
        -> seasonal_naive
    "a simple 7-day trailing moving average by hour of day"
        -> moving_average, window_days=7
    "trained on historical air temperature and calendar features"
        -> weather_regression (it reads the site's own weather; do not fetch
           weather yourself and do not pass it)
    "ARIMA", "exponential smoothing", "the Theta method"
        -> arima, ets, theta (sktime)
    "probabilistic", "quantify the uncertainty", "prediction interval",
    "a foundation model", "zero-shot"
        -> chronos, which returns a forecast band as well as a path
    "a machine-learning model", "gradient boosting"
        -> gradient_boosting
  Where the question names none, omit `model` and the seasonal naive baseline
  is used and reported as a default rather than a choice.
- `cutoff` is the instant the forecast starts and is exclusive: the model sees
  data strictly before it. To forecast the 24 hours of 2017-08-01, pass
  cutoff="2017-08-01" and horizon_hours=24. Passing the day *after* trains on
  the day being forecast and scores a model against its own input.
- Score a forecast only when its horizon lies inside the snapshot, which ends
  in 2017: evaluate_forecast(forecast_handle="$s1.handle") then reports MAE,
  RMSE, MAPE, MASE and the peak-timing error. A forecast beyond the snapshot
  has no ground truth and evaluate_forecast will refuse."""

ANOMALY_RULES = """
- "Does this feature improve detection", "evaluate whether", "false positive
  rate", "precision/recall" is validate_anomaly_detection, which plants
  anomalies of known kinds at known hours and scores each configuration
  against them. Counting flags cannot answer it: more flags is not better
  detection and fewer is not worse, and without labels neither number can be
  shown to be right. `configurations` is a list of detect_anomalies keyword
  mappings, each scored on the identical planted series. For "value change
  features" compare feature_set="level" against "level_and_change".
- "How sensitive is the rate to the threshold" is sweep_anomaly_sensitivity,
  not detect_anomalies run once at one of the settings being asked about. It
  takes `thresholds`, `context_hours` and `normalize` as lists and returns the
  count at each point. On the Hog education cohort the count runs from 4,906
  to zero across defensible settings, so a single figure quoted without its
  settings is a choice reported as a finding.
- detect_anomalies takes more tests than the three above. "Modified z-score"
  or "median absolute deviation" is method="modified_zscore" at threshold 3.5;
  "absolute change from the preceding hour" is method="first_difference";
  "local outlier factor", "LOF" or "distance-based" is
  method="local_outlier_factor", which also takes `context_hours` (168 by
  default) and `scale_windows` ("minmax" or "none"). A clock window is
  hour_from/hour_to and wraps midnight: 8 PM to 7 AM is hour_from=20,
  hour_to=7.- An anomaly question names its test, and the three tests answer different
  questions. Pass `method` to detect_anomalies:
    "exceeded twice its typical daily average"
        -> multiple_of_typical, threshold=2.0
    "more than 2 standard deviations from its typical daily average"
        -> zscore, threshold=2.0
    "the days whose load shape deviates most from its typical profile"
        -> daily_profile, which compares the shape of the day rather than its
           total, and is the only one that can see an ordinary-sized day run
           at the wrong hours
  Computing the mean and the standard deviation with
  calculate_consumption_statistics reports the line but never applies it: the
  question asked which days crossed it.
- "Has the baseline undergone a structural change", "a step change", "a level
  shift" is detect_change_points, not an anomaly scan. An anomaly is one odd
  day; a change point is the date after which the building is different.
- "How many times did daily consumption increase by more than 20% compared
  with the previous day" is count_step_changes(grain="daily",
  change_at_least_pct=20, direction="increase"). It is neither of the two
  above: detect_change_points finds the single persistent regime shift in a
  series, at most one per building, and detect_anomalies with
  method="first_difference" thresholds in standard deviations of that
  building's own hour-to-hour change, so the same 20% is 0.8 sigma in one
  building and 4 in another. The question states the scale; count on it.
- "Which building's load shape correlates most strongly with X's" is
  find_similar_timeseries(building_id=X, metric="correlation", k=1). Drawing
  X's own profile with calculate_load_profile answers a question about X and
  none about the portfolio."""

PATTERN_RULES = """
- "How stable are the clusters" is compare_cluster_stability, which clusters
  weekday and weekend load shapes separately and reports the silhouette of
  each arm plus the adjusted Rand index between them. Running
  cluster_timeseries twice and comparing the label numbers answers nothing:
  cluster 2 on weekdays and cluster 0 at the weekend may be the same
  buildings.
- "What explains consumption", "which combination of characteristics and
  weather", "regression coefficients" is explain_consumption. It fits the
  target against building characteristics, weather and optionally the k-means
  cluster, and reports the coefficients *and* the RMSE -- the second being
  what says whether the first explains the spread or merely describes it.
  Ranking buildings by consumption answers what the differences are and
  nothing about what accounts for them.
- find_similar_timeseries takes metric="correlation", "cosine" or "euclidean",
  and they select different buildings: cosine compares the profiles as
  vectors, correlation centres each on its own mean first, so two flat
  buildings correlate near zero and have cosine near one.- The catalog has three type dimensions finer than building_type, and a
  question naming one means that one. `building_subtype` is recorded on 1,615
  of 1,636 buildings and has 104 values (Dormitory, College Laboratory, Fire
  Station, K-12 School, Library); `industry` and `subindustry` are recorded on
  579. "How many dormitories", "college laboratories versus classrooms" and
  "primary/secondary schools" are all subtype questions, and answering them
  with building_type conflates a dormitory with a lecture hall because both
  are Education. find_buildings takes building_subtype, industry and
  subindustry; search_catalog lists them with entity="building_subtypes",
  "industries" or "subindustries" when the question asks what exists.
- To report one row per type dimension rather than per building, pass
  across="building_subtype" (or "industry", "subindustry", "building_type") to
  calculate_consumption_statistics or calculate_normalized_performance, and
  group_by=... to profile_metadata_completeness. Those tools group on it
  directly; listing every building and leaving the reader to aggregate is not
  the answer to "by subtype".
- cluster_timeseries takes `method`. "Hierarchical", "agglomerative" and
  "dendrogram" mean method="hierarchical" (Ward linkage); "average linkage",
  "complete linkage" and "single linkage" are those methods by name; anything
  else is k-means. `distance` is euclidean unless the question says cosine or
  manhattan -- and Ward and k-means are defined only for Euclidean, so a
  question asking for either under another metric is refused rather than
  quietly run under the wrong one.
- Every clustering reports silhouette, Davies-Bouldin and Calinski-Harabasz
  together, so a question naming one of them needs no special argument. They
  disagree often -- single linkage can win silhouette and lose
  Calinski-Harabasz badly, which is what chaining looks like -- and the
  disagreement is information, not a fault to be resolved.
- "Group the portfolio into load-shape archetypes" is cluster_timeseries;
  "group them into high, medium and low baseload" is not. The second names the
  bands and the measure, so it is calculate_normalized_performance with
  metric="baseload_per_area" and bands=3: clustering discovers groups, and
  this question already has them."""

# Whether the topical blocks are attached per question or to all of them.
# Scoping is the tidier design and it is *not* obviously better: the planner is
# deterministic at temperature 0, so changing any prompt changes which
# questions it gets right, and scoping moved three answers onto the right side
# and two onto the wrong one. Left as a switch so the choice is made on
# measurement rather than on taste, and set from the measurement.
# Measured, twice. The first time scoping was a wash -- three answers fixed,
# two broken -- and it was left off. After the topical blocks grew by another
# thirty lines it was measured again and is now clearly better: Q024 and Q043
# come back and nothing is lost, because the three answers scoping used to
# cost are now held by structural repairs rather than by prompt text.
SCOPE_RULES_TO_QUESTION = os.environ.get("ENERGYX_SCOPE_RULES", "1") == "1"

TOPICAL_RULES: tuple[tuple[str, str], ...] = (
    (r"\bforecast\w*|\bpredict\w*|\bmape\b|\brmse\b|\bcvrmse\b|\bmae\b"
     r"|\barima\b|\blight\s?gbm\b|\bbacktest\w*|\baccurac\w+"
     r"|\bnext\s+\d+\s*(?:hour|day|week)|\btomorrow\b|\bhorizon\b",
     FORECASTING_RULES),
    (r"\banomal\w*|\boutlier\w*|\bz[-\s]?score|\bspike|\bunusual"
     r"|\bsigma\b|\bdeviat\w+|\bchange[-\s]?point|\bstep[-\s]?change"
     r"|\bstructural\s+(?:change|break)|\bsuspicious|\birregular",
     ANOMALY_RULES),
    (r"\bcluster\w*|\barchetype|\bsilhouette|\blinkage|\bdendrogram"
     r"|\bsimilar\w*|\bcorrelat\w*\s+most|\bsubtype|\bsub[-\s]?categor"
     r"|\bindustr\w+|\bdormitor\w+|\bk[-\s]?means|\bagglomerative"
     r"|\bdavies|\bcalinski|\bembedding",
     PATTERN_RULES),
)


def _topical_only(question: str) -> str:
    """Just the topical blocks, for a prompt that already carries the core."""
    lowered = (question or "").lower()
    blocks = [block for pattern, block in TOPICAL_RULES
              if re.search(pattern, lowered)]
    return ("Rules for this kind of sub-goal:" + "".join(blocks) + "\n\n"
            if blocks else "")


def planning_rules(question: str) -> str:
    """The shared rules, plus the topical ones this question could need.

    Every bullet in the shared block is paid for by every question, and a
    planner reading forty lines about clustering before answering "how many
    buildings are in the UK" plans it differently -- observably so: unrelated
    answers move when a clustering section is added to the prompt.
    """
    lowered = (question or "").lower()
    if not SCOPE_RULES_TO_QUESTION:
        return SHARED_PLANNING_RULES + "".join(b for _, b in TOPICAL_RULES)
    extra = [block for pattern, block in TOPICAL_RULES
             if re.search(pattern, lowered)]
    return SHARED_PLANNING_RULES + "".join(extra)


SPECIALIST_SYSTEM = """You are a specialist agent of EnergyX. Turn ONE sub-goal \
into concrete tool calls using ONLY your own tools.

Rules:
- Respond with JSON only: {"steps": [...]}
- Each step: {"id", "tool", "args", "depends_on", "resolution"}
- Prefix every id with the sub-goal id so ids stay unique.
- Refer to earlier results with "$<step>.handle" or "$<step>.<column>".
- Use the columns a step actually returns; a reference to a column it does not
  return fails the whole plan.""" + SHARED_PLANNING_RULES


class HierarchicalPlanner:
    """Two-stage planning: Planning assigns sub-goals, specialists expand them.

    This is what makes single- and multi-agent modes *architecturally* different
    rather than differently labelled. In ``single`` one planner sees all 58
    tools and writes the whole plan; here the router sees none of them -- only
    agent roles -- and each specialist sees only its own subset.

    The comparison is then informative: prompts are smaller, failures are
    localised to one specialist, and tool-selection errors cannot cross agent
    boundaries.
    """

    name = "hierarchical"

    def __init__(self, llm, fallback: DeterministicPlanner | None = None,
                 strict: bool = False):
        self.llm = llm
        self.fallback = fallback or DeterministicPlanner()
        self.strict = strict
        self.calls: list[dict] = []

    def _roles(self) -> str:
        return "\n".join(f"- {name}: {role}" for name, role in AGENT_ROLES.items())

    def plan(self, intent: Intent, mode: str = "multi", context: str = "",
             grounding: str = "", known_facts=None) -> tuple[Plan, dict]:
        self.calls = []
        if known_facts is not None and known_facts.total_facts:
            grounding = (grounding + "\n- already computed for some buildings ("
                         + known_facts.summary() + "). Specialists should still "
                         "plan the normal tool: it reads these automatically. "
                         "lookup_derived_facts is for questions about what was "
                         "previously recorded, not a cheaper route to an answer.")
        prompt = ((f"{grounding}\n\n" if grounding else "")
                  + f"Question: {intent.question}\n"
                  + f"Resolved intent: {json.dumps(intent.to_dict(), default=str)}\n"
                  + f"Specialist agents:\n{self._roles()}\n\n"
                  + "Return the JSON sub-goal plan.")
        meta: dict = {"planner": "hierarchical", "stages": 1}
        try:
            response = self.llm.complete(prompt, system=ROUTER_SYSTEM)
            self._record(response, "routing")
            payload = response.json()
            steps = [PlanStep(id=str(s["id"]), agent=str(s.get("agent")),
                              subgoal=str(s.get("subgoal", "")),
                              depends_on=s.get("depends_on", []),
                              resolution=s.get("resolution", "compute"),
                              purpose=str(s.get("subgoal", "")))
                     for s in payload["steps"]]
            if not steps:
                raise ValueError("router produced no sub-goals")
        except Exception as exc:  # noqa: BLE001
            meta.update(fell_back=True, reason=f"routing: {type(exc).__name__}: {exc}")
            meta.update(self._totals())
            if self.strict:
                raise PlannerRefused(str(meta.get('reason') or 'no usable plan'), meta)
            return self.fallback.plan(intent, known_facts=known_facts), meta

        plan = Plan(question=intent.question, notes=list(payload.get("notes", [])))
        # A router sub-goal becomes several concrete steps, so its id stops
        # existing. Everything that pointed at the sub-goal has to be pointed at
        # the step that now produces its result -- the last one the specialist
        # emitted. Without this remap, "2 depends_on 1" survives expansion into a
        # plan where step 1 is gone, and the whole plan fails validation: the
        # hierarchical planner fell back on almost every multi-step question
        #.
        produced: dict[str, str] = {}
        for step in steps:
            expanded = self._expand(step, intent, grounding=grounding)
            meta["stages"] += 1
            if expanded is None:
                meta.update(fell_back=True,
                            reason=f"{step.agent} could not expand its sub-goal")
                meta.update(self._totals())
                if self.strict:
                    raise PlannerRefused(
                        str(meta.get("reason") or "no usable plan"), meta)
                return self.fallback.plan(intent, known_facts=known_facts), meta
            prefix = _safe_id(step.id)
            local = {}
            for sub in expanded:                 # namespace, so ids stay unique
                original = str(sub.id)
                sub.id = f"{prefix}_{_safe_id(original)}"
                local[original] = sub.id
                local[_safe_id(original)] = sub.id
            resolve = lambda ref: produced.get(ref) or local.get(ref) or ref
            for sub in expanded:
                sub.depends_on = [resolve(d) for d in sub.depends_on]
                sub.args = _remap_refs(sub.args, resolve)
            produced[step.id] = expanded[-1].id
            plan.steps.extend(expanded)
        # Steps carry their own sub-goal's earlier dependencies too.
        for sub in plan.steps:
            sub.depends_on = [d for d in dict.fromkeys(sub.depends_on) if d != sub.id]
        plan.fill_agents()
        repairs = plan.normalise()          # a mislabelled hint is not a failure
        if repairs:
            plan.notes.extend(repairs)
            meta["repairs"] = repairs
        problems = plan.validate()
        if problems:
            # Hand the problems back before giving up, as the single planner
            # does. A specialist that wrote building_ids=[] can pass the step
            # it depends on when told so, and falling straight to keyword
            # routing never gave it the chance -- which cost multi-agent a
            # correct answer on a question single got right.
            retry = self._reexpand(intent, steps, problems, meta, known_facts,
                                   grounding=grounding)
            if retry is not None:
                meta["fell_back"] = False
                meta["corrected"] = True
                meta.update(self._totals())
                return retry, meta
            meta.update(fell_back=True, reason=f"invalid plan: {problems[:3]}")
            meta.update(self._totals())
            if self.strict:
                raise PlannerRefused(str(meta.get('reason') or 'no usable plan'), meta)
            return self.fallback.plan(intent, known_facts=known_facts), meta
        meta["fell_back"] = False
        meta.update(self._totals())
        return plan, meta

    def _reexpand(self, intent: Intent, steps: list[PlanStep],
                  problems: list[str], meta: dict, known_facts,
                  grounding: str = "") -> Plan | None:
        """Expand the sub-goals again, with the validator's complaints attached.

        Only the specialists whose steps were named in a problem are asked
        again; the rest stood up to validation and re-rolling them would throw
        away work that was already right.
        """
        plan = Plan(question=intent.question)
        produced: dict[str, str] = {}
        for step in steps:
            complaint = "; ".join(p for p in problems
                                  if p.split(":")[0].strip().startswith(_safe_id(step.id)))
            expanded = self._expand(step, intent, complaint=complaint,
                                    grounding=grounding)
            if expanded is None:
                return None
            prefix = _safe_id(step.id)
            local = {}
            for sub in expanded:
                original = str(sub.id)
                sub.id = f"{prefix}_{_safe_id(original)}"
                local[original] = sub.id
                local[_safe_id(original)] = sub.id
            resolve = lambda ref: produced.get(ref) or local.get(ref) or ref
            for sub in expanded:
                sub.depends_on = [resolve(d) for d in sub.depends_on]
                sub.args = _remap_refs(sub.args, resolve)
            produced[step.id] = expanded[-1].id
            plan.steps.extend(expanded)
        for sub in plan.steps:
            sub.depends_on = [d for d in dict.fromkeys(sub.depends_on) if d != sub.id]
        plan.fill_agents()
        plan.notes.extend(plan.normalise())
        return None if plan.validate() else plan

    def _expand(self, step: PlanStep, intent: Intent,
                complaint: str = "", grounding: str = "") -> list[PlanStep] | None:
        specs = registry.for_agent(step.agent or "")
        if not specs:
            return None
        catalogue = "\n".join(LLMPlanner._compact(spec) for spec in specs)
        # The specialist writes the arguments, so the specialist needs to know
        # what the snapshot holds. Grounding only the router leaves periods to
        # be invented one stage below the stage that was told the real ones.
        prompt = ((f"{grounding}\n\n" if grounding else "")
                  + f"Question: {intent.question}\n"
                  # The router is given the resolved intent and the specialist
                  # was not, so the specialist re-derived the entities from the
                  # raw question -- and turned the building id
                  # Lamb_office_Bertha into city="Bertha", a city that does not
                  # exist. The names were already resolved one stage up.
                  + f"Resolved intent: {json.dumps(intent.to_dict(), default=str)}\n"
                  f"Your sub-goal ({step.id}): {step.subgoal}\n"
                  f"Earlier steps you may reference: {step.depends_on}\n"
                  f"Your tools (* = required):\n{catalogue}\n\n"
                  + (f"Your previous steps were rejected: {complaint}\n"
                     "Keep the approach and fix what was named.\n\n"
                     if complaint else "")
                  # The specialist's system prompt carries the core rules; the
                  # topical ones travel with the sub-goal that needs them.
                  + _topical_only(step.subgoal or intent.question)
                  + "Return the JSON steps.")
        try:
            response = self.llm.complete(prompt, system=SPECIALIST_SYSTEM)
            self._record(response, f"expand:{step.agent}")
            payload = response.json()
            out = []
            for raw in payload["steps"]:
                out.append(PlanStep(
                    id=str(raw["id"]), tool=str(raw["tool"]),
                    args=raw.get("args", {}), agent=step.agent,
                    depends_on=raw.get("depends_on", [])
                    or list(step.depends_on),
                    resolution=raw.get("resolution", step.resolution),
                    purpose=step.subgoal or ""))
            return out or None
        except Exception:  # noqa: BLE001
            return None

    def _record(self, response, purpose: str) -> None:
        self.calls.append({"purpose": purpose,
                           "prompt_tokens": response.prompt_tokens,
                           "completion_tokens": response.completion_tokens,
                           "duration_s": response.duration_s,
                           "provider": response.provider, "model": response.model})

    def _totals(self) -> dict:
        return {"prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
                "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
                "duration_s": sum(c["duration_s"] for c in self.calls),
                "planning_calls": len(self.calls),
                "provider": self.calls[0]["provider"] if self.calls else "?",
                "model": self.calls[0]["model"] if self.calls else "?"}
