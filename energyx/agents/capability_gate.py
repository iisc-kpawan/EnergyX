"""Refuse what the snapshot cannot support, citing what it actually has.

gated questions by *fuel*: gas, water, steam are not in an electricity
snapshot. This gates them by *capability* -- resolution, metering level, span,
dataset count -- which is the same class of failure and the larger one. Asked
for 15-minute data, the system computed hourly statistics for 490 buildings and
returned them as the answer; asked for circuit-level submetering, it answered
about whole-building meters. Both look exactly like correct answers.

Every limit is read from the catalog, never hard-coded, so a snapshot that
gains finer data stops refusing without a code change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "15-minute", "15 min", "sub-hourly", "minute-level", "1-second"
SUB_HOURLY = re.compile(
    r"\b(\d+)[\s-]*(?:minute|min|second|sec)s?\b(?![\w-]*ly)"
    r"|\bsub-?hourly\b|\bminute[\s-]level\b|\bsecond[\s-]level\b"
    r"|\bhigh[\s-]frequency\b", re.I)
SUBMETER = re.compile(
    r"\bcircuit[\s-]level\b|\bsub-?meter(?:ing|ed|s)?\b|\bappliance[\s-]level\b"
    r"|\bend[\s-]use\s+(?:meter|submeter|breakdown)|\bplug[\s-]load\s+meter"
    r"|\bpanel[\s-]level\b|\bzone[\s-]level\b|\bfloor[\s-]level\b", re.I)
SECOND_DATASET = re.compile(
    r"\b(?:second|another|other|additional|external)\s+"
    r"(?:registered\s+)?dataset\b|\bacross\s+datasets\b|\bbetween\s+datasets\b"
    r"|\bcross-?dataset\b", re.I)
# Questions whose answer needs ground truth this snapshot does not carry. These
# are not out of scope by fuel and not blocked by resolution or metering
# level: the measurements exist, but nothing in the snapshot can settle
# the claim being asked for. Answering anyway substitutes a proxy -- a
# weekday/weekend demand ratio offered as evidence of occupancy -- and presents
# it as the answer.
NEEDS_GROUND_TRUTH = (
    (r"\b(?:reliabl[ey]|accurat\w+|valid\w*|confiden\w+|trustworth\w+)\b"
     r"[\w\s,]{0,40}\b(?:infer\w*|estimat\w*|predict\w*|detect\w*)\b"
     r"|\b(?:infer\w*|estimat\w*|predict\w*)\b[\w\s,]{0,40}"
     r"\b(?:reliabl[ey]|accurat\w+|confiden\w+)\b",
     "verification against ground truth",
     "this snapshot holds electricity measurements and building metadata only. "
     "Whether an inference is reliable can only be settled against observed "
     "ground truth -- occupancy records, sub-metering or a labelled set -- and "
     "none is present"),
    (r"\b(?:demand\s+flexibilit\w+|shiftable\s+(?:load|demand|kw)\b"
     r"|\bcurtailab\w+|\bdemand\s+response\s+potential"
     r"|\bflexible\s+(?:load|capacity))\b",
     "flexibility_baseline",
     "quantifying shiftable or curtailable load needs equipment, control or "
     "contractual baselines -- what may be shifted, by how much and for how "
     "long. This snapshot meters whole-building electricity only"),
)

YEARS_REQUIRED = re.compile(
    r"\bat\s+least\s+(\d+)\s+(?:full\s+|complete\s+|continuous\s+)*years?\b"
    r"|\b(\d+)\+?\s+years?\s+of\s+(?:continuous\s+|complete\s+)*data\b", re.I)


@dataclass(frozen=True)
class SnapshotLimits:
    """What this snapshot can actually support, read from the catalog."""

    finest_minutes: int
    levels: frozenset
    datasets: int
    max_span_days: int
    first_year: int = 0
    last_year: int = 0
    empty_fields: frozenset = frozenset()

    @classmethod
    def from_db(cls, db) -> "SnapshotLimits":
        row = db.df("""
            SELECT MIN(resolution_minutes) AS finest,
                   MAX(DATE_DIFF('day', start_timestamp, end_timestamp)) AS span
            FROM electricity_streams""").iloc[0]
        levels = db.df("SELECT DISTINCT stream_level AS l FROM electricity_streams")
        datasets = db.df("SELECT COUNT(*) AS n FROM datasets")["n"].iloc[0]
        years = db.df("""SELECT MIN(YEAR(period_start)) AS lo,
                                MAX(YEAR(period_start)) AS hi
                         FROM agg_yearly""").iloc[0]
        # A column can be declared and hold nothing. climate_zone is on every
        # site and null on all nineteen, so "benchmark against peers in the same
        # climate zone" had no peer set to build and was answered anyway, from a
        # grouping the question did not ask for. Read which fields are empty
        # rather than listing them, so a later snapshot that populates one stops
        # refusing without anyone remembering to edit this.
        empty = set()
        for table, columns in (("sites", ("climate_zone",)),
                               ("electricity_streams", ("end_use",))):
            for column in columns:
                populated = db.df(
                    f"SELECT COUNT({column}) AS n FROM {table}")["n"].iloc[0]
                if not int(populated):
                    empty.add(column)
        return cls(finest_minutes=int(row["finest"] or 60),
                   levels=frozenset(str(v) for v in levels["l"].dropna()),
                   datasets=int(datasets),
                   max_span_days=int(row["span"] or 0),
                   first_year=int(years["lo"] or 0),
                   last_year=int(years["hi"] or 0),
                   empty_fields=frozenset(empty))


# Phrases that need a field the snapshot may declare and never populate.
EMPTY_FIELD_PHRASES = {
    "climate_zone": r"\bclimate\s*zones?\b|\bsame climate\b",
    "end_use": r"\bend[\s-]?uses?\b|\bsub[\s-]?meter\w*\b|\bcircuit[\s-]?level\b",
}

# Events a question can name that sit outside a snapshot's window. Dated here
# rather than matched loosely, so the refusal can say which year it falls in.
ERA_PHRASES = {
    "the pandemic lockdowns": r"\bpandemic\b|\blockdowns?\b|\bcovid\b",
}
ERA_YEARS = {"the pandemic lockdowns": 2020}


def unsupported(question: str, limits: SnapshotLimits) -> tuple[str, str] | None:
    """The capability a question needs and this snapshot lacks.

    Returns ``(predicate, explanation)`` or ``None``. The explanation always
    names the value the snapshot does have, so a refusal is informative rather
    than a shrug.
    """
    lowered = question.lower()

    # A field the snapshot declares but never populates cannot support an
    # answer, and substituting a different grouping silently changes the
    # question. Name the empty field instead.
    for field, phrases in EMPTY_FIELD_PHRASES.items():
        if field in limits.empty_fields and re.search(phrases, lowered):
            return (f"{field}_unavailable",
                    f"the question needs {field.replace('_', ' ')}, which this "
                    f"snapshot declares for every row and populates for none; "
                    f"no grouping on it exists to answer from")

    # A period outside the window the snapshot holds. Naming a year it does not
    # cover is not a narrower question, it is a different one: a lockdown
    # analysis over 2016-2017 describes an event that had not happened.
    if limits.first_year and limits.last_year:
        # A year can name a building's age instead of a period of data. The
        # catalog carries year_built back to the 1800s, so "buildings built
        # before 1980" asks nothing of the measurement window, and refusing it
        # for falling outside 2016-2017 refuses a question nobody asked.
        from .planner import vintage_years            # question vocabulary
        asked = ({int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)}
                 - vintage_years(question))
        outside = {y for y in asked
                   if y < limits.first_year or y > limits.last_year}
        if asked and outside and not (asked - outside):
            return ("period_outside_snapshot",
                    f"the question asks about {sorted(outside)[0]}; this "
                    f"snapshot covers {limits.first_year} to {limits.last_year} "
                    f"and holds nothing outside it")
    for era, pattern in ERA_PHRASES.items():
        if re.search(pattern, lowered) and not (
                limits.first_year <= ERA_YEARS[era] <= limits.last_year):
            return ("period_outside_snapshot",
                    f"the question asks about {era}, which falls in "
                    f"{ERA_YEARS[era]}; this snapshot covers "
                    f"{limits.first_year} to {limits.last_year}")

    found = SUB_HOURLY.search(question)
    if found:
        asked = int(found.group(1)) if found.group(1) else 0
        if not asked or asked < limits.finest_minutes:
            unit = f"{asked}-minute" if asked else "sub-hourly"
            return ("temporal_resolution",
                    f"the question asks for {unit} data; every stream in this "
                    f"snapshot is recorded at {limits.finest_minutes}-minute "
                    f"resolution, and finer data cannot be recovered from it")

    if SUBMETER.search(question) and limits.levels == frozenset({"whole_building"}):
        return ("metering_level",
                "the question asks for circuit-, appliance- or zone-level "
                "metering; every stream in this snapshot is a whole-building "
                "meter, so no sub-building breakdown exists")

    if SECOND_DATASET.search(question) and limits.datasets < 2:
        return ("registered_datasets",
                f"the question compares datasets, and {limits.datasets} dataset "
                f"is registered in this snapshot")

    for pattern, predicate, why in NEEDS_GROUND_TRUTH:
        if re.search(pattern, question, re.I):
            return predicate, why

    found = YEARS_REQUIRED.search(question)
    if found:
        years = int(found.group(1) or found.group(2))
        have = limits.max_span_days / 365.25
        if years > have:
            return ("temporal_span",
                    f"the question requires at least {years} years of continuous "
                    f"data; the longest stream in this snapshot spans "
                    f"{limits.max_span_days} days (~{have:.1f} years)")
    return None


# Text that is program output, not a question. Pasting an error message into the
# box returned 1,572 buildings: the words "electricity" and "capability" were
# enough to route it to a catalog filter, and nothing checked that the input was
# a question at all.
NOT_A_QUESTION = (
    re.compile(r"\b(?:INSUFFICIENT_DATA|EMPTY|PARTIAL|Traceback|Exception|"
               r"Failing predicate|stack trace)\b"),
    re.compile(r"^\s*(?:OK|ERROR|WARN(?:ING)?|DEBUG|INFO)\b"),
    re.compile(r"\b\d+\.\d+s\b.*\bCannot answer\b|\bCannot answer:.*predicate"),
    re.compile(r"^[\W\d_]+$"),                      # punctuation or digits only
)


def looks_like_a_question(text: str) -> tuple[bool, str]:
    """Whether the input is something a person asked.

    Returns ``(ok, why_not)``. Deliberately permissive: a real question with odd
    grammar must pass, so this rejects only text carrying the marks of machine
    output or too little content to act on.
    """
    stripped = (text or "").strip()
    if len(stripped) < 8 or len(stripped.split()) < 3:
        return False, "the input is too short to be a question"
    for pattern in NOT_A_QUESTION:
        if pattern.search(stripped):
            return False, ("the input looks like program output rather than a "
                           "question -- it carries a status code, a predicate "
                           "name or a stack trace")
    letters = sum(c.isalpha() or c.isspace() for c in stripped)
    if letters / max(len(stripped), 1) < 0.55:
        return False, ("the input is mostly punctuation, identifiers or digits "
                       "rather than words")
    return True, ""
