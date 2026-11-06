"""Check what a question assumes, before anything is computed from it.

Every other guard in this system asks whether a *plan* is sound. This one asks
whether the *question* is, against the data actually in the snapshot, and it
exists because the failures it catches are silent by construction.

The worst of them: a building id with a typo in it resolves to nothing, the
resolver reports "no building named", and the question is answered over all
1,636 buildings. "What was the peak demand of Peacock_office_Annabelle" comes
back with the portfolio's peak -- a real number, correctly computed, answering
a question nobody asked. Nothing downstream can catch that, because by the
time a plan exists the building has already vanished from it.

Findings are graded rather than uniform:

    fatal    the question cannot be answered as asked; refuse and say why
    warning  it can be answered, but something the reader assumed is untrue
    note     worth stating in the answer, not worth stopping for

A fatal finding is a refusal, which is the honest outcome: an analyst who
mistyped a building id wants to be told, not handed a portfolio average.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# A token shaped like an identifier: at least two underscore-separated parts,
# starting with a letter. Loose on purpose -- the point is to notice that the
# question *meant* to name something, not to validate a format.
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){1,4}\b")

# Words that look like identifiers but are ours, not the catalog's.
NOT_ENTITIES = frozenset({
    "air_temperature", "dew_temperature", "cloud_coverage", "wind_speed",
    "sea_level_pressure", "precip_depth_1hr", "precip_depth_6hr",
    "building_id", "site_id", "building_type", "floor_area_m2", "year_built",
    "total_kwh", "mean_demand", "load_factor", "per_occupant",
    "seasonal_naive", "moving_average", "tiny_time_mixer", "lightgbm_weather",
    "knn_weather", "weather_regression", "gradient_boosting",
    "local_outlier_factor", "modified_zscore", "first_difference",
    "daily_profile", "multiple_of_typical", "isolation_forest",
    "heating_degree_days", "cooling_degree_days", "building_subtype",
})


@dataclass
class Finding:
    severity: str          # fatal | warning | note
    code: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "message": self.message, "suggestion": self.suggestion}


@dataclass
class PremiseReport:
    findings: list = field(default_factory=list)

    @property
    def fatal(self) -> list:
        return [f for f in self.findings if f.severity == "fatal"]

    @property
    def ok(self) -> bool:
        return not self.fatal

    def messages(self) -> list[str]:
        return [f.message for f in self.findings]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "findings": [f.to_dict() for f in self.findings]}


def check(question: str, intent, db, vocabulary=None) -> PremiseReport:
    """Everything the question assumes that the data can confirm or deny."""
    report = PremiseReport()
    _unknown_identifiers(question, intent, report, vocabulary)
    _site_agreement(intent, report, vocabulary)
    _cohort_exists(intent, report, db)
    _unmatched_category(question, intent, report, db, vocabulary)
    _period_has_data(intent, report, db)
    return report


def _unknown_identifiers(question, intent, report, vocabulary) -> None:
    """A token that names something the catalog has never heard of.

    Checked against every vocabulary, not only buildings, because a site or a
    city spelled wrong fails the same silent way.
    """
    if vocabulary is None:
        return
    known = set(vocabulary.building_ids) | set(vocabulary.sites)
    lowered = {v.lower() for v in known}
    # A token is only *trying* to be a building id when it is shaped like one:
    # this dataset names them Site_type_Name, so the first part is a site the
    # catalog holds. Without that test the check fires on every underscored
    # word a question uses -- "sub_primaryspaceusage" is a column of the
    # source data, and calling it a missing building would be a false alarm in
    # the one place a false alarm is most expensive, since a fatal finding
    # stops the answer.
    sites = {s.lower() for s in vocabulary.sites}
    for token in IDENTIFIER.findall(question or ""):
        if token in known or token.lower() in NOT_ENTITIES:
            continue
        if token.lower() in lowered:
            continue
        head = token.split("_", 1)[0].lower()
        close = difflib.get_close_matches(token, vocabulary.building_ids,
                                          n=3, cutoff=0.75)
        if head not in sites and not close:
            continue               # not an identifier this catalog would hold
        report.findings.append(Finding(
            severity="fatal", code="unknown_identifier",
            message=(f"{token!r} is not in this snapshot. A name the catalog "
                     f"does not hold resolves to nothing, and the question "
                     f"would be answered over every building instead of that "
                     f"one."),
            suggestion=(f"did you mean {', '.join(close)}?" if close
                        else f"{token.split('_')[0]} is a site here, so this "
                             f"looks like a building id; check the spelling")))


def _site_agreement(intent, report, vocabulary) -> None:
    """A question naming both a building and a site should agree with itself."""
    contradicted = (intent.entities or {}).get("site_contradicted")
    if not contradicted:
        return
    named = (intent.entities or {}).get("building_ids") or []
    mapping = getattr(vocabulary, "site_of_building", None) or {}
    actual = ", ".join(f"{b} is at site {mapping.get(b, '?')}" for b in named[:3])
    report.findings.append(Finding(
        severity="warning", code="site_mismatch",
        message=(f"the question names site {contradicted}, but {actual}"),
        suggestion=("the building id is used and the site clause ignored; "
                    "drop the site or correct it")))


# Words that sit before "building(s)" and are not meant as a type filter.
NOT_A_TYPE = frozenset({
    "commercial", "non", "nonresidential", "all", "the", "these", "those",
    "each", "every", "any", "some", "many", "multiple", "several", "other",
    "same", "similar", "comparable", "target", "reference", "single", "one",
    "two", "three", "four", "five", "ten", "individual", "such", "this",
    "that", "per", "its", "their", "our", "which", "what", "more", "most",
    "least", "best", "worst", "top", "bottom", "new", "old", "large", "small",
    "whole", "entire", "total", "average", "typical", "given", "named",
})

# "entertainment buildings", "residential buildings" -- a word offered as a
# category, immediately before the noun.
# Deliberately not "site": a word before "site" names the site, not a type,
# so "Eagle site" is correct usage and flagging it would be noise in the one
# place noise is least affordable.
CATEGORY_BEFORE_NOUN = re.compile(
    r"\b([A-Za-z][A-Za-z/&-]{2,})\s+(?:buildings?|facilit(?:y|ies)|"
    r"propert(?:y|ies)|premises)\b", re.I)


def _unmatched_category(question, intent, report, db, vocabulary) -> None:
    """A word offered as a building category that the catalog cannot honour.

    This is the failure "entertainment buildings in Europe" makes: the word
    resolves to no type, the filter silently disappears, and the question is
    answered over every building in Europe. Nothing in the resolved intent
    shows it, because the thing to notice is what is *absent* from it.

    Where the word is a *subtype* rather than a type, that is said, because
    the question is answerable with one word changed.
    """
    if vocabulary is None:
        return
    entities = intent.entities or {}
    if entities.get("building_type") or entities.get("building_subtype"):
        return
    types = {t.lower() for t in vocabulary.building_types}
    subtypes = {s.lower() for s in (vocabulary.building_subtypes or set())}
    named_places = ({s.lower() for s in vocabulary.sites}
                    | {c.lower() for c in vocabulary.cities}
                    | {c.lower() for c in vocabulary.countries})
    for word in CATEGORY_BEFORE_NOUN.findall(question or ""):
        token = word.lower()
        if token in NOT_A_TYPE or token in types or token in named_places:
            continue
        # "incorporating building metadata", "comparing building types" -- a
        # gerund before the noun is a verb phrase, not a category.
        if token.endswith("ing") and token not in types and token not in subtypes:
            continue
        # Does the catalog know it under another heading?
        try:
            rows = db.df(
                "SELECT DISTINCT building_subtype AS v FROM buildings "
                "WHERE lower(building_subtype) LIKE ? "
                "AND building_subtype IS NOT NULL", [f"%{token}%"])
            elsewhere = rows["v"].tolist()
        except Exception:                                   # noqa: BLE001
            elsewhere = []
        if elsewhere:
            report.findings.append(Finding(
                severity="warning", code="category_is_a_subtype",
                message=(f"{word!r} is not a building type here; it is a "
                         f"subtype -- {', '.join(elsewhere[:2])}"),
                suggestion=("the type filter matched nothing and was dropped, "
                            "so the question runs over every building; name "
                            "the subtype instead")))
        # Deliberately nothing for a word the catalog has never heard of.
        # "Find buildings", "largest buildings", "peer buildings" are English,
        # not categories, and flagging them buries the one finding that
        # matters -- a word that *is* a category in this catalog, under a
        # heading the question did not use.


def _cohort_exists(intent, report, db) -> None:
    """Does the cohort the question names hold any buildings at all?

    A type word the catalog does not hold resolves to nothing and the filter
    silently disappears: "entertainment buildings in Europe" becomes every
    building in Europe, because Entertainment is a *subtype* here and all ten
    of them are in the US. The count is the only thing that shows it.
    """
    entities = intent.entities or {}
    filters, params, described = [], [], []
    for key, column in (("building_type", "b.building_type_canonical"),
                        ("building_subtype", "b.building_subtype"),
                        ("industry", "b.industry"),
                        ("subindustry", "b.subindustry"),
                        ("site_id", "b.site_id"),
                        ("city", "s.city"),
                        ("country", "s.country_canonical")):
        if entities.get(key):
            filters.append(f"{column} = ?")
            params.append(entities[key])
            described.append(f"{key.replace('_', ' ')} {entities[key]}")
    if entities.get("country_all") and len(entities["country_all"]) > 1:
        filters = [f for f in filters if "country_canonical" not in f]
        params = [p for p, f in zip(params, filters + [""]) if True][:len(filters)]
        filters.append("s.country_canonical IN (SELECT unnest(?))")
        params.append(list(entities["country_all"]))
        described = [d for d in described if not d.startswith("country ")]
        described.append("countries " + ", ".join(entities["country_all"]))
    if len(filters) < 2:
        return                    # one filter alone is checked well enough elsewhere
    try:
        found = int(db.df(
            "SELECT COUNT(*) AS n FROM buildings b JOIN sites s USING (site_id) "
            "WHERE " + " AND ".join(filters), params).iloc[0]["n"])
    except Exception:             # noqa: BLE001
        return
    wanted = " and ".join(described)
    if found == 0:
        report.findings.append(Finding(
            severity="fatal", code="empty_cohort",
            message=f"no building matches {wanted}",
            suggestion="one of those filters holds nothing in this snapshot; "
                       "check it against the catalog"))
    elif found < 3:
        report.findings.append(Finding(
            severity="warning", code="tiny_cohort",
            message=f"only {found} building matches {wanted}",
            suggestion="a cohort this small supports description, not a "
                       "comparison or a distribution"))


def _period_has_data(intent, report, db) -> None:
    """Does the cohort the question names hold readings in the period it names?

    A window with no data is answered "no results matched", which reads as a
    fact about the buildings rather than about the window -- and a window with
    *some* data is answered without saying how little.
    """
    window = intent.time_range or {}
    start, end = window.get("start"), window.get("end")
    if not start:
        return
    entities = intent.entities or {}
    named = entities.get("building_ids") or []
    where, params = ["timestamp_local >= ?"], [start]
    if end:
        where.append("timestamp_local < ?"); params.append(end)
    if named:
        where.append("building_id IN (SELECT unnest(?))"); params.append(list(named))
    elif entities.get("site_id"):
        where.append("site_id = ?"); params.append(entities["site_id"])
    else:
        return                     # a portfolio-wide window is always populated
    try:
        row = db.df(
            "SELECT COUNT(*) AS hours, COUNT(DISTINCT building_id) AS buildings "
            "FROM electricity WHERE " + " AND ".join(where)
            + " AND value IS NOT NULL", params).iloc[0]
    except Exception:              # noqa: BLE001
        return
    hours, buildings = int(row["hours"]), int(row["buildings"])
    subject = (", ".join(named[:2]) if named else f"site {entities['site_id']}")
    if hours == 0:
        report.findings.append(Finding(
            severity="fatal", code="empty_period",
            message=(f"{subject} has no readings between {start} and "
                     f"{end or 'the end of the record'}"),
            suggestion="check the dates against the snapshot's coverage"))
        return
    expected = _expected_hours(start, end)
    if expected and hours < 0.5 * expected * max(buildings, 1):
        report.findings.append(Finding(
            severity="warning", code="thin_period",
            message=(f"{subject} holds {hours:,} readings in that window, "
                     f"where a complete record would hold about "
                     f"{expected * max(buildings, 1):,}"),
            suggestion="the answer covers the hours that exist, not the window"))


def _expected_hours(start: str, end: str | None) -> int:
    from datetime import date
    if not end:
        return 0
    try:
        opens = date.fromisoformat(str(start)[:10])
        closes = date.fromisoformat(str(end)[:10])
    except ValueError:
        return 0
    return max((closes - opens).days, 0) * 24
