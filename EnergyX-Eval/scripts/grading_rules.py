#!/usr/bin/env python3
"""The conventions the grader applies, separated from the grader itself.

These four decisions are the whole of the tolerance policy, and each exists
because a stricter reading failed an answer that was right.
"""
from __future__ import annotations

import re

#: Relative tolerance for a continuous quantity. A mean, a correlation or a
#: fitted slope agrees when it lands within this of the reference.
TOLERANCE = 0.02

#: Keys whose value is a count, an index or a cluster number -- an integer
#: where being close is not being right.
EXACT_KEYS = ("count", "n_", "best_k", "k", "clusters", "buildings_", "n")

#: How many names an answer can reasonably be expected to recite. Beyond this
#: the question is answered by the count and a table, not by enumeration: 285
#: buildings have twelve unbroken months, and an answer that says "285
#: buildings" and hands over the list is right. Requiring the names failed it
#: at "2% of 285 named", which measured the length of the summary rather than
#: the correctness of the finding.
NAMEABLE = 25

#: Keys that describe how well a method separated the data rather than what it
#: found. A silhouette moves with the normalisation convention -- clustering
#: raw kWh and clustering mean-normalised shapes score differently on the same
#: partition -- so requiring it to match would fail an answer that agrees on
#: the finding and differs on a protocol choice. They are still checked and
#: reported; they just do not decide the verdict.
INFORMATIONAL_KEYS = ("silhouette",)

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")

#: A clock time is a label, not a quantity. "nighttime (20:00-06:59)" parses
#: as 20, 0, -6 and 59 -- the -6 because the range hyphen reads as a minus --
#: and each became a figure the answer was expected to recite.
CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")

#: A bracketed span that carries a clock time is the definition of a window,
#: not the finding: every number inside it -- the hours, the minutes and the
#: hyphen the parser reads as a minus sign -- is context.
PAREN_CLOCK = re.compile(r"\([^()]*\d{1,2}:\d{2}[^()]*\)")


def is_informational(key: str) -> bool:
    """Whether this key reports method quality rather than the finding."""
    lowered = key.lower()
    return any(term in lowered for term in INFORMATIONAL_KEYS)


def is_exact(key: str) -> bool:
    lowered = key.lower()
    return any(lowered == k or lowered.startswith(k) for k in EXACT_KEYS)


def numbers(text: str) -> list[float]:
    """Every figure in a piece of prose, with clock times removed first."""
    out = []
    for token in NUMBER.findall(CLOCK.sub(" ", text or "")):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out
