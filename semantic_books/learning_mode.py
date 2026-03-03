"""Learning mode heuristics for theory/practical filtering."""

from __future__ import annotations

import re
from typing import Dict, List


MODE_THEORY = "theory"
MODE_PRACTICAL = "practical"
MODE_BALANCED = "balanced"
MODE_UNKNOWN = "unknown"

THEORY_PATTERNS = [
    r"\btheory\b",
    r"\btheoretical\b",
    r"\bmathematical\b",
    r"\bproof\b",
    r"\bfoundation(?:al)?\b",
    r"\bprinciples?\b",
    r"\badvanced\b",
]

PRACTICAL_PATTERNS = [
    r"\bpractical\b",
    r"\bhands[\s-]?on\b",
    r"\bworkshop\b",
    r"\bproject(?:s)?\b",
    r"\btutorial\b",
    r"\bguide\b",
    r"\bexample(?:s)?\b",
    r"\bimplementation\b",
    r"\bbuild\b",
]


def _count_matches(text: str, patterns: List[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def infer_learning_mode(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return MODE_UNKNOWN

    theory_score = _count_matches(normalized, THEORY_PATTERNS)
    practical_score = _count_matches(normalized, PRACTICAL_PATTERNS)

    if theory_score == 0 and practical_score == 0:
        return MODE_UNKNOWN
    if theory_score > practical_score:
        return MODE_THEORY
    if practical_score > theory_score:
        return MODE_PRACTICAL
    return MODE_BALANCED


def learning_mode_labels() -> Dict[str, str]:
    return {
        MODE_THEORY: "Theory",
        MODE_PRACTICAL: "Practical",
        MODE_BALANCED: "Balanced",
        MODE_UNKNOWN: "Unknown",
    }

