"""
Variable Extractor
------------------
Identifies the roles of variables mentioned in a statistical
query: outcome (dependent), grouping (independent), predictor,
and repeated-measures subject identifier.

Uses pattern matching and keyword heuristics — no external
model required, runs fully offline and on CPU.
"""

from __future__ import annotations
import re


# Keyword banks for role detection
OUTCOME_PATTERNS = [
    r"difference in (\w+)",
    r"effect on (\w+)",
    r"change in (\w+)",
    r"impact on (\w+)",
    r"predict (\w+)",
    r"(\w+) as outcome",
    r"(\w+) as dependent",
    r"(\w+) score",
    r"(\w+) time",
    r"(\w+) level",
    r"(\w+) rate",
]

GROUP_PATTERNS = [
    r"between (\w+) and (\w+)",
    r"across (\w+)",
    r"among (\w+)",
    r"(\w+) vs (\w+)",
    r"(\w+) versus (\w+)",
    r"in (\w+) and (\w+)",
    r"comparing (\w+)",
]

REPEATED_KEYWORDS = [
    "over time", "across sessions", "repeated measures",
    "longitudinal", "before and after", "pre and post",
    "multiple times", "same subjects", "within subject",
    "across conditions", "across time points",
]

PREDICTOR_KEYWORDS = [
    "predict", "predicts", "affects", "influences",
    "drives", "causes", "explains", "associated with",
]


def extract_variables(query: str) -> dict:
    """
    Extract variable roles from a natural language statistical query.

    Args:
        query: Raw user query string.

    Returns:
        dict with keys:
            - outcome_variable (str | None)
            - grouping_variables (list[str])
            - has_predictor (bool)
            - likely_repeated_measures (bool)
            - variable_count_estimate (str): 'one', 'two', 'multiple'
    """
    q = query.lower()

    # --- Outcome variable ---
    outcome = None
    for pattern in OUTCOME_PATTERNS:
        match = re.search(pattern, q)
        if match:
            outcome = match.group(1)
            break

    # --- Grouping variables ---
    groups = []
    for pattern in GROUP_PATTERNS:
        matches = re.findall(pattern, q)
        for m in matches:
            if isinstance(m, tuple):
                groups.extend([g for g in m if g])
            else:
                groups.append(m)
    groups = list(set(groups))

    # --- Repeated measures flag ---
    is_repeated = any(kw in q for kw in REPEATED_KEYWORDS)

    # --- Predictor present ---
    has_predictor = any(kw in q for kw in PREDICTOR_KEYWORDS)

    # --- Variable count estimate ---
    word_count = len(q.split())
    if word_count < 8:
        var_count = "one"
    elif word_count < 15:
        var_count = "two"
    else:
        var_count = "multiple"

    return {
        "outcome_variable": outcome,
        "grouping_variables": groups,
        "has_predictor": has_predictor,
        "likely_repeated_measures": is_repeated,
        "variable_count_estimate": var_count,
    }
