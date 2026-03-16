"""
Ambiguity Detector
------------------
Detects underspecified or ambiguous statistical queries and
generates targeted clarification questions.

This is a critical layer: downstream test selection will fail
silently on ambiguous inputs. Detecting ambiguity early and
asking for clarification produces more reliable pipelines.

Key ambiguity types detected:
  1. Missing variable — no outcome or grouping variable identified
  2. Unclear design — cannot distinguish independent vs repeated
  3. Missing sample info — query does not clarify group sizes/structure
  4. Scope ambiguity — "significant" without specifying what
"""

from __future__ import annotations


CLARIFICATION_TEMPLATES = {
    "no_variables": (
        "Your query does not clearly specify which variables to analyse. "
        "Could you clarify: what is the outcome variable you are "
        "interested in, and what defines your groups?"
    ),
    "independent_vs_repeated": (
        "It is unclear whether participants were measured once each "
        "(independent groups) or multiple times (repeated measures). "
        "Could you confirm: were the same subjects measured under "
        "different conditions, or were different subjects in each group?"
    ),
    "no_outcome": (
        "Your query mentions a comparison but does not specify what "
        "is being measured. What is the outcome variable "
        "(e.g., score, reaction time, blood pressure)?"
    ),
    "vague_significance": (
        "Your query asks whether results are 'significant' but does "
        "not specify what comparison or relationship to test. "
        "Could you rephrase with a specific variable and group?"
    ),
    "too_short": (
        "Your query is very brief. Could you provide more detail: "
        "what variables are involved and what comparison are you "
        "trying to make?"
    ),
}

VAGUE_TERMS = [
    "significant", "results", "data", "analysis",
    "test", "check", "see", "find out", "look at",
]


def detect_ambiguity(query: str, extracted_vars: dict) -> dict:
    """
    Detect ambiguity in a statistical query.

    Args:
        query: Original user query.
        extracted_vars: Output from variable_extractor.extract_variables().

    Returns:
        dict with keys:
            - is_ambiguous (bool)
            - ambiguity_types (list[str])
            - clarification_questions (list[str])
    """
    q = query.lower().strip()
    ambiguity_types = []
    clarifications = []

    # Check 1: Too short / vague
    if len(q.split()) < 5:
        ambiguity_types.append("too_short")
        clarifications.append(CLARIFICATION_TEMPLATES["too_short"])

    # Check 2: No variables found at all
    no_outcome = extracted_vars["outcome_variable"] is None
    no_groups = len(extracted_vars["grouping_variables"]) == 0
    no_predictor = not extracted_vars["has_predictor"]

    if no_outcome and no_groups and no_predictor:
        ambiguity_types.append("no_variables")
        clarifications.append(CLARIFICATION_TEMPLATES["no_variables"])

    # Check 3: No outcome variable
    elif no_outcome and not no_groups:
        ambiguity_types.append("no_outcome")
        clarifications.append(CLARIFICATION_TEMPLATES["no_outcome"])

    # Check 4: Independent vs repeated measures unclear
    repeated_keywords = ["time", "session", "condition", "period",
                         "trial", "week", "day", "measure"]
    mentions_time = any(kw in q for kw in repeated_keywords)
    is_flagged_repeated = extracted_vars["likely_repeated_measures"]

    if mentions_time and not is_flagged_repeated and no_outcome:
        ambiguity_types.append("independent_vs_repeated")
        clarifications.append(
            CLARIFICATION_TEMPLATES["independent_vs_repeated"]
        )

    # Check 5: Vague significance queries
    if any(vague in q for vague in VAGUE_TERMS) and len(q.split()) < 8:
        if "vague_significance" not in ambiguity_types:
            ambiguity_types.append("vague_significance")
            clarifications.append(
                CLARIFICATION_TEMPLATES["vague_significance"]
            )

    return {
        "is_ambiguous": len(ambiguity_types) > 0,
        "ambiguity_types": ambiguity_types,
        "clarification_questions": clarifications,
    }
