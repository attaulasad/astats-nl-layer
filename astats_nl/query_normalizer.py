"""
Query Normalizer
----------------
Standardizes statistical terminology variants before
classification. Handles abbreviations, synonyms, and
common informal phrasings.
"""

SYNONYM_MAP = {
    # Comparison synonyms
    "is there a difference": "compare groups",
    "do groups differ": "compare groups",
    "are they different": "compare groups",
    "which is higher": "compare groups",
    "which is better": "compare groups",
    # Repeated measures synonyms
    "over time": "across sessions",
    "before and after": "repeated measures",
    "pre and post": "repeated measures",
    "longitudinal": "repeated measures",
    "same subjects": "repeated measures",
    "within subject": "repeated measures",
    # Regression synonyms
    "what drives": "predict outcome",
    "what causes": "predict outcome",
    "what influences": "predict outcome",
    "what factors": "predict outcome",
    "what predicts": "predict outcome",
    # Correlation synonyms
    "are they related": "correlation",
    "is there a relationship": "correlation",
    "do they go together": "correlation",
    "association between": "correlation",
    # Normality synonyms
    "is it normal": "normality test",
    "normally distributed": "normality test",
    "bell curve": "normality test",
    "follows normal distribution": "normality test",
}


def normalize(query: str) -> str:
    """
    Normalize a user query by replacing informal or
    variant phrasings with canonical statistical terms.

    Args:
        query: Raw user query string.

    Returns:
        Normalized query string.
    """
    normalized = query.lower().strip()
    for phrase, replacement in SYNONYM_MAP.items():
        normalized = normalized.replace(phrase, replacement)
    return normalized
