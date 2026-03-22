"""
Intent Classifier
-----------------
Maps free-form natural language statistical queries to one of
seven canonical statistical intent categories using zero-shot
classification (facebook/bart-large-mnli).

This module is the primary NL understanding component of the
AStats front-end pipeline.
"""

from __future__ import annotations
from transformers import pipeline as hf_pipeline
from .query_normalizer import normalize


INTENT_LABELS = [
    "compare two independent groups",
    "compare repeated measures or paired data",
    "compare three or more groups",
    "find correlation between two variables",
    "predict outcome using regression",
    "test normality of a distribution",
    "test independence between categorical variables",
]

INTENT_DESCRIPTIONS = {
    "compare two independent groups": (
        "Two separate, unrelated groups are being compared "
        "(e.g., t-test, Mann-Whitney U)."
    ),
    "compare repeated measures or paired data": (
        "Same subjects measured more than once, or matched pairs "
        "(e.g., paired t-test, Wilcoxon, Friedman)."
    ),
    "compare three or more groups": (
        "Three or more independent groups are compared "
        "(e.g., one-way ANOVA, Kruskal-Wallis)."
    ),
    "find correlation between two variables": (
        "Measuring the strength of linear/monotonic relationship "
        "(e.g., Pearson r, Spearman rho)."
    ),
    "predict outcome using regression": (
        "One or more predictors used to estimate a continuous or "
        "categorical outcome (e.g., linear regression, logistic regression)."
    ),
    "test normality of a distribution": (
        "Checking whether a variable follows a normal distribution "
        "(e.g., Shapiro-Wilk, Kolmogorov-Smirnov)."
    ),
    "test independence between categorical variables": (
        "Examining whether two categorical variables are independent "
        "(e.g., chi-square test)."
    ),
}


class IntentClassifier:
    """
    Zero-shot intent classifier for statistical NL queries.

    Uses facebook/bart-large-mnli to classify a user query
    into one of the seven canonical statistical intent categories.
    Model is loaded once at instantiation and reused.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
        )

    def classify(self, query: str) -> dict:
        """
        Classify a free-form statistical query.

        Args:
            query: Raw natural language query from user.

        Returns:
            dict with keys:
                - original_query (str)
                - normalized_query (str)
                - predicted_intent (str)
                - confidence (float)
                - description (str)
                - all_scores (dict[str, float])
        """
        normalized = normalize(query)
        result = self._classifier(normalized, INTENT_LABELS)

        top_intent = result["labels"][0]
        top_score = round(result["scores"][0], 4)

        return {
            "original_query": query,
            "normalized_query": normalized,
            "predicted_intent": top_intent,
            "confidence": top_score,
            "description": INTENT_DESCRIPTIONS[top_intent],
            "all_scores": {
                label: round(score, 4)
                for label, score in zip(result["labels"], result["scores"])
            },
        }
