import pytest
from astats_nl.intent_classifier import IntentClassifier

@pytest.fixture(scope="module")
def clf():
    return IntentClassifier()

def test_two_group_comparison(clf):
    result = clf.classify(
        "Is there a difference in scores between males and females?"
    )
    assert result["predicted_intent"] == "compare two independent groups"
    assert result["confidence"] > 0.3

def test_repeated_measures(clf):
    result = clf.classify(
        "Does performance change across the three sessions?"
    )
    assert result["predicted_intent"] in [
        "compare repeated measures or paired data",
        "compare three or more groups",
    ]

def test_regression(clf):
    result = clf.classify("What predicts recovery time?")
    assert result["predicted_intent"] == "predict outcome using regression"

def test_correlation(clf):
    result = clf.classify("Are age and income correlated?")
    assert result["predicted_intent"] == "find correlation between two variables"

def test_normality(clf):
    result = clf.classify("Is this variable normally distributed?")
    assert result["predicted_intent"] == "test normality of a distribution"

def test_output_keys(clf):
    result = clf.classify("Compare the two groups")
    assert all(k in result for k in [
        "original_query", "normalized_query",
        "predicted_intent", "confidence",
        "description", "all_scores"
    ])
