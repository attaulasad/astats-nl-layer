import pytest
from astats_nl.variable_extractor import extract_variables

def test_detects_grouping_variable():
    result = extract_variables(
        "Is there a difference between males and females?"
    )
    assert len(result["grouping_variables"]) > 0

def test_detects_repeated_measures():
    result = extract_variables(
        "Does performance change over time across sessions?"
    )
    assert result["likely_repeated_measures"] is True

def test_detects_predictor():
    result = extract_variables("What variables predict recovery?")
    assert result["has_predictor"] is True

def test_no_variables_short_query():
    result = extract_variables("significant")
    assert result["outcome_variable"] is None
    assert result["grouping_variables"] == []

def test_output_keys():
    result = extract_variables("test query")
    assert all(k in result for k in [
        "outcome_variable", "grouping_variables",
        "has_predictor", "likely_repeated_measures",
        "variable_count_estimate"
    ])
