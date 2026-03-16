import pytest
from astats_nl.pipeline import AStatsNLPipeline

@pytest.fixture(scope="module")
def pipeline():
    return AStatsNLPipeline()

def test_pipeline_returns_all_keys(pipeline):
    result = pipeline.run(
        "Is there a difference between the two groups?"
    )
    assert all(k in result for k in [
        "query", "intent", "variables",
        "ambiguity", "ready_for_analysis"
    ])

def test_clear_query_is_not_ambiguous(pipeline):
    result = pipeline.run(
        "Is there a difference in blood pressure between males and females?"
    )
    assert result["ready_for_analysis"] is True

def test_vague_query_is_ambiguous(pipeline):
    result = pipeline.run("significant")
    assert result["ambiguity"]["is_ambiguous"] is True
    assert len(result["ambiguity"]["clarification_questions"]) > 0
