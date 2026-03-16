"""
AStats NL Layer — Demo
Run this to see the full pipeline in action on 6 sample queries.
"""

from astats_nl.pipeline import AStatsNLPipeline

pipeline = AStatsNLPipeline()

demo_queries = [
    "Is there a significant difference in blood pressure between males and females?",
    "Does reaction time change across the three testing sessions?",
    "What variables best predict patient recovery time?",
    "Are age and income correlated in this sample?",
    "significant",          # Ambiguous — should trigger clarification
    "test the data",        # Ambiguous — should trigger clarification
]

for query in demo_queries:
    result = pipeline.run(query)
    pipeline.display(result)
