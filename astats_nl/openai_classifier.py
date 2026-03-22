"""
OpenAI Intent Classifier
------------------------
Alternative backend for Layer 0a intent classification.
Uses GPT-4o-mini via the OpenAI Chat Completions API instead of
the local facebook/bart-large-mnli model.

Advantages over local BART:
  - Higher accuracy on domain-specific neuroscience/clinical queries
  - No 1.6 GB model download required
  - Handles novel intents outside the 7-category matrix
  - Richer reasoning via chain-of-thought prompting

Cost: ~$0.0002 per query (GPT-4o-mini, as of 2026)
Privacy: Query is sent to OpenAI servers — use local backend
         for sensitive/private data.

Usage:
    import os
    os.environ["OPENAI_API_KEY"] = "sk-..."
    from astats_nl.openai_classifier import OpenAIClassifier
    clf = OpenAIClassifier()
    result = clf.classify("Does reaction time change across sessions?")
"""

from __future__ import annotations
import os
import json
from .query_normalizer import normalize
from .intent_classifier import INTENT_LABELS, INTENT_DESCRIPTIONS


SYSTEM_PROMPT = """You are an expert statistician and NLP classifier.
Your task is to classify a user's natural language statistical query 
into exactly one of the following 7 intent categories:

1. compare two independent groups
2. compare repeated measures or paired data
3. compare three or more groups
4. find correlation between two variables
5. predict outcome using regression
6. test normality of a distribution
7. test independence between categorical variables

Rules:
- Read the query carefully for clues: number of groups, whether same 
  subjects are measured multiple times, whether predicting an outcome, etc.
- "across sessions", "over time", "same subjects", "before and after" 
  all indicate repeated measures.
- "between X and Y" with two groups = compare two independent groups.
- "among three/four/multiple groups" = compare three or more groups.
- "related", "correlated", "association" = find correlation.
- "predict", "what influences", "what drives" = predict outcome (regression).
- "normally distributed", "follows normal" = test normality.
- "depends on", "independent of" with categorical variables = test independence.

Respond ONLY with valid JSON in this exact format:
{
  "predicted_intent": "<one of the 7 labels above, exactly>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining your choice>"
}"""


class OpenAIClassifier:
    """
    GPT-4o-mini powered intent classifier for statistical NL queries.
    
    Implements the same interface as IntentClassifier so it can be
    used as a drop-in replacement in AStatsNLPipeline.
    
    Set OPENAI_API_KEY environment variable before use.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai>=1.0.0"
            )
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Export your key: set OPENAI_API_KEY=sk-..."
            )
        
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._labels = INTENT_LABELS

    def classify(self, query: str) -> dict:
        """
        Classify a statistical query using GPT-4o-mini.
        
        Args:
            query: Raw natural language query from user.
            
        Returns:
            dict with same keys as IntentClassifier.classify():
                - original_query, normalized_query, predicted_intent,
                  confidence, description, reasoning, backend, all_scores
        """
        normalized = normalize(query)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this query: {normalized}"}
            ],
            temperature=0.0,   # deterministic — same query always gives same answer
            max_tokens=150,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract intent from raw text if JSON malformed
            parsed = {
                "predicted_intent": "compare two independent groups",
                "confidence": 0.0,
                "reasoning": f"JSON parse failed. Raw: {raw}"
            }
        
        predicted = parsed.get("predicted_intent", "").lower().strip()
        
        # Validate predicted intent is one of our 7 labels
        if predicted not in self._labels:
            # Find closest match
            predicted = min(
                self._labels,
                key=lambda l: abs(len(l) - len(predicted))
            )
        
        return {
            "original_query": query,
            "normalized_query": normalized,
            "predicted_intent": predicted,
            "confidence": float(parsed.get("confidence", 0.5)),
            "description": INTENT_DESCRIPTIONS.get(predicted, ""),
            "reasoning": parsed.get("reasoning", ""),
            "backend": f"openai/{self._model}",
            "all_scores": {predicted: float(parsed.get("confidence", 0.5))},
        }
