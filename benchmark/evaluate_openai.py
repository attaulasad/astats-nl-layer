"""
OpenAI Backend Benchmark
Runs the same 35-query benchmark using GPT-4o-mini backend.
"""
import json
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import classification_report
from astats_nl.pipeline import AStatsNLPipeline
from astats_nl.intent_classifier import INTENT_LABELS

console = Console()
QUERIES_PATH = Path(__file__).parent / "queries.json"

def run_openai_benchmark():
    queries = json.loads(QUERIES_PATH.read_text())
    pipeline = AStatsNLPipeline(backend="openai")

    correct = 0
    total = 0
    true_labels = []
    pred_labels = []

    for item in queries:
        if item.get("expected_ambiguous"):
            continue
        result = pipeline.run(item["query"])
        predicted = result["intent"]["predicted_intent"]
        expected  = item["expected_intent"]
        true_labels.append(expected)
        pred_labels.append(predicted)
        total += 1
        if predicted == expected:
            correct += 1

    accuracy = correct / total
    console.print(f"\n[bold cyan]OpenAI GPT-4o-mini Accuracy:[/bold cyan] "
                  f"[bold green]{accuracy:.1%}[/bold green] ({correct}/{total})\n")

    report = classification_report(
        true_labels, pred_labels,
        labels=INTENT_LABELS,
        target_names=[l[:30] for l in INTENT_LABELS],
        zero_division=0,
    )
    console.print(report)

if __name__ == "__main__":
    run_openai_benchmark()
