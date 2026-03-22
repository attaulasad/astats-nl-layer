"""
Benchmark Evaluator
-------------------
Evaluates intent classification accuracy against ground-truth labels.
Reports per-query results, overall accuracy, and per-category F1.
"""

import json
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import classification_report

from astats_nl.pipeline import AStatsNLPipeline
from astats_nl.intent_classifier import INTENT_LABELS

console = Console()
QUERIES_PATH = Path(__file__).parent / "queries.json"


def run_benchmark():
    queries = json.loads(QUERIES_PATH.read_text())
    pipeline = AStatsNLPipeline()

    results_table = Table(
        title="AStats NL Layer — Benchmark Results",
        box=box.ROUNDED,
        show_lines=True,
    )
    results_table.add_column("ID",        width=4,  style="bold cyan")
    results_table.add_column("Query",     width=45)
    results_table.add_column("Expected",  width=28, style="green")
    results_table.add_column("Predicted", width=28)
    results_table.add_column("Conf.",     width=6)
    results_table.add_column("✓",         width=3)

    correct = 0
    total_with_label = 0
    true_labels = []
    pred_labels = []

    for item in queries:
        result = pipeline.run(item["query"])
        predicted  = result["intent"]["predicted_intent"]
        confidence = result["intent"]["confidence"]
        expected   = item.get("expected_intent")
        is_ambiguous_expected = item.get("expected_ambiguous", False)

        if is_ambiguous_expected:
            is_correct = result["ambiguity"]["is_ambiguous"]
            expected_display  = "AMBIGUOUS"
            predicted_display = (
                "[bold yellow]AMBIGUOUS[/bold yellow]"
                if is_correct else predicted
            )
        elif expected:
            is_correct = predicted == expected
            total_with_label += 1
            true_labels.append(expected)
            pred_labels.append(predicted)
            if is_correct:
                correct += 1
            expected_display  = expected
            predicted_display = (
                f"[bold green]{predicted}[/bold green]"
                if is_correct
                else f"[bold red]{predicted}[/bold red]"
            )
        else:
            is_correct        = False
            expected_display  = "N/A"
            predicted_display = predicted

        results_table.add_row(
            str(item["id"]),
            item["query"][:44],
            expected_display,
            predicted_display,
            f"{confidence:.0%}",
            "✓" if is_correct else "✗",
        )

    console.print(results_table)

    accuracy = correct / total_with_label if total_with_label else 0
    console.print(
        f"\n[bold cyan]Intent Classification Accuracy:[/bold cyan] "
        f"[bold green]{accuracy:.1%}[/bold green] "
        f"({correct}/{total_with_label} labelled queries)\n"
    )

    # Per-category Precision, Recall, F1
    console.rule("[bold cyan]Per-Category F1 Report[/bold cyan]")
    report = classification_report(
        true_labels, pred_labels,
        labels=INTENT_LABELS,
        target_names=[l[:30] for l in INTENT_LABELS],
        zero_division=0,
    )
    console.print(report)


if __name__ == "__main__":
    run_benchmark()
