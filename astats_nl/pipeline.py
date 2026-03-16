"""
AStats NL Pipeline
------------------
Combines all NL understanding modules into a single end-to-end
pipeline:

  Raw Query
    → QueryNormalizer        (standardize terminology)
    → IntentClassifier       (what statistical test is needed?)
    → VariableExtractor      (what variables are involved?)
    → AmbiguityDetector      (is the query clear enough?)
    → Structured Output      (ready for downstream test selection)

This is the main entry point for the AStats NL layer.
"""

from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .intent_classifier import IntentClassifier
from .variable_extractor import extract_variables
from .ambiguity_detector import detect_ambiguity

console = Console()


class AStatsNLPipeline:
    """
    End-to-end NL understanding pipeline for AStats.

    Usage:
        pipeline = AStatsNLPipeline()
        result = pipeline.run("Is there a difference in scores between groups?")
        pipeline.display(result)
    """

    def __init__(self):
        console.print("[bold cyan]Loading AStats NL Pipeline...[/bold cyan]")
        self._classifier = IntentClassifier()
        console.print("[bold green]✓ Pipeline ready.[/bold green]\n")

    def run(self, query: str) -> dict:
        """
        Process a natural language statistical query through
        the full NL understanding pipeline.

        Args:
            query: Free-form statistical question from user.

        Returns:
            Structured dict with intent, variables, and ambiguity info.
        """
        # Stage 1: Intent classification
        intent_result = self._classifier.classify(query)

        # Stage 2: Variable extraction
        var_result = extract_variables(query)

        # Stage 3: Ambiguity detection
        ambiguity_result = detect_ambiguity(query, var_result)

        return {
            "query": query,
            "intent": intent_result,
            "variables": var_result,
            "ambiguity": ambiguity_result,
            "ready_for_analysis": not ambiguity_result["is_ambiguous"],
        }

    def display(self, result: dict) -> None:
        """Print a rich formatted summary of pipeline output."""

        # Header
        console.print(
            Panel(
                f"[bold white]{result['query']}[/bold white]",
                title="[bold cyan]AStats NL Layer — Query Analysis[/bold cyan]",
                border_style="cyan",
            )
        )

        # Intent table
        intent = result["intent"]
        intent_table = Table(box=box.ROUNDED, show_header=False)
        intent_table.add_column("Field", style="bold cyan", width=22)
        intent_table.add_column("Value", style="white")
        intent_table.add_row(
            "Predicted Intent", f"[bold green]{intent['predicted_intent']}[/bold green]"
        )
        intent_table.add_row(
            "Confidence", f"[bold yellow]{intent['confidence']:.1%}[/bold yellow]"
        )
        intent_table.add_row("Description", intent["description"])
        intent_table.add_row(
            "Normalized Query", f"[dim]{intent['normalized_query']}[/dim]"
        )
        console.print(intent_table)

        # Variables table
        v = result["variables"]
        var_table = Table(box=box.ROUNDED, show_header=False)
        var_table.add_column("Field", style="bold magenta", width=22)
        var_table.add_column("Value", style="white")
        var_table.add_row(
            "Outcome Variable",
            str(v["outcome_variable"]) if v["outcome_variable"] else "[dim]Not detected[/dim]",
        )
        var_table.add_row(
            "Grouping Variables",
            ", ".join(v["grouping_variables"]) if v["grouping_variables"] else "[dim]None[/dim]",
        )
        var_table.add_row(
            "Repeated Measures",
            "[bold green]Yes[/bold green]" if v["likely_repeated_measures"] else "No",
        )
        var_table.add_row(
            "Has Predictor",
            "[bold green]Yes[/bold green]" if v["has_predictor"] else "No",
        )
        console.print(var_table)

        # Ambiguity
        amb = result["ambiguity"]
        if amb["is_ambiguous"]:
            console.print(
                Panel(
                    "\n".join(
                        f"[bold yellow]?[/bold yellow] {q}"
                        for q in amb["clarification_questions"]
                    ),
                    title="[bold yellow]⚠ Clarification Needed[/bold yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    "[bold green]✓ Query is clear. Ready for statistical analysis.[/bold green]",
                    border_style="green",
                )
            )

        console.rule()
