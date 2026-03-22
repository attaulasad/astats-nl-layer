"""
AStats — OpenAI vs Local Backend Comparison
=============================================
Runs the same 3 real-dataset queries through both backends:
  - Local:  facebook/bart-large-mnli  (free, offline)
  - OpenAI: gpt-4o-mini               (API, ~$0.0002/query)

Shows intent classification accuracy comparison.

Run:
    set OPENAI_API_KEY=sk-your-key-here       (Windows)
    python examples/openai_backend_comparison.py
"""

import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import box
from astats_nl.pipeline import AStatsNLPipeline

console = Console()

QUERIES = [
    {
        "dataset": "sleepstudy (neuroscience)",
        "query": "Does reaction time change significantly across the 10 testing days for the same subjects?",
        "expected_intent": "compare repeated measures or paired data",
    },
    {
        "dataset": "UCI Heart Disease (clinical)",
        "query": "Is there a difference in age level between patients with and without heart disease?",
        "expected_intent": "compare two independent groups",
    },
    {
        "dataset": "Iris (life sciences)",
        "query": "Do the three iris species differ significantly in sepal score?",
        "expected_intent": "compare three or more groups",
    },
]

def run_comparison():
    console.rule("[bold cyan]Loading Local Backend (bart-large-mnli)[/bold cyan]")
    local_pipeline = AStatsNLPipeline(backend="local")

    console.rule("[bold cyan]Loading OpenAI Backend (gpt-4o-mini)[/bold cyan]")
    openai_pipeline = AStatsNLPipeline(backend="openai")

    results = []
    for item in QUERIES:
        local_result  = local_pipeline.run(item["query"])
        openai_result = openai_pipeline.run(item["query"])

        local_intent   = local_result["intent"]["predicted_intent"]
        local_conf     = local_result["intent"]["confidence"]
        openai_intent  = openai_result["intent"]["predicted_intent"]
        openai_conf    = openai_result["intent"]["confidence"]
        expected       = item["expected_intent"]

        results.append({
            "dataset":        item["dataset"],
            "expected":       expected,
            "local_intent":   local_intent,
            "local_conf":     local_conf,
            "local_correct":  local_intent == expected,
            "openai_intent":  openai_intent,
            "openai_conf":    openai_conf,
            "openai_correct": openai_intent == expected,
            "reasoning":      openai_result["intent"].get("reasoning", ""),
        })

    # Print comparison table
    tbl = Table(title="Backend Comparison: BART-large-mnli vs GPT-4o-mini",
                box=box.ROUNDED)
    tbl.add_column("Dataset",         style="bold white",  width=22)
    tbl.add_column("Expected Intent", style="cyan",        width=30)
    tbl.add_column("BART Intent",     style="yellow",      width=30)
    tbl.add_column("BART Conf",       style="yellow",      width=10)
    tbl.add_column("BART ✓",          style="yellow",      width=6)
    tbl.add_column("GPT-4o-mini",     style="green",       width=30)
    tbl.add_column("GPT Conf",        style="green",       width=10)
    tbl.add_column("GPT ✓",           style="green",       width=6)

    for r in results:
        tbl.add_row(
            r["dataset"],
            r["expected"],
            r["local_intent"],
            f"{r['local_conf']:.1%}",
            "✅" if r["local_correct"] else "❌",
            r["openai_intent"],
            f"{r['openai_conf']:.1%}",
            "✅" if r["openai_correct"] else "❌",
        )
    console.print(tbl)

    # Print GPT reasoning
    console.rule("[bold cyan]GPT-4o-mini Reasoning (Chain-of-Thought)[/bold cyan]")
    for r in results:
        console.print(f"[bold]{r['dataset']}:[/bold] {r['reasoning']}")

    # Summary
    local_acc  = sum(r["local_correct"]  for r in results) / len(results)
    openai_acc = sum(r["openai_correct"] for r in results) / len(results)
    console.print(f"\n[bold]BART accuracy:[/bold]    {local_acc:.0%} ({sum(r['local_correct'] for r in results)}/{len(results)})")
    console.print(f"[bold]GPT-4o-mini accuracy:[/bold] {openai_acc:.0%} ({sum(r['openai_correct'] for r in results)}/{len(results)})")
    console.print("\n[dim]Cost estimate: ~$0.0006 total (3 queries × $0.0002)[/dim]")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]ERROR: Set OPENAI_API_KEY first.[/red]")
        console.print("[yellow]Windows: set OPENAI_API_KEY=sk-your-key[/yellow]")
    else:
        run_comparison()
