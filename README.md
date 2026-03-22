# AStats NL Layer

Natural-language understanding front-end for **AStats** — an agentic AI system for applied statistical workflows.

This repository implements **Layer 0** of the AStats pipeline:
- **Intent classification** for statistical questions
- **Variable extraction** (outcome, grouping, predictor detection)
- **Ambiguity detection** with clarification prompts
- **Pluggable LLM backend support**:
  - Local: `facebook/bart-large-mnli`
  - Cloud: `OpenAI GPT-4o-mini`

The goal is to translate a free-form query such as:

> “Does reaction time change significantly across the 10 testing days for the same subjects?”

into a structured statistical intent that downstream layers can use for:
- data structure inference,
- assumption checking,
- test selection,
- test execution,
- and plain-language reporting.

---

## Project Status

This repository contains a working prototype of the AStats natural-language layer.

### Implemented
- Layer 0a — Intent classifier
- Layer 0b — Variable extractor
- Layer 0c — Ambiguity detector
- Local backend (`bart-large-mnli`)
- OpenAI backend (`gpt-4o-mini`)
- Real-dataset validation scripts
- Benchmark scripts for labelled query evaluation

### In Progress
- Layer 1 — Data structure inference
- Layer 2 — Assumption checking
- Layer 3 — Statistical test selection and execution

---

## Why this project?

Researchers often know **what they want to ask**, but not:
- which statistical test is appropriate,
- whether assumptions hold,
- whether the design is repeated-measures or independent-groups,
- or how to write a methods paragraph correctly.

AStats is designed to close that gap by turning natural-language statistical questions into structured, auditable analysis steps.

---

## Architecture

The full AStats pipeline is planned as four layers:

```text
User Query + Dataset
    ↓
Layer 0a: Intent Classification
Layer 0b: Variable Extraction
Layer 0c: Ambiguity Detection
    ↓
Layer 1: Data Structure Inference
Layer 2: Assumption Checking
Layer 3: Test Selection + Execution + Explanation
