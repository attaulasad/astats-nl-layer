# AStats NL Layer

**GSoC 2026 | INCF | Project #33 — AStats**

> Natural Language Understanding front-end for the AStats agentic 
> statistical analysis system.

---

## The Problem

Before any statistical test can be selected, an agentic system must 
first correctly understand *what the user is actually asking*. This 
is harder than it looks:

| User Query | True Intent |
|---|---|
| "Is there a difference between the groups?" | Two-sample test |
| "Does the score change across sessions?" | Repeated measures |
| "What predicts recovery time?" | Regression |
| "significant" | ⚠ Ambiguous — needs clarification |

Existing tools either hard-code the query format or skip 
disambiguation entirely. **AStats NL Layer** solves this.

---

## Architecture

```
Raw User Query
    │
    ▼
┌─────────────────────┐
│  Query Normalizer   │  Standardizes statistical synonyms
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Intent Classifier  │  Zero-shot classification (bart-large-mnli)
│                     │  → 7 canonical statistical intent categories
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Variable Extractor  │  Detects outcome, grouping, predictor variables
│                     │  Flags repeated-measures designs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Ambiguity Detector  │  Identifies underspecified queries
│                     │  Generates targeted clarification questions
└─────────────────────┘
    │
    ▼
Structured Output → Ready for statistical test selection
```

---

## Intent Categories

| Category | Example Query |
|---|---|
| Compare two independent groups | "Is blood pressure different between males and females?" |
| Compare repeated measures / paired | "Did scores improve after the intervention?" |
| Compare three or more groups | "Do the four groups differ in anxiety?" |
| Find correlation | "Are age and income related?" |
| Predict outcome (regression) | "What predicts recovery time?" |
| Test normality | "Is this variable normally distributed?" |
| Test independence (chi-square) | "Does smoking status depend on education?" |

---

## Benchmark Results

Evaluated on 10 labelled queries with known ground truth:

| Metric | Score |
|---|---|
| Intent Classification Accuracy | **90%** |
| Ambiguity Detection (2 cases) | **100%** |
| Repeated Measures Detection | **100%** |

---

## Setup

```bash
git clone https://github.com/YOURUSERNAME/astats-nl-layer
cd astats-nl-layer
pip install -r requirements.txt
```

---

## Run Demo

```bash
python examples/demo.py
```

---

## Run Benchmark

```bash
python benchmark/evaluate.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Author

**Atta ul Asad**  
COMSATS University Islamabad, Vehari Campus, Pakistan  
GSoC 2026 AStats applicant — INCF Project #33  
GitHub: [your GitHub]  

---

## Related

- [AStats GSoC 2026 Project #33 — NeuroStars](https://neurostars.org/t/gsoc-2026-project-33-university-of-wisconsin-madison-astats-an-agentic-ai-approach-to-applied-statistical-practitioner-workflows/35620)
- [INCF GSoC 2026](https://www.incf.org/activities/gsoc)
