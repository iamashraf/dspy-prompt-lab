# Plan

Working title: `dspy-prompt-lab`. One repo that teaches DSPy prompt optimization by running it.

## Layout

```text
src/dspy_prompt_lab/   lm.py (provider config), data.py, metrics.py, programs.py, debug.py, report.py
examples/              01..16, one idea or optimizer each, before/after score printed at the end
data/                  tickets.jsonl (classification), help_docs.jsonl + help_qa.jsonl (RAG)
tests/                 offline tests using dspy's DummyLM, no key or network
docs/                  one page per concept, with reading links
```

## Running task

A fictional product, Tasklane. The task is to route each support ticket to one of six labels. The labels follow
house rules the model cannot guess (what the product does and does not support), so an
optimizer has room to improve the score. The RAG examples use the same product's help docs.

## Examples

| # | Idea |
|---|---|
| 01 | Signatures, `Predict`, `ChainOfThought`, a custom `Module` |
| 02 | `dspy.Example`, a metric, `dspy.Evaluate`, baseline score |
| 03 | `BootstrapFewShot`, the compiled prompt, score vs baseline |
| 04 | `MIPROv2` (light), inspect optimized instructions |
| 05 | `save` / `load` of a compiled program |
| 06 | LLM-as-judge metric, when it is appropriate |
| 07 | Tiny RAG (in-memory retriever) optimized end to end |
| 08 | Overfitting a tiny trainset, train/dev/test splits |
| 09-16 | `LabeledFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`, `COPRO`, `SIMBA`, `GEPA`, `InferRules`, `Ensemble` |

## Decisions

- Pin `engine="litellm"` in `dspy.LM`. DSPy 3.4 defaults to a native `lm15` engine, which returned
  HTTP 404 for Ollama here.
- `optuna` is a dependency because `MIPROv2` imports it.
- Offline tests use `DummyLM`. Run the examples to check them.
- No dataset downloads. All data is hand-written and in `data/`.
