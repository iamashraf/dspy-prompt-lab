# Guides

One step-by-step guide per example, in the style of a notebook. Each guide shows a step, the code, the real output from the reference run, and what to notice. The code blocks are exact excerpts of the example files, and a test checks that.

Run an example first, then read its guide next to the output you get. The numbers in the guides come from a local `qwen3-coder:30b` run, so yours will differ. See [results](../results.md) for why.

The order below matches the [walkthrough](../walkthrough.md).

## Foundations

| Example | Guide | Model calls |
|---|---|---|
| 01 | [Signatures and modules](01-signatures-and-modules.md) | 5 |
| 02 | [Evaluate a baseline](02-evaluate-baseline.md) | 72 |

## Demos only

| Example | Guide | Model calls |
|---|---|---|
| 09 | [LabeledFewShot](09-labeled-fewshot.md) | 72 |
| 03 | [BootstrapFewShot](03-bootstrap-fewshot.md) | 118 |
| 05 | [Save and load a compiled program](05-save-load-compiled.md) | 81 |
| 11 | [KNNFewShot](11-knn-fewshot.md) | 73 |
| 10 | [BootstrapFewShotWithRandomSearch](10-bootstrap-random-search.md) | 421 |
| 16 | [Ensemble](16-ensemble.md) | 252 |

## Instructions

| Example | Guide | Model calls |
|---|---|---|
| 04 | [MIPROv2](04-miprov2.md) | 486 |
| 12 | [COPRO](12-copro.md) | 182 |
| 15 | [InferRules](15-infer-rules.md) | 226 |
| 14 | [GEPA](14-gepa.md) | 395 |
| 13 | [SIMBA](13-simba.md) | 493 |

## Metrics and data

| Example | Guide | Model calls |
|---|---|---|
| 06 | [A metric that uses an LLM as the judge](06-custom-metric-with-llm-judge.md) | 36 |
| 07 | [Optimize a RAG pipeline](07-rag-pipeline-optimization.md) | 40 |
| 08 | [Overfitting and data splits](08-overfitting-and-splits.md) | 240 |
