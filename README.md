# dspy-prompt-lab

Learn DSPy prompt optimization by running it. This repo has sixteen small, runnable examples, one per idea or optimizer, a shared hand-written dataset, and offline tests that need no API key.

> Status: the code and tests are done. The full documentation (concepts, optimizer guide, results, debugging) is being written.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The default model runs locally on [Ollama](https://ollama.com), so no key is needed.

```bash
uv sync
cp .env.example .env        # pick a provider: ollama (default), openai, or anthropic
make test                   # offline tests, no model or network
uv run python examples/02_evaluate_baseline.py
```

Each example prints a before/after score and writes logs to `artifacts/logs/`.

## Examples

| File | Idea |
|---|---|
| `01_signatures_and_modules.py` | Signatures, `Predict`, `ChainOfThought`, a custom module |
| `02_evaluate_baseline.py` | Dataset, metric, `dspy.Evaluate`, baseline score |
| `03_bootstrap_fewshot.py` | `BootstrapFewShot` |
| `04_miprov2.py` | `MIPROv2` |
| `05_save_load_compiled.py` | Save and load a compiled program |
| `06_custom_metric_with_llm_judge.py` | LLM-as-judge metric |
| `07_rag_pipeline_optimization.py` | Tiny RAG, optimized |
| `08_overfitting_and_splits.py` | Overfitting and train/dev/test splits |
| `09`-`16` | `LabeledFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`, `COPRO`, `SIMBA`, `GEPA`, `InferRules`, `Ensemble` |

The plan is in [docs/PLAN.md](docs/PLAN.md). To contribute, read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
