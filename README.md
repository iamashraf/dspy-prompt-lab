# dspy-prompt-lab

[![ci](https://github.com/iamashraf/dspy-prompt-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/iamashraf/dspy-prompt-lab/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Learn DSPy prompt optimization by running it. The repo has sixteen small examples, one per optimizer or idea. Each one scores a program before and after optimization, so you see what an optimizer changed and what it cost.

The examples share one hand-written dataset, run on a local Ollama model or on Anthropic, and write a log and a trace of every model call. The tests run offline. Tested with DSPy 3.4.1.

## Quick start

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/). The default model runs locally on [Ollama](https://ollama.com), so you need no API key.

```bash
git clone https://github.com/iamashraf/dspy-prompt-lab.git
cd dspy-prompt-lab
uv sync
cp .env.example .env         # choose ollama (default) or anthropic
make test                    # offline tests, no model or network
make ollama                  # starts Ollama with podman if none is running
uv run python examples/02_evaluate_baseline.py
```

Every example ends with a result line, and writes a log and a trace of every model call to `artifacts/logs/`.

```text
RESULT 03_bootstrap_fewshot: test 69.4% -> 86.1%  (62,063 prompt + 1,566 completion tokens, 175s, ollama_chat/qwen3-coder:30b)
```

Setup for each provider is in [getting started](docs/getting-started.md).

## The task

The examples route support tickets for a made-up product, Tasklane, to one of six queues. The labels follow a house policy that the model cannot guess. When a ticket mixes topics, the first matching rule wins, and `howto` means the product can do it while `feature` means it cannot. The data is 108 tickets in [`data/`](data/), split into 36 train, 36 dev and 36 test. The RAG examples use 15 help-center pages and 36 questions.

## Examples

| Example | Shows |
|---|---|
| [01](examples/01_signatures_and_modules.py) | Signatures, `Predict`, `ChainOfThought`, a custom module |
| [02](examples/02_evaluate_baseline.py) | Dataset, metric, `dspy.Evaluate`, baseline score |
| [03](examples/03_bootstrap_fewshot.py) | `BootstrapFewShot` |
| [04](examples/04_miprov2.py) | `MIPROv2` |
| [05](examples/05_save_load_compiled.py) | Save a compiled program and load it later |
| [06](examples/06_custom_metric_with_llm_judge.py) | LLM-as-judge metric |
| [07](examples/07_rag_pipeline_optimization.py) | Tiny RAG program, optimized |
| [08](examples/08_overfitting_and_splits.py) | Overfitting and train, dev, test splits |
| [09](examples/09_labeled_fewshot.py) | `LabeledFewShot` |
| [10](examples/10_bootstrap_random_search.py) | `BootstrapFewShotWithRandomSearch` |
| [11](examples/11_knn_fewshot.py) | `KNNFewShot` |
| [12](examples/12_copro.py) | `COPRO` |
| [13](examples/13_simba.py) | `SIMBA` |
| [14](examples/14_gepa.py) | `GEPA` |
| [15](examples/15_infer_rules.py) | `InferRules` |
| [16](examples/16_ensemble.py) | `Ensemble` |

## What the reference run found

From a local `qwen3-coder:30b`, on 36 test tickets that start at 69.4%. Labeled demos did best, at 86.1%. The instruction optimizers wrote general per-label rules and never found the precedence rule. One ticket is 2.8 points, so gaps under about 8 points are noise. The full table and the limits are in [results](docs/results.md).

## Documentation

- [Getting started](docs/getting-started.md): setup, providers, running in a container
- [Walkthrough](docs/walkthrough.md): run the experiments in order and see what each one shows
- [Concepts](docs/concepts.md): signatures, modules, metrics, optimizers, splits
- [Optimizers](docs/optimizers.md): what each one changes and when to pick it
- [Results](docs/results.md): numbers from the reference run and how far to trust them
- [Debugging](docs/debugging.md): logs, traces, and fixes for common problems
- [Further reading](docs/further-reading.md): DSPy docs and papers

## Layout

```text
src/dspy_prompt_lab/   provider config (lm.py), data, metrics, programs, run logs (debug.py)
examples/              the sixteen examples
data/                  tickets, help docs, questions
tests/                 offline tests
docs/                  the pages above, plus the original plan
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Report security problems as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
