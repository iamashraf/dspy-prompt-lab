# Optimizers

DSPy has many optimizers. This page compares the ones this repo runs and explains why it skips the rest. For a guided run, use the [walkthrough](walkthrough.md).

## Comparison

Scores are the test score on 36 tickets with a local model, from a baseline of 69.4%. See [results](results.md) for tokens, time, and why gaps under about 8 points are noise.

| Optimizer | Changes | Needs | Example | Test score |
|---|---|---|---|---|
| `LabeledFewShot` | demos | nothing | [09](../examples/09_labeled_fewshot.py) | 83.3% |
| `BootstrapFewShot` | demos | metric | [03](../examples/03_bootstrap_fewshot.py) | 80.6% bootstrapped, 86.1% labeled |
| `BootstrapFewShotWithRandomSearch` | demos | metric, dev set | [10](../examples/10_bootstrap_random_search.py) | 77.8% |
| `KNNFewShot` | demos, per input | embedding model | [11](../examples/11_knn_fewshot.py) | 83.3% |
| `Ensemble` | combines programs | several programs | [16](../examples/16_ensemble.py) | 80.6% |
| `COPRO` | instruction | metric, a model that returns several completions | [12](../examples/12_copro.py) | 66.7% |
| `InferRules` | instruction (adds rules) | metric, dev set | [15](../examples/15_infer_rules.py) | 80.6% |
| `MIPROv2` | instruction and demos | metric, `optuna` | [04](../examples/04_miprov2.py) | 77.8% |
| `SIMBA` | instruction and demos | metric returning floats, varied outputs | [13](../examples/13_simba.py) | 69.4% (no change) |
| `GEPA` | instruction | metric with feedback text, reflection model | [14](../examples/14_gepa.py) | 72.2% |

## Choosing one

1. Start with `LabeledFewShot`. It costs no compile calls. If a fancier optimizer cannot beat it, skip the fancier one.
2. If you have a hundred or more labeled examples and the right example depends on the input, use `KNNFewShot`.
3. If the model must be told rules, not just shown examples, use an instruction optimizer: `MIPROv2` for a general search, `GEPA` when you can write useful feedback text, `InferRules` when you want rules you can read.
4. If results vary between runs, use `Ensemble` or `BootstrapFewShotWithRandomSearch`.

## Notes on each

**LabeledFewShot.** Draws `k` training examples with a fixed seed, so the same trainset gives the same demos every run. To get different demos, pass a different trainset.

**BootstrapFewShot.** Runs the program on training examples and keeps the runs that pass the metric. These demos come from the model, so they show what it already does right. Add labeled demos to cover what it gets wrong.

**BootstrapFewShotWithRandomSearch.** Builds several candidates and keeps the best on the dev set. The best dev score overstates the test score, so check test.

**KNNFewShot.** Picks the nearest training examples for each input. With bootstrapped demos it re-runs a mini bootstrap for every query, about k+1 calls per input. Set `max_bootstrapped_demos=0` for one call per input.

**Ensemble.** Runs every member and votes with `dspy.majority`. It costs one call per member per input.

**COPRO.** Proposes `breadth` instructions per round for `depth` rounds. It needs several completions per call, which Ollama does not support, so on Ollama it tries one candidate per round. This repo has not run it on a hosted model.

**InferRules.** Bootstraps demos, asks a model for rules, and keeps the best rule set on the dev set. The rules appear in the instruction, so you can read them.

**MIPROv2.** Bootstraps demo sets, proposes instructions, and searches combinations with Bayesian optimization (it needs `optuna`). `auto="light"`, `"medium"` or `"heavy"` sets the budget. Recent versions removed the permission prompt, so do not pass `requires_permission_to_run`.

**SIMBA.** Samples the program several times per example and learns from examples where the outputs differ. If the model gives the same answer every time, SIMBA has nothing to learn from. Its metric must return a float.

**GEPA.** A reflection model reads your metric's feedback text and rewrites the instruction. Better feedback gives better rewrites. A stronger reflection model helps, so set `DSPY_LAB_REFLECTION_MODEL`.

## Not built here

| Optimizer | Why it is not in this repo |
|---|---|
| `BootstrapFewShotWithOptuna` | Same idea as random search with Optuna as the search. `optuna` is installed because `MIPROv2` needs it, but this repo has not tried this optimizer. |
| `AvatarOptimizer` | Targets agents that call tools. This repo has no tool-using agent. |
| `BootstrapFinetune`, `BetterTogether` | Train model weights. They need a GPU or a provider's fine-tuning API. |

Docs: [optimizers overview](https://dspy.ai/learn/optimization/optimizers/). Per-optimizer API pages are linked in each example's docstring.
