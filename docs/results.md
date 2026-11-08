# Results

This page gives the numbers from one run of every example, the method, and the limits. Do not treat this as a benchmark. It shows what the examples print and how much to trust it.

## Method

- Model. `qwen3-coder:30b` on a local Ollama, temperature 0, 4 threads. This is not the default `llama3.1:8b`.
- Cache off (`DSPY_LAB_CACHE=0`), so tokens and times are real.
- Data. 36 train, 36 dev and 36 test tickets, six per label in each split. The score is the share of the 36 test tickets labeled correctly. Examples 06 and 07 use the 18 dev questions.
- Optimizers fit on train and use dev for selection. The score is on test, which no optimizer sees.
- Each example ran once. There are no repeat runs, so there is no measured variance.

## Table

Every row starts from the same baseline of 69.4% test, except 06 and 07, which use RAG questions.

| Example | Optimizer or idea | Test after | Prompt tokens | Completion tokens | Time |
|---|---|---|---|---|---|
| 03 | `BootstrapFewShot`, labeled demos | 86.1% | 62,063 | 1,566 | 175 s |
| 03 | `BootstrapFewShot`, bootstrapped demos only | 80.6% | same run | same run | same run |
| 05 | `BootstrapFewShot`, saved and reloaded | 83.3% | 48,766 | 1,077 | 131 s |
| 09 | `LabeledFewShot` | 83.3% | 35,038 | 958 | 102 s |
| 11 | `KNNFewShot` | 83.3% | 26,119 | 970 | 131 s |
| 08 | `LabeledFewShot`, size picked by dev | 83.3% | 245,524 | 3,173 | 472 s |
| 15 | `InferRules` | 80.6% | 211,347 | 2,991 | 436 s |
| 16 | `Ensemble` of three | 80.6% | 130,841 | 3,342 | 358 s |
| 04 | `MIPROv2`, light | 77.8% | 208,359 | 7,283 | 833 s |
| 10 | `BootstrapFewShotWithRandomSearch` | 77.8% | 154,741 | 5,549 | 591 s |
| 14 | `GEPA` | 72.2% | 163,476 | 7,193 | 1,810 s |
| 13 | `SIMBA` | 69.4% | 131,151 | 6,683 | 1,237 s |
| 12 | `COPRO` | 66.7% | 50,261 | 2,613 | 225 s |
| 02 | Baseline only (dev 72.2%) | 69.4% | 16,438 | 953 | 90 s |
| 06 | Judge vs containment, on 18 questions | 94% vs 89% | 10,814 | 2,473 | 129 s |
| 07 | RAG, `BootstrapFewShot`, on 18 questions | 88.9% to 83.3% | 53,491 | 4,272 | 331 s |
| 01 | Prompt walkthrough | not scored | 1,055 | 219 | 16 s |

Example 08 also prints train and dev scores for 6, 18 and 36 training tickets. See the [walkthrough](walkthrough.md#step-16-catch-overfitting-08_overfitting_and_splitspy).

## Local model and Anthropic, on one example

Example 03 ran on both. The Anthropic run used `claude-haiku-5-5` through the `ant` CLI login, with the cache off, once.

| | Local `qwen3-coder:30b` | Anthropic `claude-haiku-5-5` |
|---|---|---|
| Baseline test score | 69.4% | 72.2% |
| Bootstrapped demos only | 80.6% | 72.2% |
| Labeled demos | 86.1% | 77.8% |
| Model calls | 118 | 118 |
| Prompt tokens | 62,063 | 94,651 |
| Completion tokens | 1,566 | 6,750 |
| Time | 175 s | 42 s |
| Temperature | 0, results repeat | not accepted by the API, sampling is on |

What differs:

- Anthropic starts higher and gains less. Labeled demos added 5.6 points there against 16.7 locally. The Anthropic gain is two tickets, which is inside the noise.
- The direction is the same on both. Labeled demos beat bootstrapped demos, and bootstrapped demos added nothing on Anthropic.
- Anthropic used 1.5 times the prompt tokens and 4.3 times the completion tokens. Some of the extra completion tokens are thinking tokens that are billed but not shown. See [getting started](getting-started.md#anthropic-without-a-key).
- Anthropic finished four times faster.
- Anthropic runs can differ from run to run. The same baseline scored 75.0% in example 02 and 72.2% in example 03.

## Why your numbers will differ

- Different model. A smaller model starts lower, and the gaps between optimizers change.
- Small test set. One ticket is 2.8 points. For a score near 80% on 36 tickets, one standard error is about 7 points. Gaps under about 8 points are within noise, so most of the ordering in the table is not reliable.
- Selection noise. Random search picked a candidate with 83.3% on dev, and it scored 77.8% on test. MIPROv2's best trial scored 83.3% on dev and its final program 77.8% on test.
- One run each. The same example on the same model can score differently if sampling is on.
- Cache. A repeat run with the cache on finishes in seconds and reports 0 tokens.

## What the run supports

- Every example that adds labeled demos beat the baseline by 10 points or more. That is larger than the noise.
- No optimizer found the precedence rule from the labels. GEPA learned one piece of the policy, and `InferRules` and `MIPROv2` wrote general definitions.
- `SIMBA` made no change because the model gave the same answer on every sample.
- `COPRO` ended 2.8 points below baseline, which is one ticket. The test cannot separate that from no change.

## What it does not support

- It does not rank the optimizers. The gaps between 77.8% and 86.1% are within a couple of standard errors.
- It does not say how these optimizers do on a larger task or a different metric. Only example 03 was also run on Anthropic, see below.
- It does not test `COPRO` as designed. Ollama returns one completion per call, so it tried one candidate per round.

## Other limits

- Times are not a clean comparison. A `podman build` and container tests ran during the `GEPA` run, which took 1,810 s against 638 s in an earlier run. `SIMBA` took 1,237 s against 870 s earlier.
- The `SIMBA` log shows three type-mismatch warnings from DSPy's own rule-writing step. The values it names are floats, and the run completed. This repo did not investigate them.
