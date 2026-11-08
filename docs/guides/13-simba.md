# Example 13: SIMBA

You will see how SIMBA looks for tickets where the model gives different answers, and why it learns nothing when the model is stable.

| | |
|---|---|
| Run | `uv run python examples/13_simba.py` |
| Model calls | 493 |
| Reference run | test 69.4% to 69.4%, 131,151 prompt + 6,683 completion tokens, 1,237 s |

## Step 1. Measure the baseline

The baseline uses the same test set as every other example.

```python
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score
```

Output:

```text
2026/11/07 22:40:40 INFO dspy.evaluate.evaluate: Average Metric: 25 / 36 (69.4%)
```

**What to notice.** 25 of 36 tickets are right before any optimizing.

## Step 2. Wrap the metric so it returns a float

SIMBA does its math with floats. The code comment says a bool logged a type warning.

```python
    def float_metric(gold, pred, trace=None) -> float:
        return float(label_match(gold, pred))  # SIMBA computes with floats; a bool logged a type warning
```

**What to notice.** This is the only change to the metric. The scoring rule is still `label_match`.

## Step 3. Build the optimizer

SIMBA samples the program several times on a mini-batch of tickets. It looks for tickets where the scores differ between samples.

```python
    optimizer = dspy.SIMBA(
        metric=float_metric, bsize=12, num_candidates=4, max_steps=4, max_demos=4, num_threads=4
    )
```

**What to notice.** `bsize=12` and `max_steps=4` give 4 batches of 12 tickets, 4 samples each.

## Step 4. Compile with a sampling model

Variation needs randomness. This block runs the compile with a second model at temperature 0.7 and no cache.

```python
    # A separate LM that samples (temperature 0.7, no cache) so repeated runs can differ.
    with dspy.context(lm=get_lm(temperature=0.7, cache=False)):
        compiled = optimizer.compile(make_triage(), trainset=train, seed=0)
```

Output:

```text
2026/11/07 22:40:41 INFO dspy.teleprompt.simba: Starting batch 1 of 4.
2026/11/07 22:40:41 INFO dspy.teleprompt.simba: Sampling program trajectories on 12 examples x 4 samples.
2026/11/07 22:42:21 INFO dspy.teleprompt.simba: Batch 1: Processing bucket #1, with max score 1.0, max-to-min gap 0.0, and max-to-avg gap 0.0.
2026/11/07 22:42:21 INFO dspy.teleprompt.simba_utils: Skipping rule generation as good score 1.0 is at or below the 10th percentile *or* bad score 1.0 is at or above the 90th percentile.
```

**What to notice.** A gap of 0.0 means all four samples scored the same, so there is no good and bad attempt to compare. In this run, 19 of 20 buckets had a gap of 0.0. Only batch 2, bucket 1 had a gap of 1.0, and SIMBA wrote one rule from it.

## Step 5. Check what changed

Print the final instruction and the demo count.

```python
    print("\n--- instruction after SIMBA ---")
    print(compiled.predictors()[0].signature.instructions)
    print(f"demos: {len(compiled.predictors()[0].demos)}")
    print_result("13_simba", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
--- instruction after SIMBA ---
Route a support ticket to the right queue.
demos: 0
RESULT 13_simba: test 69.4% -> 69.4%  (131,151 prompt + 6,683 completion tokens, 1237s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The final program has the original instruction and no demos. SIMBA spent 131,151 prompt tokens and kept nothing. SIMBA fits tasks where the model is unstable, and this classifier is stable.

## Try it

- Raise `temperature` in `get_lm(temperature=0.7, cache=False)` and check whether more buckets show a gap above 0.0.
- Lower `bsize` so each batch holds fewer tickets, as the file's debug note suggests.
- Run it with a smaller model that makes more varied mistakes.

## Read next

- [Walkthrough, step 13](../walkthrough.md)
- [Optimizers](../optimizers.md)
