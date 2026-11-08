# Example 03: BootstrapFewShot

You will compare two kinds of few-shot demos. Bootstrapped demos are runs the model got right. Labeled demos are training examples copied in as they are.

| | |
|---|---|
| Run | `uv run python examples/03_bootstrap_fewshot.py` |
| Model calls | 118 |
| Reference run | test 69.4% to 86.1% (labeled demos), 62,063 prompt + 1,566 completion tokens, 175 s |

## Step 1. Measure the baseline

All scores come from the 36-ticket test set.

```python
evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)

baseline = evaluate(make_triage()).score
print(f"Baseline on test: {baseline:.1f}%")
```

Output:

```text
Baseline on test: 69.4%
```

## Step 2. Compile with each demo type

The loop builds one program with 6 bootstrapped demos and one with 16 labeled demos. `compile` returns a new program.

```python
    optimizer = dspy.BootstrapFewShot(
        metric=label_match,
        max_bootstrapped_demos=bootstrapped,  # runs the model produced itself and the metric accepted
        max_labeled_demos=labeled,  # training examples copied in as-is
    )
    compiled = optimizer.compile(make_triage(), trainset=train)  # returns a new program
```

Output:

```text
Bootstrapped 6 full traces after 9 examples for up to 1 rounds, amounting to 9 attempts.
   bootstrapped demos only:  6 demos -> 80.6%
Bootstrapped 0 full traces after 0 examples for up to 1 rounds, amounting to 0 attempts.
        labeled demos only: 16 demos -> 86.1%
```

**What to notice.** The model got 6 of the first 9 tickets right on its own. Those demos teach what it already knew. The labeled demos include tickets it gets wrong, so they carry the hidden labeling rules.

## Step 3. Read the compiled prompt

The script makes one call with the labeled-demo program and prints the prompt.

```python
compiled, after = results["labeled demos only"]
print("\n--- the compiled prompt: one call with the labeled-demo program ---")
compiled(ticket=test[0].ticket)
dspy.inspect_history(n=1)
```

The instruction is unchanged:

```text
In adhering to this structure, your objective is: 
        Route a support ticket to the right queue.
```

One of the 16 demo tickets carries a policy rule:

```text
The audit log vanished after we downgraded. Is that a bug?
```

**What to notice.** That demo is labeled `billing`, not `bug`. The model can only learn that from a labeled demo.

## Step 4. Save each version

The loop also saves each compiled program, so you can diff and compare them later.

```python
results[name] = (compiled, evaluate(compiled).score)
save_version(compiled, "03_" + name.split()[0], results[name][1], "test", lm)
```

**What to notice.** Each call writes a JSON file and an index row under `artifacts/prompts/`. [Versioning](../versioning.md) shows how to compare the two files.

## Step 5. Read the result

```python
print_result("03_bootstrap_fewshot", baseline, after, lm, started)
print("Test has 36 tickets, so one ticket is 2.8 points. Read gaps under ~8 points as noise.")
```

Output:

```text
RESULT 03_bootstrap_fewshot: test 69.4% -> 86.1%  (62,063 prompt + 1,566 completion tokens, 175s, ollama_chat/qwen3-coder:30b)
Test has 36 tickets, so one ticket is 2.8 points. Read gaps under ~8 points as noise.
```

**What to notice.** The labeled gain is 16.7 points and the bootstrapped gain is 11.2 points. Both are from one run. On Anthropic `claude-haiku-5-5` the same script went from 72.2% to 77.8% in 42 s with 94,651 prompt + 6,750 completion tokens, which is within the noise range.

## Try it

- Set both demo counts to 8 and see how the score moves.
- Print `compiled.predictors()[0].demos` to see what reached the prompt.
- Replace `label_match` with a metric that always returns `False` and read the "0 full traces" log line.

## Read next

- [Walkthrough, step 4](../walkthrough.md#step-4-let-the-model-write-its-own-examples-03_bootstrap_fewshotpy)
- [Optimizers](../optimizers.md)
