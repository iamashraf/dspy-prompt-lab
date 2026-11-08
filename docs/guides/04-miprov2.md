# Example 04: MIPROv2

You will let MIPROv2 propose new instructions, bootstrap demos, and search over the combinations. Then you will inspect what it chose.

| | |
|---|---|
| Run | `uv run python examples/04_miprov2.py` |
| Model calls | 486 |
| Reference run | test 69.4% to 77.8%, 208,359 prompt + 7,283 completion tokens, 833 s |

## Step 1. Measure the baseline and the starting instruction

The baseline instruction is one short line.

```python
program = make_triage()
baseline = evaluate(program).score
print(f"Baseline on test: {baseline:.1f}%")
print("Baseline instruction:", program.predictors()[0].signature.instructions)
```

Output:

```text
Baseline on test: 69.4%
Baseline instruction: Route a support ticket to the right queue.
```

## Step 2. Compile with a light search

`auto="light"` sets a small budget. MIPROv2 scores candidates on `valset`, never on test.

```python
    optimizer = dspy.MIPROv2(
        metric=label_match,
        auto="light",  # "light" | "medium" | "heavy": how big the search is
        num_threads=4,
        seed=0,
    )
    compiled = optimizer.compile(
        program,
        trainset=train,
        valset=dev,  # candidates are scored here, never on test
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )
```

Output:

```text
Bootstrapping set 1/6
Bootstrapping set 2/6
Bootstrapping set 3/6
Bootstrapped 3 full traces after 6 examples for up to 1 rounds, amounting to 6 attempts.
Bootstrapping set 4/6
Bootstrapped 3 full traces after 4 examples for up to 1 rounds, amounting to 4 attempts.
Bootstrapping set 5/6
Bootstrapped 1 full traces after 1 examples for up to 1 rounds, amounting to 1 attempts.
Bootstrapping set 6/6
```

**What to notice.** The run has three phases. It bootstraps demo sets, proposes 3 instructions, then tries combinations.

## Step 3. Read the new instruction

The optimizer returns a new instruction and new demos. Both are plain text you can inspect.

```python
predictor = compiled.predictors()[0]
print("\n--- optimized instruction ---")
print(predictor.signature.instructions)
```

Output, trimmed:

```text
You are a critical customer support ticket classifier for a collaborative workspace tool. Your classification directly impacts user experience - misclassified tickets can delay customer issues by hours or days, potentially causing lost ...
When classifying:
- Assign "account" to tickets about sign-in issues, login problems, domain changes, or authentication failures
- Assign "howto" to questions about platform usage, feature explanations, or procedural guidance
- Assign "feature" to requests for new functionality or improvements
- Assign "billing" to payment, subscription, or pricing inquiries
- Assign "security" to concerns about account security, data breaches, or privacy issues
```

**What to notice.** The instruction now has per-label rules. It has no rule for `bug`. The printed instruction also contains the line `Now classify this ticket: {ticket_text}`.

## Step 4. Read the demos and trial scores

```python
for demo in predictor.demos:
    print(f"  {demo.ticket[:60]!r} -> {demo.label}")

scores = [t.get("full_eval_score") for t in compiled.trial_logs.values()]
print(f"\nValidation score of each full evaluation (from trial_logs): {scores}")
```

Output:

```text
--- 3 demos ---
  'Where do I change my email? The settings page only shows a s' -> account
  "Our trial ended and now I can't edit boards. How do I pay?" -> billing
  'Can I get an audit log of changes on our boards?' -> howto
Validation score of each full evaluation (from trial_logs): [72.22, 83.33, 83.33, 72.22, 80.56, 80.56, 75.0, 83.33, 77.78, 75.0, 83.33]
```

**What to notice.** The first score is the original program at 72.22. The best trials reached 83.33 on dev, which is 4 tickets above it.

## Step 5. Score on test

```python
after = evaluate(compiled).score
print_result("04_miprov2", baseline, after, lm, started)
```

Output:

```text
RESULT 04_miprov2: test 69.4% -> 77.8%  (208,359 prompt + 7,283 completion tokens, 833s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** Test gained 3 tickets (8.4 points) from one run, slightly above the 8-point noise line from example 03. The dev score of 83.33 fell to 77.8 on test. This run cost about 3 times the tokens of example 03 for a smaller gain.

## Try it

- Set `auto="medium"` and compare the trial scores.
- Set `verbose=True` on `MIPROv2` to read the full proposal log.
- Change `seed=0` to `seed=1` and see whether the best instruction changes.

## Read next

- [Walkthrough, step 9](../walkthrough.md#step-9-search-for-instructions-and-demos-04_miprov2py)
- [Optimizers](../optimizers.md)
