# Example 02: Evaluate a baseline

You will measure an unoptimized program on the dev and test sets, then read its failures. Every later example compares against this number.

| | |
|---|---|
| Run | `uv run python examples/02_evaluate_baseline.py` |
| Model calls | 72 |
| Reference run | dev 72.2%, test 69.4%, 16,438 prompt + 953 completion tokens, 90 s |

## Step 1. Load the data

The data has three splits of 36 tickets each, with six balanced labels.

```python
train, dev, test = load_tickets()
print(f"train={len(train)}  dev={len(dev)}  test={len(test)} (6 labels, balanced)")
print("one example:", train[0])
```

Output:

```text
train=36  dev=36  test=36 (6 labels, balanced)
one example: Example({'ticket': 'Our SSO stopped working after the renewal date passed. The card on file expired.', 'label': 'billing'}) (input_keys={'ticket'})
```

**What to notice.** Each `Example` holds a ticket and a gold label. Only `ticket` is an input key.

## Step 2. Score the baseline on dev and test

`dspy.Evaluate` runs the program on every example and applies the metric. `num_threads` runs calls in parallel.

```python
program = make_triage()
result = dspy.Evaluate(devset=dev, metric=label_match, num_threads=4)(program)
test_score = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)(program).score
```

```python
print(f"\nBaseline (Predict, no optimization): dev {result.score:.1f}%   test {test_score:.1f}%")
```

Output:

```text
Baseline (Predict, no optimization): dev 72.2%   test 69.4%
```

**What to notice.** One ticket is 2.8 points on 36 tickets. Dev and test differ by one ticket here, so treat them as the same level.

## Step 3. Read the failures

A score does not say what to fix. `show_failures` prints the misses with predicted and gold labels.

```python
show_failures(result)
```

Output:

```text
10 of 36 failed. First 5:
  "I'd like a refund because the Slack integration double-posts and my te"
    predicted='bug'  gold='billing'
  'Can I invite a client to see one board only, without access to the res'
    predicted='feature'  gold='howto'
  'How do I leave a workspace I was invited to?'
    predicted='howto'  gold='account'
  'Can I turn on two-factor authentication with an authenticator app?'
    predicted='security'  gold='howto'
  'We were promised a nonprofit discount in January. How do I apply it?'
    predicted='howto'  gold='billing'
```

**What to notice.** The misses sit between neighbor labels, such as a refund request that mentions a bug. The prompt does not state the labeling rules for these cases, so the model has to guess.

## Try it

- Set `num_threads=1` and compare the run time.
- Print `show_failures(result)` for the test result too.
- Change `train[0]` to `dev[0]` in the print and look at a dev ticket.

## Read next

- [Walkthrough, step 2](../walkthrough.md#step-2-measure-a-baseline-02_evaluate_baselinepy)
- [Concepts](../concepts.md)
