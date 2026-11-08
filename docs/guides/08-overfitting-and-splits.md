# Example 08: Overfitting and data splits

You will optimize with 6, 18 and 36 training tickets and print the train and dev score for each. The gap between them is the warning sign.

| | |
|---|---|
| Run | `uv run python examples/08_overfitting_and_splits.py` |
| Model calls | 240 |
| Reference run | test 69.4% to 83.3%, 245,524 prompt + 3,173 completion tokens, 472 s |

## Step 1. Keep three splits

Train is what the optimizer sees. Dev is for picking between configurations. Test is for one final score.

```python
    _, dev, test = load_tickets()
    baseline_test = score(make_triage(), test)
```

Output:

```text
Average Metric: 25 / 36 (69.4%)
```

**What to notice.** The test baseline is 69.4%, and the script measures it once.

## Step 2. Score on train and on dev

The helper scores any program on any split.

```python
def score(program, split):
    return dspy.Evaluate(devset=split, metric=label_match, num_threads=4)(program).score
```

For each training size, the script compiles with `LabeledFewShot` and scores both splits. The prompt contains the labeled demos themselves, so the model can memorize them.

```python
    for per_label in (1, 3, 6):  # 6, 18, 36 training tickets
        train, _, _ = load_tickets(n_train=per_label)
        # Labeled demos are copied from the trainset into the prompt, so they are the easiest to memorize.
        compiled = dspy.LabeledFewShot(k=len(train)).compile(make_triage(), trainset=train)
        row = (score(compiled, train), score(compiled, dev))
        print(f"{len(train):>10} {row[0]:>6.1f}% {row[1]:>6.1f}%")
```

Output:

```text
  trainset   train     dev
         6  100.0%   80.6%
        18   88.9%   86.1%
        36   91.7%   83.3%
```

**What to notice.** With 6 tickets, train is 100.0% and dev is 80.6%, a gap of 19.4 points. With 18 tickets the gap is 2.8 points, and with 36 it is 8.4. The train score overstates quality. The gap between train and dev is what to watch.

## Step 3. Pick by dev, report once on test

The script keeps the configuration with the best dev score. It then scores that one program on test.

```python
    if best is None or row[1] > best[0]:
        best = (row[1], compiled)
```

```python
    after = score(best[1], test)
```

Output:

```text
RESULT 08_overfitting: test 69.4% -> 83.3%  (245,524 prompt + 3,173 completion tokens, 472s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The dev winner was the 18-ticket run at 86.1%. Its test score is 83.3%, lower than its dev score. Dev picked it, so dev overstates it a little, and test is the number to report.

## Try it

- Change `(1, 3, 6)` to `(1, 2, 4, 6)` and see where the train and dev gap closes.
- Replace `LabeledFewShot` with `BootstrapFewShot` and compare the gaps.
- Print the dev score of each run next to its test score to see how much dev flatters the pick.

## Read next

- [Step 16 in the walkthrough](../walkthrough.md#step-16-catch-overfitting-08_overfitting_and_splitspy)
- [Concepts](../concepts.md#train-dev-and-test)
