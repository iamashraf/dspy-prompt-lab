# Example 16: Ensemble

You will see how a majority vote across three programs compares with each program alone.

| | |
|---|---|
| Run | `uv run python examples/16_ensemble.py` |
| Model calls | 252 |
| Reference run | test 69.4% to 80.6%, 130,841 prompt + 3,342 completion tokens, 358 s |

## Step 1. Build three different members

Ensemble does not search. It needs programs you already have. Each member here uses `LabeledFewShot` on a different third of the training set, so the demos differ.

```python
    members = [
        dspy.LabeledFewShot(k=8).compile(make_triage(), trainset=train[i::3]) for i in range(3)
    ]
```

**What to notice.** `LabeledFewShot` uses a fixed seed. The same training set would give the same demos and three identical members.

## Step 2. Score each member

```python
    for i, member in enumerate(members):
        print(f"member {i}: {evaluate(member).score:.1f}% on test")
```

Output:

```text
member 0: 75.0% on test
member 1: 80.6% on test
member 2: 75.0% on test
```

**What to notice.** The members score differently, so the vote can change some answers.

## Step 3. Vote

`dspy.majority` picks the most common answer across the members. Each ticket costs three model calls.

```python
    ensemble = dspy.Ensemble(reduce_fn=dspy.majority).compile(members)
    print_result("16_ensemble", baseline, evaluate(ensemble).score, lm, started)
```

Output:

```text
RESULT 16_ensemble: test 69.4% -> 80.6%  (130,841 prompt + 3,342 completion tokens, 358s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The vote scored 80.6%, the same as the best member. It beat the baseline of 69.4%, but it did not beat member 1. The vote removes the need to pick the best member in advance.

## Try it

- Change `range(3)` and `train[i::3]` to five members.
- Make two members share the same slice and check whether the vote changes.
- Swap `dspy.majority` for another `reduce_fn`.

## Read next

- [Walkthrough, step 8](../walkthrough.md)
- [Optimizers](../optimizers.md)
