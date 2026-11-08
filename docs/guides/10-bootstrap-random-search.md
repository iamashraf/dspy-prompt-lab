# Example 10: BootstrapFewShotWithRandomSearch

You will build several candidate demo sets, score each on the dev set, and keep the winner. This is example 03 repeated with different seeds.

| | |
|---|---|
| Run | `uv run python examples/10_bootstrap_random_search.py` |
| Model calls | 421 |
| Reference run | test 69.4% to 77.8%, 154,741 prompt + 5,549 completion tokens, 591 s |

## Step 1. Set up the search

Each candidate gets up to 4 bootstrapped and 4 labeled demos. The default is 16 candidates. This script uses 6 to stay cheap.

```python
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=label_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=6,  # the default is 16; 6 keeps this example cheap
        num_threads=4,
    )
```

**What to notice.** The optimizer needs the metric, because it scores every candidate with it.

## Step 2. Compile with a validation set

Pass the dev split as `valset`. The optimizer picks the winner on dev, not on test.

```python
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
```

Output:

```text
Going to sample between 1 and 4 traces per predictor.
Will attempt to bootstrap 6 candidate sets.
New best score: 72.22 for seed -3
Scores so far: [72.22]
New best score: 77.78 for seed -2
Scores so far: [72.22, 77.78]
```

**What to notice.** The log reports a running best score after each candidate. The best moves from 72.22 to 77.78 within two candidates.

## Step 3. Read the candidate scores

DSPy found 9 candidates, more than the 6 asked for, and the script prints every dev score.

```python
    print("Candidate scores on dev:", [round(c["score"], 1) for c in compiled.candidate_programs])
```

Output:

```text
9 candidate programs found.
Candidate scores on dev: [83.3, 80.6, 80.6, 80.6, 77.8, 77.8, 77.8, 75.0, 72.2]
```

**What to notice.** Dev scores range from 72.2% to 83.3%. Different demo sets change the result by up to 11 points.

## Step 4. Score the winner on test

```python
    print_result("10_random_search", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
RESULT 10_random_search: test 69.4% -> 77.8%  (154,741 prompt + 5,549 completion tokens, 591s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The best dev score was 83.3%, but the test score is 77.8%. That gap is selection noise, which is why you report on a split the optimizer never saw. This run cost about 4 times the tokens of example 09 and scored lower on test.

## Try it

- Set `num_candidate_programs=2` and see how few candidates you need.
- Set `max_bootstrapped_demos=0` and compare with example 09.
- Swap `dev` for a slice of `train` as `valset` and watch the dev scores rise.

## Read next

- [Walkthrough, step 7](../walkthrough.md)
- [Optimizers](../optimizers.md)
