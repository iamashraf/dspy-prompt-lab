# Example 05: Save and load a compiled program

Optimization costs calls and time, so do it once. You will save the learned state to JSON and load it into a fresh program with no training calls.

| | |
|---|---|
| Run | `uv run python examples/05_save_load_compiled.py` |
| Model calls | 81 |
| Reference run | test 69.4% to 83.3%, 48,766 prompt + 1,077 completion tokens, 131 s |

## Step 1. Score the baseline

The script measures the plain triage program on the 36 test tickets first.

```python
    train, _, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score
```

Output:

```text
Average Metric: 25 / 36 (69.4%)
```

**What to notice.** 25 of 36 tickets are correct before any optimization.

## Step 2. Optimize and save

`BootstrapFewShot` runs once. `save()` writes the result to a `.json` file.

```python
    compiled = dspy.BootstrapFewShot(metric=label_match, max_bootstrapped_demos=6).compile(
        make_triage(), trainset=train
    )
    PATH.parent.mkdir(exist_ok=True)
    compiled.save(PATH)
    print(f"Saved {PATH} ({PATH.stat().st_size:,} bytes). Head of the file:")
    print(PATH.read_text()[:300], "...")
```

Output:

```text
Bootstrapped 6 full traces after 9 examples for up to 1 rounds, amounting to 9 attempts.
Saved artifacts/triage_compiled.json (2,606 bytes). Head of the file:
{
  "traces": [],
  "train": [],
  "demos": [
    {
      "augmented": true,
      "ticket": "Our SSO stopped working after the renewal date passed. The card on file expired.",
      "label": "billing"
    },
```

**What to notice.** The file is plain text and only 2,606 bytes. It holds the demos for the predictor.

## Step 3. Load into a new program

Build the same program class, then call `load()`. No optimizer runs here.

```python
    loaded = make_triage()
    loaded.load(PATH)
```

The script then compares the loaded demos with the saved ones.

```python
    same = [dict(d) for d in loaded.predictors()[0].demos] == [
        dict(d) for d in compiled.predictors()[0].demos
    ]
    print(f"\nLoaded {len(loaded.predictors()[0].demos)} demos. Same as the saved program: {same}")
```

Output:

```text
Loaded 16 demos. Same as the saved program: True
```

**What to notice.** 16 demos came back, and the script confirms they match the saved program. Loaded demos are plain dicts, so the script compares them as dicts.

Why 16 when the script asked for at most 6 bootstrapped demos? `BootstrapFewShot` fills the rest with labeled training examples up to `max_labeled_demos`, which defaults to 16. DSPy computes the count as `max_labeled_demos` minus the bootstrapped demos, so the 6 bootstrapped demos plus 10 labeled ones make 16.

## Step 4. Score the loaded program

```python
    after = evaluate(loaded).score
    print_result("05_save_load", baseline, after, lm, started)
```

Output:

```text
Average Metric: 30 / 36 (83.3%)
RESULT 05_save_load: test 69.4% -> 83.3%  (48,766 prompt + 1,077 completion tokens, 131s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** 30 of 36 is five more tickets than the baseline. The loaded program ran with no optimizer call. One ticket is 2.8 points, so treat small gaps with care.

## Try it

- Open `artifacts/triage_compiled.json` and read the demos.
- Comment out the `compile` call and load the existing file to see that no training calls run.
- Load the file into a differently built program and compare `len(predictor.demos)` before and after.

## Read next

- [Step 5 in the walkthrough](../walkthrough.md#step-5-save-and-reload-05_save_load_compiledpy)
- [Optimizers](../optimizers.md)
