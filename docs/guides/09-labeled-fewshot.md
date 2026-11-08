# Example 09: LabeledFewShot

You will copy 12 labeled training tickets into the prompt and see how far that free step goes. Every other optimizer should beat it to justify its cost.

| | |
|---|---|
| Run | `uv run python examples/09_labeled_fewshot.py` |
| Model calls | 72 |
| Reference run | test 69.4% to 83.3%, 35,038 prompt + 958 completion tokens, 102 s |

## Step 1. Measure the baseline

Score the plain triage program on the 36 test tickets first. This is the number to beat.

```python
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score
```

**What to notice.** The baseline only prints in the final RESULT line, as `69.4%`.

## Step 2. Copy k examples into the prompt

`LabeledFewShot` picks 12 training tickets and stores them as demos. It makes no model calls and needs no metric.

```python
    # sample=True draws k examples with a fixed seed. To get different demos, change the trainset
    # you pass in (reorder it or take a slice), as example 16 does.
    compiled = dspy.LabeledFewShot(k=12).compile(make_triage(), trainset=train)
    print(f"{len(compiled.predictors()[0].demos)} demos copied from the trainset")
```

Output:

```text
12 demos copied from the trainset
```

**What to notice.** The draw uses a fixed seed, so the same trainset gives the same demos on every run.

## Step 3. Score the compiled program

Run the same evaluation on the compiled program.

```python
    print_result("09_labeled_fewshot", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
RESULT 09_labeled_fewshot: test 69.4% -> 83.3%  (35,038 prompt + 958 completion tokens, 102s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The score rose 13.9 points (5 tickets). The cost is only the extra prompt tokens, because the 72 calls are the two test evaluations of 36 tickets each.

On Anthropic `claude-haiku-5-5` (one run) the same script went from 75.0% to 80.6%, using 53,766 prompt + 3,889 completion tokens in 17 s. The Anthropic baseline varies by two tickets from run to run, so that gain is within noise.

## Try it

- Change `k=12` to `k=4`, then `k=24`, and compare scores and prompt tokens.
- Reorder `train` (for example `train[::-1]`) to get a different set of demos.
- Print `compiled.predictors()[0].demos` and read what the model sees.

## Read next

- [Walkthrough, step 3](../walkthrough.md)
- [Optimizers](../optimizers.md)
