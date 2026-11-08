# Example 14: GEPA

You will see how GEPA reads feedback text from your metric and rewrites the instruction from the mistakes.

| | |
|---|---|
| Run | `uv run python examples/14_gepa.py` |
| Model calls | 395 |
| Reference run | test 69.4% to 72.2%, 163,476 prompt + 7,193 completion tokens, 1,810 s |

## Step 1. Write a metric that explains its score

Most metrics return a number. This one also returns text that says what went wrong, and GEPA reads that text.

```python
def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score 1/0 and say why. The feedback text is what GEPA's reflection LM reads."""
    ok = label_match(gold, pred)
    if ok:
        return dspy.Prediction(score=1.0, feedback=f"Correct: {gold.label}.")
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Wrong. Predicted '{pred.label}' but the correct label is '{gold.label}'. "
            f"Ticket: {gold.ticket!r}. Work out what rule separates these two labels."
        ),
    )
```

**What to notice.** The feedback names both labels and shows the ticket. A person could fix the prompt from it.

## Step 2. Set up a reflection model

The reflection model writes the new instructions. The code reads `DSPY_LAB_REFLECTION_MODEL` so you can pick a stronger one. Without it, the same model reflects.

```python
    reflection_lm = get_lm(
        model=os.getenv("DSPY_LAB_REFLECTION_MODEL") or None, temperature=1.0, max_tokens=4000
    )
```

## Step 3. Run GEPA

`max_metric_calls=300` caps the budget. GEPA scores each candidate on the dev set.

```python
    optimizer = dspy.GEPA(
        metric=feedback_metric,
        max_metric_calls=300,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        num_threads=4,
        track_stats=True,  # adds `detailed_results` to the returned program
        seed=0,
    )
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
```

Output:

```text
2026/11/07 22:10:22 INFO dspy.evaluate.evaluate: Average Metric: 25 / 36 (69.4%)
2026/11/07 22:11:15 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.7222222222222222 over 36 / 36 examples
2026/11/07 22:11:19 INFO dspy.teleprompt.gepa.gepa: Iteration 1: All subsample scores perfect for parent 0. Skipping.
2026/11/07 22:18:50 INFO dspy.teleprompt.gepa.gepa: Iteration 10: Found a better program on the valset with score 0.7777777777777778.
```

**What to notice.** When a mini-batch is all correct, GEPA skips the iteration because there is no mistake to reflect on.

## Step 4. Read the result

The script prints the new instruction and the dev score of every candidate.

```python
    print("\n--- instruction after GEPA ---")
    print(compiled.predictors()[0].signature.instructions)
    print("candidate val scores:", [round(s, 2) for s in compiled.detailed_results.val_aggregate_scores])
    print_result("14_gepa", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
--- instruction after GEPA ---
Route support tickets to the appropriate queue based on the content and intent of the ticket. Use the following rules: 1) Classify as 'security' if the ticket involves unauthorized access, suspicious activity, account compromise, or security concerns (e.g., unexpected logins, unauthorized invitations, suspicious behavior). 2) Classify as 'howto' if the ticket asks a question about how to use a spe...
candidate val scores: [0.72, 0.69, 0.72, 0.69, 0.78, 0.72, 0.75]
RESULT 14_gepa: test 69.4% -> 72.2%  (163,476 prompt + 7,193 completion tokens, 1810s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The rewritten instruction says a question about SAML SSO with Azure AD is `howto`, not `feature`. That rule came from a mistake. The test score moved from 25 to 26 of 36 tickets, which is one ticket and within noise. The best dev score, 0.78, is higher than the test score.

## Try it

- Add a hint about label precedence to the feedback text and compare the instruction.
- Set `DSPY_LAB_REFLECTION_MODEL` to a stronger model.
- Pass `log_dir=` to `dspy.GEPA` so a run can resume.

## Read next

- [Walkthrough, step 12](../walkthrough.md)
- [Optimizers](../optimizers.md)
