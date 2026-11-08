# Example 15: InferRules

You will see how InferRules writes plain-language rules from examples and appends them to the instruction, so you can read what it learned.

| | |
|---|---|
| Run | `uv run python examples/15_infer_rules.py` |
| Model calls | 226 |
| Reference run | test 69.4% to 80.6%, 211,347 prompt + 2,991 completion tokens, 436 s |

## Step 1. Measure the baseline

```python
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score
```

Output:

```text
2026/11/07 21:48:46 INFO dspy.evaluate.evaluate: Average Metric: 25 / 36 (69.4%)
```

## Step 2. Compile with InferRules

InferRules first bootstraps demos. It then asks a model to write rules that explain them, and it keeps the rule set that scores best on the dev set.

```python
    optimizer = dspy.InferRules(
        num_candidates=4, num_rules=6, num_threads=4, metric=label_match, max_bootstrapped_demos=6
    )
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
```

Output:

```text
Bootstrapped 6 full traces after 10 examples for up to 1 rounds, amounting to 10 attempts.
2026/11/07 21:50:29 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 1 with score 86.11. Current best score: 86.11
2026/11/07 21:51:49 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 2 with score 83.33. Current best score: 86.11
2026/11/07 21:53:06 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 3 with score 83.33. Current best score: 86.11
2026/11/07 21:54:23 INFO dspy.teleprompt.infer_rules: Evaluated Candidate 4 with score 83.33. Current best score: 86.11
2026/11/07 21:54:23 INFO dspy.teleprompt.infer_rules: Final best score: 86.11
```

**What to notice.** These scores are on the dev set, not the test set. The first candidate won.

## Step 3. Read the rules

The rules sit in the instruction, so printing it shows what the optimizer learned.

```python
    print("\n--- instruction with induced rules ---")
    print(compiled.predictors()[0].signature.instructions)
    print_result("15_infer_rules", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
--- instruction with induced rules ---
Route a support ticket to the right queue.
Please adhere to the following rules when making your prediction:
1. Classify tickets related to payments, invoices, billing cycles, or subscription issues as 'billing'.
2. Assign tickets involving login problems, authentication, domain changes, or account access as 'account'.
3. Label questions or requests for instructions on how to perform a task or use a feature as 'howto'.
4. Categorize requests for new features, enhancements, or inquiries about feature availability as 'feature'.
5. Assign tickets reporting app behavior anomalies, inconsistencies, crashes, or unexpected functionality as 'bug'.
6. Classify tickets involving unauthorized access, data privacy, security audits, or suspicious activities as 'security'.
RESULT 15_infer_rules: test 69.4% -> 80.6%  (211,347 prompt + 2,991 completion tokens, 436s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The test score rose from 25 to 29 of 36 tickets. The rules are one definition per label. None of them states which label wins when a ticket fits two.

## Try it

- Raise `num_rules` and check whether the rules get more specific.
- Lower `max_bootstrapped_demos` and compare the dev scores.
- Check whether any rule covers a ticket that fits two labels.

## Read next

- [Walkthrough, step 11](../walkthrough.md)
- [Optimizers](../optimizers.md)
