# Example 06: A metric that uses an LLM as the judge

Exact checks fail on free text with many valid phrasings. You will build a judge metric and compare it with a cheap containment check on the same answers.

| | |
|---|---|
| Run | `uv run python examples/06_custom_metric_with_llm_judge.py` |
| Model calls | 36 |
| Reference run | containment 89% vs judge 94% on 18 questions, 10,814 prompt + 2,473 completion tokens, 129 s |

## Step 1. Define the judge signature

The judge gets the question, the reference and the answer. It returns a boolean.

```python
class Judge(dspy.Signature):
    """Decide whether the answer states the same fact as the reference answer. Ignore wording."""

    question: str = dspy.InputField()
    reference: str = dspy.InputField()
    answer: str = dspy.InputField()
    same_fact: bool = dspy.OutputField()
```

**What to notice.** The docstring tells the judge to ignore wording, so it compares meaning where containment compares text.

## Step 2. Wrap the judge as a metric

The metric runs the judge under its own LM, so the judge model can differ from the program model.

```python
    judge = dspy.Predict(Judge)

    def judge_metric(gold, pred, trace=None) -> bool:
        with dspy.context(lm=judge_lm):  # the judge runs on its own LM, not the program's
            verdict = judge(question=gold.question, reference=gold.answer, answer=pred.answer)
        return bool(verdict.same_fact)
```

**What to notice.** Set `DSPY_LAB_JUDGE_MODEL` to judge with another model of the same provider. In the reference run the judge and the program used the same local model.

## Step 3. Score the same answers twice

Each answer gets the containment check and the judge verdict. That is one extra call per question.

```python
    for ex in dev:
        pred = program(question=ex.question)
        rows.append((ex, pred, key_in_answer(ex, pred), judge_metric(ex, pred)))
```

Output:

```text
Containment metric: 89%   LLM judge: 94%   (18 questions)
```

**What to notice.** The judge accepts one more answer than containment. That is 1 of 18 questions, so the gap is small.

## Step 4. Read the disagreements

Read the cases where the two metrics differ. This is how you check a judge before you trust it.

```python
    for ex, pred, a, b in rows:
        if a != b:
            print(f"  Q: {ex.question}\n    reference: {ex.answer!r}\n    answer:    {pred.answer!r}")
            print(f"    containment={a}  judge={b}")
```

Output:

```text
Where they disagree:
  Q: I lost the phone with my authenticator app. What can I use to log in?
    reference: 'One of the 10 recovery codes'
    answer:    'You can use one of the 10 recovery codes that were shown during the initial setup of your two-factor authentication to log in.'
    containment=False  judge=True
```

**What to notice.** The answer is correct, and the judge is right to accept it. Containment missed it because the wording differs from the reference.

## Try it

- Set `DSPY_LAB_JUDGE_MODEL` to a different model and see if the verdicts change.
- Rerun and check whether any verdict flips. A judge that flips is noise.
- Change the docstring of `Judge` to be stricter and compare the two scores.

## Read next

- [Step 14 in the walkthrough](../walkthrough.md#step-14-judge-with-a-model-06_custom_metric_with_llm_judgepy)
- [Concepts](../concepts.md)
