# Concepts

Six ideas explain every example. Each has a short definition and the file where you can see it.

## Signature

A signature declares what goes into a model call and what comes out. DSPy turns it into a prompt.

```python
class Triage(dspy.Signature):
    """Route a support ticket to the right queue."""
    ticket: str = dspy.InputField()
    label: Literal["billing", "bug", "howto", "account", "security", "feature"] = dspy.OutputField()
```

The docstring becomes the instruction. The types constrain the output. See `src/dspy_prompt_lab/programs.py` and example 01. Docs: [signatures](https://dspy.ai/learn/programming/signatures/).

## Module

A module is a strategy for running a signature. `dspy.Predict` asks for the answer directly. `dspy.ChainOfThought` asks for reasoning first. You can also write your own `dspy.Module` with a `forward` method that chains several calls. See example 01. Docs: [modules](https://dspy.ai/learn/programming/modules/).

## Example and dataset

A `dspy.Example` is one labeled item, such as a ticket and its label. `with_inputs("ticket")` marks which fields are inputs. The rest are labels. See `src/dspy_prompt_lab/data.py`. Docs: [data](https://dspy.ai/learn/evaluation/data/).

## Metric

A metric is a function `f(gold, pred, trace=None)` that returns a score for one prediction. Return a bool or a float. DSPy passes `trace` during optimization and leaves it `None` during evaluation. See `src/dspy_prompt_lab/metrics.py`. Docs: [metrics](https://dspy.ai/learn/evaluation/metrics/).

A metric is the most important choice you make. The optimizer improves whatever the metric rewards, including its mistakes. Example 06 shows a containment metric that passed a refusal and an LLM judge that caught it.

## Optimizer

An optimizer takes a program, a metric and some data, and returns a better program. It changes one or both of two things.

- The demos, which are examples shown in the prompt.
- The instruction, which is the text that tells the model what to do.

The program you get back is a new object. The original is unchanged. See [optimizers](optimizers.md).

## Train, dev and test

| Split | Used for | Rule |
|---|---|---|
| Train | The optimizer learns from it | The score on it is flattering |
| Dev | Choosing between configurations, and the optimizer's `valset` | You may look at it often |
| Test | The number you report | Look once, at the end |

This repo splits 108 tickets into 36 each, with six per label in every split. Example 08 shows why this matters. With 6 training tickets, the score on train was 100% and on dev 80.6%.

Never tune settings on the test split. The examples follow this rule. They fit on train, select on dev, and report test.
