# Example 01: Signatures and modules

You will see what a DSPy signature and module are, and the exact prompt DSPy builds from them. No optimizer runs here.

| | |
|---|---|
| Run | `uv run python examples/01_signatures_and_modules.py` |
| Model calls | 5 |
| Reference run | not scored, 1,055 prompt + 219 completion tokens, 16 s on a local 30B model |

## Step 1. Declare a signature as a string

A signature says what goes in and what comes out. The field names carry the meaning.

```python
qa = dspy.Predict("question -> answer")
print("string signature  :", qa(question="What is the capital of France? One word.").answer)
```

Output:

```text
string signature  : Paris
```

**What to notice.** You wrote no prompt text. `dspy.Predict` turned the two field names into a prompt.

## Step 2. Declare a signature as a class

A class adds types and an instruction. `Triage.label` is a `Literal`, so only six labels are valid.

```python
predict = dspy.Predict(Triage)
print("class signature   :", predict(ticket=TICKET).label)
```

Output:

```text
class signature   : billing
```

**What to notice.** The class constrains the output to one of the six labels.

## Step 3. Add reasoning with ChainOfThought

`ChainOfThought` uses the same signature and adds a `reasoning` field that the model writes first.

```python
pred = dspy.ChainOfThought(Triage)(ticket=TICKET)
print("chain of thought  :", pred.label)
print("  reasoning       :", pred.reasoning.strip()[:200])
```

Output:

```text
chain of thought  : billing
  reasoning       : The ticket describes a situation where the customer was charged twice for their Team plan subscription. This is a billing issue that requires investigation and processing of a refund for the duplicate
```

**What to notice.** The module changed the strategy. The signature stayed the same.

## Step 4. Chain two steps in a custom module

A module declares sub-modules in `__init__` and uses them in `forward`. An optimizer later finds and tunes those sub-modules.

```python
class CleanThenTriage(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clean = dspy.Predict("ticket -> short_summary")
        self.triage = dspy.ChainOfThought(Triage)

    def forward(self, ticket: str):
        summary = self.clean(ticket=ticket).short_summary
        return self.triage(ticket=summary)
```

```python
program = CleanThenTriage()
print("custom module     :", program(ticket=TICKET).label)
print("  sub-modules     :", [name for name, _ in program.named_predictors()])
```

Output:

```text
custom module     : billing
  sub-modules     : ['clean', 'triage.predict']
```

**What to notice.** `named_predictors()` lists the two prompts an optimizer could tune.

## Step 5. Read the prompt DSPy built

`dspy.inspect_history` prints the last prompt and reply. This is the prompt you did not write.

```python
dspy.inspect_history(n=1)
print_usage(lm)
```

Output:

```text
Your input fields are:
1. `ticket` (str):
Your output fields are:
1. `reasoning` (str): 
2. `label` (Literal['billing', 'bug', 'howto', 'account', 'security', 'feature']):
All interactions will be structured in the following way, with the appropriate values filled in.
[[ ## ticket ## ]]
{ticket}
[[ ## reasoning ## ]]
{reasoning}
[[ ## label ## ]]
{label}        # note: the value you produce must exactly match (no extra characters) one of: billing; bug; howto; account; security; feature
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Route a support ticket to the right queue.
```

**What to notice.** The instruction is one short line. Optimizers in later examples rewrite it or add demos below it.

## Try it

- Change `TICKET` to a security or how-to ticket and rerun.
- Add a second field to the `clean` signature string, such as `"ticket -> short_summary, tone"`, and rerun.
- Swap `ChainOfThought` for `Predict` in `CleanThenTriage` and compare the prompt.

## Read next

- [Walkthrough, step 1](../walkthrough.md#step-1-see-what-dspy-sends-01_signatures_and_modulespy)
- [Concepts](../concepts.md)
