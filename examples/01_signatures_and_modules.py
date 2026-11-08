"""01. The DSPy programming model: signatures, modules, and a custom module.

DSPy replaces a hand-written prompt with two things:
  * a *signature*: what goes in and what comes out, declared as fields
  * a *module*: a strategy for turning the signature into LM calls (Predict, ChainOfThought, ...)

Nothing is optimized here. This file only shows what you write, and what DSPy sends.

Run:  uv run python examples/01_signatures_and_modules.py
Cost: 1,055 prompt + 219 completion tokens, 16 s on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: `dspy.inspect_history(n=1)` prints the exact prompt and reply of the last call (used below).
       Every run also writes artifacts/logs/<example>-<time>.log and a .trace.jsonl of all LM calls.
       DSPY_LAB_DEBUG=1 raises the log level to DEBUG. Set cache=False in get_lm() if you suspect a stale cached reply.
Docs: https://dspy.ai/learn/programming/signatures/  https://dspy.ai/learn/programming/modules/
"""

import dspy

from dspy_prompt_lab.lm import configure_lm, print_usage
from dspy_prompt_lab.programs import Triage

TICKET = "I was charged twice for my Team plan this month, please refund one."


# A custom module: any class with __init__ (declare sub-modules) and forward (use them).
# Sub-modules assigned in __init__ are what an optimizer later finds and tunes.
class CleanThenTriage(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clean = dspy.Predict("ticket -> short_summary")
        self.triage = dspy.ChainOfThought(Triage)

    def forward(self, ticket: str):
        summary = self.clean(ticket=ticket).short_summary
        return self.triage(ticket=summary)


def main():
    lm = configure_lm()

    # 1. A signature as a string. Field names carry the meaning ("question -> answer").
    qa = dspy.Predict("question -> answer")
    print("string signature  :", qa(question="What is the capital of France? One word.").answer)

    # 2. A signature as a class. Types, a docstring (the instruction) and descriptions
    #    constrain the output. Triage.label is a Literal, so only six values are valid.
    predict = dspy.Predict(Triage)
    print("class signature   :", predict(ticket=TICKET).label)

    # 3. ChainOfThought: same signature, plus a `reasoning` field the model writes first.
    pred = dspy.ChainOfThought(Triage)(ticket=TICKET)
    print("chain of thought  :", pred.label)
    print("  reasoning       :", pred.reasoning.strip()[:200])

    # 4. A custom module that chains two steps.
    program = CleanThenTriage()
    print("custom module     :", program(ticket=TICKET).label)
    print("  sub-modules     :", [name for name, _ in program.named_predictors()])

    # What DSPy actually sent to the model for the last call: the "prompt" you did not write.
    print("\n--- last prompt DSPy built ---")
    dspy.inspect_history(n=1)
    print_usage(lm)


if __name__ == "__main__":
    main()
