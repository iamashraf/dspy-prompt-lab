"""15. InferRules: induce natural-language rules from the examples.

InferRules bootstraps demos, asks an LM to write candidate *rules* that explain them, and keeps
the rule set that scores best on the validation set. The rules are appended to the instruction,
so you can read what it learned. A good fit for hidden policies like this repo's ticket routing.

Run:  uv run python examples/15_infer_rules.py
Cost: 211,347 prompt + 2,991 completion tokens, 7 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: read the induced rules in the printed instruction. If they are vague, raise `num_rules`
       or give it more bootstrapped demos.
Docs: https://dspy.ai/api/optimizers/InferRules/
"""

import time

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result


def main():
    started = time.time()
    lm = configure_lm()
    train, dev, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    optimizer = dspy.InferRules(
        num_candidates=4, num_rules=6, num_threads=4, metric=label_match, max_bootstrapped_demos=6
    )
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
    print("\n--- instruction with induced rules ---")
    print(compiled.predictors()[0].signature.instructions)
    print_result("15_infer_rules", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
