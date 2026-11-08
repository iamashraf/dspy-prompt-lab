"""13. SIMBA: learn from the examples the program is inconsistent on.

SIMBA runs the program several times per example on mini-batches, finds examples where outputs
vary a lot (hard cases), and either adds a demo or writes a rule from the contrast between a good
and a bad attempt. It repeats for `max_steps`. Needs temperature above 0 to see variation, so the
program samples at 0.7 here.

Run:  uv run python examples/13_simba.py
Cost: 131,151 prompt + 6,683 completion tokens, 21 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: SIMBA logs each step's batch scores. If a batch has no variation, nothing is learned: raise
       the temperature, or lower `bsize` so hard examples stand out.
Docs: https://dspy.ai/api/optimizers/SIMBA/
"""

import time

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm, get_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result


def main():
    started = time.time()
    lm = configure_lm()
    train, _, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    def float_metric(gold, pred, trace=None) -> float:
        return float(label_match(gold, pred))  # SIMBA computes with floats; a bool logged a type warning

    optimizer = dspy.SIMBA(
        metric=float_metric, bsize=12, num_candidates=4, max_steps=4, max_demos=4, num_threads=4
    )
    # A separate LM that samples (temperature 0.7, no cache) so repeated runs can differ.
    with dspy.context(lm=get_lm(temperature=0.7, cache=False)):
        compiled = optimizer.compile(make_triage(), trainset=train, seed=0)
    print("\n--- instruction after SIMBA ---")
    print(compiled.predictors()[0].signature.instructions)
    print(f"demos: {len(compiled.predictors()[0].demos)}")
    print_result("13_simba", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
