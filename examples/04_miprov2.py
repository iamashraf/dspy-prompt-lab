"""04. MIPROv2: optimize the instruction *and* the demos together.

MIPROv2 does three things:
  1. bootstraps candidate demo sets (like example 03),
  2. asks an LM to propose candidate instructions, grounded in your data and program,
  3. searches over the (instruction, demos) combinations with Bayesian optimization,
     scoring each on a validation set.

`auto="light"` picks a small search budget. This is the one to reach for when the task has
rules the model has to be *told* (here: the labeling policy), not just shown.

Run:  uv run python examples/04_miprov2.py
Cost: 208,359 prompt + 7,283 completion tokens, 14 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: the log shows each trial's score. `track_stats=True` (default) keeps them on `compiled.trial_logs`.
       Pass `verbose=True` to MIPROv2 for the full proposal log, and `log_dir=` to keep it. If the
       instruction did not change, the best candidate was the original: compare scores in the log.
Docs: https://dspy.ai/api/optimizers/MIPROv2/  https://arxiv.org/abs/2406.11695
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

    program = make_triage()
    baseline = evaluate(program).score
    print(f"Baseline on test: {baseline:.1f}%")
    print("Baseline instruction:", program.predictors()[0].signature.instructions)

    optimizer = dspy.MIPROv2(
        metric=label_match,
        auto="light",  # "light" | "medium" | "heavy": how big the search is
        num_threads=4,
        seed=0,
    )
    compiled = optimizer.compile(
        program,
        trainset=train,
        valset=dev,  # candidates are scored here, never on test
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    # The result: a new instruction and new demos, both inspectable.
    predictor = compiled.predictors()[0]
    print("\n--- optimized instruction ---")
    print(predictor.signature.instructions)
    print(f"\n--- {len(predictor.demos)} demos ---")
    for demo in predictor.demos:
        print(f"  {demo.ticket[:60]!r} -> {demo.label}")

    scores = [t.get("full_eval_score") for t in compiled.trial_logs.values()]
    print(f"\nValidation score of each full evaluation (from trial_logs): {scores}")

    after = evaluate(compiled).score
    print_result("04_miprov2", baseline, after, lm, started)


if __name__ == "__main__":
    main()
