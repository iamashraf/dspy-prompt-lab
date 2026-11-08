"""02. Measure first: dataset, metric, dev set, baseline score.

You cannot tell whether an optimizer helped without a number from before. This script
builds that number and is the baseline the later examples try to beat.

The task: route a support ticket for a fictional product (Tasklane) to one of six queues.
The labels follow house rules the model cannot guess, such as "CSV export exists, PDF
export does not, so a PDF request is a `feature`, not a `howto`".

Run:  uv run python examples/02_evaluate_baseline.py
Cost: 16,438 prompt + 953 completion tokens, 2 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: read the failures, not just the score (`show_failures` below). `dspy.Evaluate` also takes
       `provide_traceback=True` and `max_errors=` to surface crashes inside the program, and
       `save_as_json="artifacts/eval.json"` to keep every input, prediction and score for diffing.
Docs: https://dspy.ai/learn/evaluation/data/  https://dspy.ai/learn/evaluation/metrics/
"""

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.debug import show_failures
from dspy_prompt_lab.lm import configure_lm, print_usage
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage


def main():
    lm = configure_lm()
    train, dev, test = load_tickets()
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)} (6 labels, balanced)")
    print("one example:", train[0])

    # num_threads runs calls in parallel. Lower it if your provider rate-limits you.
    program = make_triage()
    result = dspy.Evaluate(devset=dev, metric=label_match, num_threads=4)(program)
    test_score = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)(program).score

    print(f"\nBaseline (Predict, no optimization): dev {result.score:.1f}%   test {test_score:.1f}%")
    print("Later examples optimize on train, select on dev, and report test.")

    # Always read the failures. A score alone does not say what to fix.
    show_failures(result)
    print_usage(lm)


if __name__ == "__main__":
    main()
