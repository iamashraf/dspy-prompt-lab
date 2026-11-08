"""08. What people get wrong: judging a program on the data it was optimized on.

An optimizer that sees your examples can memorize them. The score on the trainset then
overstates how the program does on new inputs. This script optimizes with a growing trainset
and prints three numbers per run:

  train  - score on the examples the optimizer saw (flattering, do not trust)
  dev    - score on examples used to compare and pick between configurations
  test   - score on examples nobody touched; look at it once, at the end

Run:  uv run python examples/08_overfitting_and_splits.py
Cost: 3 optimizer runs + 9 evaluations. Free on Ollama, about $0.10 on gpt-4o-mini.
Debug: if train is high and dev is low, you are fitting the examples, not the task. Add data or
       reduce demos before trying a stronger optimizer.
Docs: https://dspy.ai/learn/evaluation/data/  (train/dev/test guidance)
"""

import time

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result


def score(program, split):
    return dspy.Evaluate(devset=split, metric=label_match, num_threads=4)(program).score


def main():
    started = time.time()
    lm = configure_lm()
    _, dev, test = load_tickets()
    baseline_test = score(make_triage(), test)

    print(f"{'trainset':>10} {'train':>7} {'dev':>7}")
    best = None
    for per_label in (1, 3, 6):  # 6, 18, 36 training tickets
        train, _, _ = load_tickets(n_train=per_label)
        # Labeled demos are copied from the trainset into the prompt, so they are the easiest to memorize.
        compiled = dspy.LabeledFewShot(k=len(train)).compile(make_triage(), trainset=train)
        row = (score(compiled, train), score(compiled, dev))
        print(f"{len(train):>10} {row[0]:>6.1f}% {row[1]:>6.1f}%")
        if best is None or row[1] > best[0]:
            best = (row[1], compiled)

    # Choose by dev, report once on test.
    after = score(best[1], test)
    print("\nWith few examples, train is near perfect because the answers are in the prompt, while dev")
    print("is lower. The gap, not the train score, is the warning.")
    print_result("08_overfitting", baseline_test, after, lm, started)


if __name__ == "__main__":
    main()
