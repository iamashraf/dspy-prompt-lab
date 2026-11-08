"""09. LabeledFewShot: the simplest optimizer. Copy k training examples into the prompt.

No LM calls to compile, no metric. It picks k random labeled examples as demos. It is the floor
to beat: if a fancier optimizer cannot beat this, it is not worth its cost.

Run:  uv run python examples/09_labeled_fewshot.py
Cost: compile is free (no LM calls). One evaluation of 36 tickets.
Debug: `compiled.predictors()[0].demos` lists the copied examples. The draw is seeded (random.Random(0)): same trainset, same demos, every run.
Docs: https://dspy.ai/api/optimizers/LabeledFewShot/
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
    train, _, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    # sample=True draws k examples with a fixed seed. To get different demos, change the trainset
    # you pass in (reorder it or take a slice), as example 16 does.
    compiled = dspy.LabeledFewShot(k=12).compile(make_triage(), trainset=train)
    print(f"{len(compiled.predictors()[0].demos)} demos copied from the trainset")
    print_result("09_labeled_fewshot", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
