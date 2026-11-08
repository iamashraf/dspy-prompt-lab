"""16. Ensemble: combine several compiled programs and take a majority vote.

Ensemble is not a search. You give it programs you already have (here, three LabeledFewShot
programs, each built from a different third of the trainset) and it runs them all and reduces
the outputs, by default with `dspy.majority`. It trades 3x the calls for stability: one unlucky demo set matters less.

Run:  uv run python examples/16_ensemble.py
Cost: 3 LM calls per ticket at inference. Compile is free.
Debug: if members score alike, voting adds nothing. Members that disagree are where it helps.
Docs: https://dspy.ai/api/optimizers/Ensemble/
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

    # LabeledFewShot is seeded, so the same trainset gives the same demos. Give each member its own
    # slice of the trainset so the members actually differ.
    members = [
        dspy.LabeledFewShot(k=8).compile(make_triage(), trainset=train[i::3]) for i in range(3)
    ]
    for i, member in enumerate(members):
        print(f"member {i}: {evaluate(member).score:.1f}% on test")

    ensemble = dspy.Ensemble(reduce_fn=dspy.majority).compile(members)
    print_result("16_ensemble", baseline, evaluate(ensemble).score, lm, started)


if __name__ == "__main__":
    main()
