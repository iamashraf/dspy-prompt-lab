"""10. BootstrapFewShotWithRandomSearch: try many demo sets, keep the best.

BootstrapFewShot (example 03) builds one demo set. Random search builds several candidate
programs (different demos, different order), scores each on a validation set, and returns the
winner. More calls than 03, usually a more reliable result.

Run:  uv run python examples/10_bootstrap_random_search.py
Cost: num_candidate_programs x bootstrapping + scoring each on dev. Free on Ollama (a few minutes).
Debug: `compiled.candidate_programs` holds every candidate with its score. If all scores are equal,
       the valset is too small or too easy to tell them apart.
Docs: https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch/
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

    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=label_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=6,  # the default is 16; 6 keeps this example cheap
        num_threads=4,
    )
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
    print("Candidate scores on dev:", [round(c["score"], 1) for c in compiled.candidate_programs])
    print_result("10_random_search", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
