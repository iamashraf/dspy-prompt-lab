"""03. BootstrapFewShot: let the model write its own few-shot examples.

For each training ticket, BootstrapFewShot runs your program and keeps the runs the metric marks
correct, then adds them to the prompt as demonstrations ("bootstrapped" demos). It can also copy
raw training examples in ("labeled" demos). It changes demos only; the instruction text stays the
same (that is MIPROv2's job, example 04).

The catch this script shows: a bootstrapped demo is a run the model got right on its own, so it
teaches what the model already knew. The hidden labeling rules live in the tickets it gets
*wrong*, and those only reach the prompt as labeled demos.

Run:  uv run python examples/03_bootstrap_fewshot.py
Cost: about 36 training calls per optimizer run, 3 evaluations of 36. Free on Ollama.
Debug: if the log says "Bootstrapped 0 full traces", the metric rejected every run: test the metric by
       hand on one gold/prediction pair. Print `predictor.demos` to see what reached the prompt.
Docs: https://dspy.ai/api/optimizers/BootstrapFewShot/
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
    print(f"Baseline on test: {baseline:.1f}%")

    results = {}
    for name, bootstrapped, labeled in [
        ("bootstrapped demos only", 6, 0),
        ("labeled demos only", 0, 16),
    ]:
        optimizer = dspy.BootstrapFewShot(
            metric=label_match,
            max_bootstrapped_demos=bootstrapped,  # runs the model produced itself and the metric accepted
            max_labeled_demos=labeled,  # training examples copied in as-is
        )
        compiled = optimizer.compile(make_triage(), trainset=train)  # returns a new program
        results[name] = (compiled, evaluate(compiled).score)
        print(f"{name:>26}: {len(compiled.predictors()[0].demos):>2} demos -> {results[name][1]:.1f}%")

    compiled, after = results["labeled demos only"]
    print("\n--- the compiled prompt: one call with the labeled-demo program ---")
    compiled(ticket=test[0].ticket)
    dspy.inspect_history(n=1)
    print_result("03_bootstrap_fewshot", baseline, after, lm, started)
    print("Test has 36 tickets, so one ticket is 2.8 points. Read gaps under ~8 points as noise.")


if __name__ == "__main__":
    main()
