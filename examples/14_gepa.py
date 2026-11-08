"""14. GEPA: reflective optimization from textual feedback.

Most optimizers see only a number. GEPA also reads *feedback text* from your metric, such as
"you said howto, the answer was feature", and uses a reflection LM to rewrite the instruction
from those mistakes. The metric returns `dspy.Prediction(score=..., feedback=...)`.

Use a stronger model for reflection if you have one: set DSPY_LAB_REFLECTION_MODEL.

Run:  uv run python examples/14_gepa.py
Cost: capped here by max_metric_calls=300, plus reflection calls. Free on Ollama (several minutes).
Debug: GEPA's value is its feedback text. Print `feedback_metric` output for 3 failures and ask if
       a human could fix the prompt from it. `track_stats=True` returns `detailed_results`;
       `log_dir=` keeps checkpoints so a run can resume.
Docs: https://dspy.ai/api/optimizers/GEPA/overview/  https://arxiv.org/abs/2507.19457
"""

import os
import time

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm, get_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result


def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score 1/0 and say why. The feedback text is what GEPA's reflection LM reads."""
    ok = label_match(gold, pred)
    if ok:
        return dspy.Prediction(score=1.0, feedback=f"Correct: {gold.label}.")
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Wrong. Predicted '{pred.label}' but the correct label is '{gold.label}'. "
            f"Ticket: {gold.ticket!r}. Work out what rule separates these two labels."
        ),
    )


def main():
    started = time.time()
    lm = configure_lm()
    train, dev, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    reflection_lm = get_lm(
        model=os.getenv("DSPY_LAB_REFLECTION_MODEL") or None, temperature=1.0, max_tokens=4000
    )
    optimizer = dspy.GEPA(
        metric=feedback_metric,
        max_metric_calls=300,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        num_threads=4,
        track_stats=True,  # adds `detailed_results` to the returned program
        seed=0,
    )
    compiled = optimizer.compile(make_triage(), trainset=train, valset=dev)
    print("\n--- instruction after GEPA ---")
    print(compiled.predictors()[0].signature.instructions)
    print("candidate val scores:", [round(s, 2) for s in compiled.detailed_results.val_aggregate_scores])
    print_result("14_gepa", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
