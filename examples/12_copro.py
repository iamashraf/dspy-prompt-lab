"""12. COPRO: improve the instruction by coordinate ascent.

COPRO asks an LM to write `breadth` new versions of the instruction, scores each on the
trainset, then repeats for `depth` rounds starting from the best. It changes the instruction
only, never the demos. MIPROv2 (example 04) is its more capable successor; COPRO is simpler to read.

Run:  uv run python examples/12_copro.py
Cost: 50,261 prompt + 2,613 completion tokens, 4 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: pass `track_stats=True` to COPRO to keep every candidate instruction and score on
       `compiled.results_best` and `compiled.results_latest` (min/max/avg score per depth). Scores that
       never move mean the metric is flat or, on Ollama, that only 1 candidate was tried.
Docs: https://dspy.ai/api/optimizers/COPRO/
"""

import time

import dspy
import litellm

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

    if lm.model.startswith("ollama"):
        # COPRO asks for `breadth` completions in one call (the `n` parameter). Ollama only returns
        # one, so LiteLLM raises unless told to drop it, and each round then tries a single candidate.
        # A provider that returns several completions per call gives the full breadth.
        litellm.drop_params = True
        print("Ollama: one completion per call, so COPRO tries 1 new instruction per round.")

    optimizer = dspy.COPRO(
        metric=label_match, breadth=4, depth=2, init_temperature=1.0, track_stats=True
    )
    compiled = optimizer.compile(
        make_triage(), trainset=train, eval_kwargs=dict(num_threads=4, display_progress=False)
    )
    print("\n--- optimized instruction ---")
    print(compiled.predictors()[0].signature.instructions)
    for stats in compiled.results_best.values():
        print("best scores by depth:", stats["max"])
    print_result("12_copro", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
