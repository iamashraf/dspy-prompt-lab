"""05. Save an optimized program and load it later without re-optimizing.

Optimization costs money and time. Do it once, save the result, and load it in production.

A DSPy program's learned state is small: the instruction and the demos per predictor. `save()`
writes that state to JSON. To use it, build the same program class and call `load()`.

Run:  uv run python examples/05_save_load_compiled.py
Cost: 48,766 prompt + 1,077 completion tokens, 2 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: open the saved JSON, it is plain text. If a load looks like it did nothing, compare
       `predictor.signature.instructions` and `len(predictor.demos)` before and after. Load into
       the same program class that was saved: names of predictors must match.
Docs: https://dspy.ai/tutorials/saving/
Safety: `save(path)` to a .json file is plain data. `save(path, save_program=True)` writes a
        pickled whole program, and loading it runs code. Load pickles only from sources you trust.
"""

import time
from pathlib import Path

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result

PATH = Path("artifacts/triage_compiled.json")


def main():
    started = time.time()
    lm = configure_lm()
    train, _, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    # Step 1 (run once): optimize and save.
    compiled = dspy.BootstrapFewShot(metric=label_match, max_bootstrapped_demos=6).compile(
        make_triage(), trainset=train
    )
    PATH.parent.mkdir(exist_ok=True)
    compiled.save(PATH)
    print(f"Saved {PATH} ({PATH.stat().st_size:,} bytes). Head of the file:")
    print(PATH.read_text()[:300], "...")

    # Step 2 (every later run): build the same program, load the state. No optimizer, no training calls.
    loaded = make_triage()
    loaded.load(PATH)
    # Saved demos come back as plain dicts, not dspy.Example objects. The content is what matters.
    same = [dict(d) for d in loaded.predictors()[0].demos] == [
        dict(d) for d in compiled.predictors()[0].demos
    ]
    print(f"\nLoaded {len(loaded.predictors()[0].demos)} demos. Same as the saved program: {same}")

    after = evaluate(loaded).score
    print_result("05_save_load", baseline, after, lm, started)


if __name__ == "__main__":
    main()
