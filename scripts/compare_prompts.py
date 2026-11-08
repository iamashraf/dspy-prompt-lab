"""Compare two saved prompt versions on the dev split.

Run:  uv run python scripts/compare_prompts.py artifacts/prompts/A.json artifacts/prompts/B.json

Prints the instruction diff, the demos that changed, then scores both versions on the same dev
tickets and lists the tickets that flipped. One ticket is 2.8 points on 36, so read the flips,
not only the score gap.
"""

import difflib
import sys

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage


def load(path: str) -> dspy.Module:
    program = make_triage()
    program.load(path)
    return program


def main(path_a: str, path_b: str):
    configure_lm()
    a, b = load(path_a), load(path_b)
    pa, pb = a.predictors()[0], b.predictors()[0]

    diff = list(difflib.unified_diff(
        pa.signature.instructions.splitlines(), pb.signature.instructions.splitlines(), "A", "B", lineterm=""
    ))
    print("Instructions:", "\n".join(diff) if diff else "identical")
    tickets_a = {d["ticket"] for d in pa.demos}
    tickets_b = {d["ticket"] for d in pb.demos}
    print(f"Demos: A has {len(tickets_a)}, B has {len(tickets_b)}, {len(tickets_a & tickets_b)} shared")

    _, dev, _ = load_tickets()
    evaluate = dspy.Evaluate(devset=dev, metric=label_match, num_threads=4)
    ra, rb = evaluate(a), evaluate(b)
    print(f"\nDev score: A {ra.score:.1f}%  B {rb.score:.1f}%")
    for (ex, _, sa), (_, _, sb) in zip(ra.results, rb.results, strict=True):
        if bool(sa) != bool(sb):
            print(f"  {'A wrong, B right' if sb else 'A right, B wrong'}: [{ex.label}] {ex.ticket[:70]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
