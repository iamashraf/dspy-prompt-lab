"""One result line per example, so the runs are comparable and the docs table is real."""

import json
import time
from pathlib import Path

import dspy

from dspy_prompt_lab.lm import token_usage

RESULTS = Path("artifacts/results.jsonl")


def print_result(
    name: str, baseline: float, after: float, lm: dspy.LM, started: float, split: str = "test"
) -> None:
    """Print the before/after line and append it to artifacts/results.jsonl."""
    prompt, completion = token_usage()
    seconds = time.time() - started
    print(
        f"\nRESULT {name}: {split} {baseline:.1f}% -> {after:.1f}%  "
        f"({prompt:,} prompt + {completion:,} completion tokens, {seconds:.0f}s, {lm.model})"
    )
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        row = dict(
            example=name,
            split=split,
            baseline=baseline,
            after=after,
            prompt_tokens=prompt,
            completion_tokens=completion,
            seconds=round(seconds),
            model=lm.model,
        )
        f.write(json.dumps(row) + "\n")
