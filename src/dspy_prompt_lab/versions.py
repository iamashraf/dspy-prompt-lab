"""Save each compiled program as a JSON file plus an index row, so versions can be diffed and compared."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import dspy

from dspy_prompt_lab.data import DATA_DIR

PROMPTS = Path("artifacts/prompts")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_version(program: dspy.Module, name: str, score: float, split: str, lm: dspy.LM) -> Path:
    """Write the program to artifacts/prompts/<name>/<time>.json and append a row to index.jsonl.

    The row records what a comparison needs: model, split, score, dataset hash, DSPy version and
    git commit. Two versions are only comparable when model, split and dataset hash match.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = PROMPTS / name / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))
    row = dict(
        name=name,
        file=str(path),
        split=split,
        score=score,
        model=lm.model,
        dspy=dspy.__version__,
        commit=_git_commit(),
        data=hashlib.sha256((DATA_DIR / "tickets.jsonl").read_bytes()).hexdigest()[:12],
        saved_at=stamp,
    )
    with open(PROMPTS / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return path
