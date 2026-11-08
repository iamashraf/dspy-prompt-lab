"""Load the hand-written datasets in `data/`. Nothing is downloaded."""

import json
import random
from pathlib import Path

import dspy

# Works for a clone of the repo (editable install), which is how the examples are run.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

LABELS = ["billing", "bug", "howto", "account", "security", "feature"]


def _read(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tickets(
    n_train: int = 6, n_dev: int = 6, n_test: int = 6
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Split the 108 tickets into train, dev, test with the same count per label.

    The defaults give 36 / 36 / 36. `n_train` is per label, so `load_tickets(n_train=1)`
    gives a 6-example trainset. The split is deterministic. Each split is shuffled because
    optimizers such as BootstrapFewShot read the trainset in order and stop early: a trainset
    sorted by label would show them only the first labels.
    """
    by_label: dict[str, list[dict]] = {label: [] for label in LABELS}
    for row in _read("tickets.jsonl"):
        by_label[row["label"]].append(row)

    train, dev, test = [], [], []
    for rows in by_label.values():
        parts = (
            rows[:n_train],
            rows[n_train : n_train + n_dev],
            rows[n_train + n_dev : n_train + n_dev + n_test],
        )
        for out, part in zip((train, dev, test), parts, strict=True):
            out += [dspy.Example(**r).with_inputs("ticket") for r in part]
    for split in (train, dev, test):
        random.Random(0).shuffle(split)
    return train, dev, test


def load_help_docs() -> list[dict]:
    return _read("help_docs.jsonl")


def load_help_qa() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Return (train, dev) for the help-center questions: 18 each, alternating."""
    rows = _read("help_qa.jsonl")
    examples = [dspy.Example(**r).with_inputs("question") for r in rows]
    return examples[0::2], examples[1::2]
