"""Metrics. A metric is `f(gold, pred, trace=None)`; return a bool or a float in [0, 1].

DSPy passes `trace` during optimization (when it is deciding which demos to keep) and
leaves it `None` during evaluation. Returning a plain bool works for both.
"""

import re


def label_match(gold, pred, trace=None) -> bool:
    """Exact match on the ticket label."""
    return str(pred.label).strip().lower() == gold.label


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", str(text).lower())  # 10,000 == 10000
    return re.sub(r"\s+", " ", text).strip()


def key_in_answer(gold, pred, trace=None) -> bool:
    """True when the gold key fact appears as a whole word or number in the answer.

    Cheap, but it misses paraphrases. The word boundary matters: a plain substring test passed
    "does not contain information" for the key "n".
    """
    pattern = rf"(?<![\w$]){re.escape(_norm(gold.key))}(?![\w])"
    return re.search(pattern, _norm(pred.answer)) is not None
