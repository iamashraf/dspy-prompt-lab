"""Programs shared by the examples, so every example optimizes the same thing."""

import math
import re
from collections import Counter
from typing import Literal

import dspy

from dspy_prompt_lab.data import load_help_docs


class Triage(dspy.Signature):
    """Route a support ticket to the right queue."""

    ticket: str = dspy.InputField()
    label: Literal["billing", "bug", "howto", "account", "security", "feature"] = dspy.OutputField()


def make_triage(reason: bool = False) -> dspy.Module:
    """The baseline program. `reason=True` adds a reasoning step with ChainOfThought."""
    return (dspy.ChainOfThought if reason else dspy.Predict)(Triage)


_WORD = re.compile(r"[a-z0-9$]+")


class KeywordRetriever:
    """Tiny in-memory TF-IDF retriever. Good enough to teach RAG, not for production."""

    def __init__(self, docs: list[dict] | None = None):
        self.docs = docs or load_help_docs()
        self.tokens = [
            Counter(_WORD.findall(f"{d['title']} {d['text']}".lower())) for d in self.docs
        ]
        df = Counter(t for toks in self.tokens for t in toks)
        n = len(self.docs)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def __call__(self, query: str, k: int = 3) -> list[dict]:
        q = set(_WORD.findall(query.lower()))
        scores = [sum(self.idf.get(t, 0) for t in q if t in toks) for toks in self.tokens]
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[:k]]


class RAG(dspy.Module):
    """Retrieve help-center docs, then answer from them."""

    def __init__(self, k: int = 3, retriever: KeywordRetriever | None = None):
        super().__init__()
        self.k = k
        self.retriever = retriever or KeywordRetriever()
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str):
        context = [f"{d['title']}: {d['text']}" for d in self.retriever(question, self.k)]
        return self.answer(context=context, question=question)
