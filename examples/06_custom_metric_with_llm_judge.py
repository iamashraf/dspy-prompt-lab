"""06. A metric that uses an LLM as the judge.

Exact match works for labels. It fails for free-text answers: "CSV only" and "Only the CSV
format is available" mean the same thing, and a string check calls one of them wrong.
An LLM judge reads both and decides.

Use a judge when:  the output is free text, there are many valid phrasings, and you can afford
                   one extra call per example.
Avoid it when:     a cheap exact check exists (labels, numbers, JSON fields), because the judge adds
                   cost, noise, and its own biases.
Rules of thumb:    temperature 0; prefer a different (or stronger) model than the one being judged;
                   spot-check the judge against your own reading before trusting a score;
                   remember an optimizer will exploit any quirk the judge has.

This script scores the same RAG answers with the cheap metric and the judge, then shows where
they disagree. Set DSPY_LAB_JUDGE_MODEL to judge with a different model of the same provider.

Run:  uv run python examples/06_custom_metric_with_llm_judge.py
Cost: 2 x 12 question calls + 12 judge calls. Free on Ollama, about $0.01 on gpt-4o-mini.
Debug: validate a judge before trusting it. Read 10 of its verdicts next to the reference (the
       disagreement list below), and rerun to see if verdicts flip: a judge that flips is noise.
Docs: https://dspy.ai/learn/evaluation/metrics/  https://arxiv.org/abs/2306.05685 (LLM-as-judge)
"""

import os

import dspy

from dspy_prompt_lab.data import load_help_qa
from dspy_prompt_lab.lm import configure_lm, get_lm, print_usage
from dspy_prompt_lab.metrics import key_in_answer
from dspy_prompt_lab.programs import RAG


class Judge(dspy.Signature):
    """Decide whether the answer states the same fact as the reference answer. Ignore wording."""

    question: str = dspy.InputField()
    reference: str = dspy.InputField()
    answer: str = dspy.InputField()
    same_fact: bool = dspy.OutputField()


def make_judge_metric(judge_lm: dspy.LM):
    judge = dspy.Predict(Judge)

    def judge_metric(gold, pred, trace=None) -> bool:
        with dspy.context(lm=judge_lm):  # the judge runs on its own LM, not the program's
            verdict = judge(question=gold.question, reference=gold.answer, answer=pred.answer)
        return bool(verdict.same_fact)

    return judge_metric


def main():
    lm = configure_lm()
    judge_metric = make_judge_metric(get_lm(model=os.getenv("DSPY_LAB_JUDGE_MODEL") or None))
    _, dev = load_help_qa()

    program = RAG()
    rows = []
    for ex in dev:
        pred = program(question=ex.question)
        rows.append((ex, pred, key_in_answer(ex, pred), judge_metric(ex, pred)))

    cheap = sum(r[2] for r in rows) / len(rows) * 100
    judged = sum(r[3] for r in rows) / len(rows) * 100
    print(f"Containment metric: {cheap:.0f}%   LLM judge: {judged:.0f}%   ({len(rows)} questions)")

    print("\nWhere they disagree:")
    for ex, pred, a, b in rows:
        if a != b:
            print(f"  Q: {ex.question}\n    reference: {ex.answer!r}\n    answer:    {pred.answer!r}")
            print(f"    containment={a}  judge={b}")
    print_usage(lm)


if __name__ == "__main__":
    main()
