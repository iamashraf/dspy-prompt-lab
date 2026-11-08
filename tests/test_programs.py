import dspy
from dspy.utils import DummyLM

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.metrics import key_in_answer, label_match
from dspy_prompt_lab.programs import RAG, make_triage


def test_label_match_normalizes_case():
    gold = dspy.Example(label="bug")
    assert label_match(gold, dspy.Prediction(label=" Bug "))
    assert not label_match(gold, dspy.Prediction(label="billing"))


def test_key_in_answer():
    gold = dspy.Example(key="60")
    assert key_in_answer(gold, dspy.Prediction(answer="For 60 days."))
    assert not key_in_answer(gold, dspy.Prediction(answer="Forever."))


def test_evaluate_scores_a_scripted_model():
    _, dev, _ = load_tickets()
    # DummyLM replays answers in order, so one call per dev example, all "bug".
    with dspy.context(lm=DummyLM([{"label": "bug"}] * len(dev))):
        result = dspy.Evaluate(devset=dev, metric=label_match, num_threads=1)(make_triage())
    assert result.score == 16.67  # 6 of the 36 dev tickets are bugs (score is a rounded percent)


def test_bootstrap_stops_at_max_bootstrapped_demos():
    train, _, _ = load_tickets(n_train=1)  # one per label, in LABELS order
    answers = [{"label": e.label} for e in train]
    with dspy.context(lm=DummyLM(answers)):
        compiled = dspy.BootstrapFewShot(metric=label_match, max_bootstrapped_demos=2).compile(
            make_triage(), trainset=train
        )
    # Bootstrapped demos are flagged `augmented`; the rest are labeled examples used as padding.
    assert sum(bool(d.get("augmented")) for d in compiled.demos) == 2


def test_rag_passes_retrieved_context_to_the_model():
    lm = DummyLM([{"reasoning": "r", "answer": "60 days"}])
    with dspy.context(lm=lm):
        out = RAG()(question="How long can archived cards be restored?")
    assert out.answer == "60 days"
    prompt = lm.history[-1]["messages"][-1]["content"] + lm.history[-1]["messages"][0]["content"]
    assert "Archived cards are kept for 60 days" in prompt


def test_every_example_is_valid_python_and_has_a_main():
    import ast
    from pathlib import Path

    for path in sorted(Path(__file__).parents[1].glob("examples/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        assert ast.get_docstring(tree), f"{path.name} needs a docstring with Run/Cost/Docs"
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), path.name


def test_key_in_answer_needs_a_whole_word_and_ignores_thousands_commas():
    gold = dspy.Example(key="n")
    assert not key_in_answer(gold, dspy.Prediction(answer="The context has no information."))
    assert key_in_answer(gold, dspy.Prediction(answer="Press N to add a card."))
    big = dspy.Example(key="10,000")
    assert key_in_answer(big, dspy.Prediction(answer="Up to 10000 cards."))
    assert not key_in_answer(dspy.Example(key="$6"), dspy.Prediction(answer="$60 a month"))
