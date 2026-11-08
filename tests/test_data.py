from collections import Counter

from dspy_prompt_lab.data import LABELS, load_help_docs, load_help_qa, load_tickets
from dspy_prompt_lab.programs import KeywordRetriever


def test_ticket_splits_are_balanced_and_disjoint():
    train, dev, test = load_tickets()
    assert (len(train), len(dev), len(test)) == (36, 36, 36)
    for split in (train, dev, test):
        counts = Counter(e.label for e in split)
        assert set(counts) == set(LABELS) and len(set(counts.values())) == 1
    texts = [e.ticket for e in train + dev + test]
    assert len(texts) == len(set(texts))


def test_tiny_trainset_option():
    train, _, _ = load_tickets(n_train=1)
    assert len(train) == len(LABELS)


def test_inputs_are_marked():
    train, _, _ = load_tickets()
    assert list(train[0].inputs().keys()) == ["ticket"]
    assert "label" in train[0].labels()


def test_every_qa_points_at_a_real_doc_and_key_is_in_its_text():
    doc_text = {d["id"]: d["text"].lower() for d in load_help_docs()}
    train, dev = load_help_qa()
    assert len(train) == len(dev) == 18
    for ex in train + dev:
        if not ex.get("derived"):  # derived answers are computed from the doc, e.g. 8 seats x $8
            assert ex.key.lower() in doc_text[ex.doc_id], ex.question


def test_retriever_finds_the_gold_doc_in_top_3():
    retriever = KeywordRetriever()
    train, dev = load_help_qa()
    hits = [ex.doc_id in [d["id"] for d in retriever(ex.question, 3)] for ex in train + dev]
    assert sum(hits) / len(hits) >= 0.9


def test_splits_are_shuffled_not_sorted_by_label():
    # BootstrapFewShot reads the trainset in order. The first 6 must not be a single label.
    train, _, _ = load_tickets()
    assert len({e.label for e in train[:6]}) > 2
