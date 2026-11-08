"""11. KNNFewShot: pick different demos for every input.

Other optimizers fix one demo set for all inputs. KNNFewShot embeds the training examples and,
at call time, retrieves the k most similar ones to the *current* input and uses them as demos.
Good when the right example depends on the input, as with house rules that apply to some tickets
and not others.

Needs an embedding model: nomic-embed-text on Ollama (`ollama pull nomic-embed-text`), or
text-embedding-3-small on OpenAI. Anthropic has no embeddings: set DSPY_LAB_EMBED_PROVIDER.

Run:  uv run python examples/11_knn_fewshot.py
Cost: embeddings for 36 tickets, then one call per test ticket, because this script asks for labeled
      neighbours only (max_bootstrapped_demos=0). With DSPy's default (bootstrapped demos) the
      optimizer re-runs a mini-bootstrap for every query: about k+1 calls per ticket, which took
      ~11 minutes on a local 30B model for 36 tickets.
Debug: `lm.history[-1]["messages"]` shows which demos were retrieved for the last call. Bad
       neighbours usually mean a weak embedding model or too small a trainset.
Docs: https://dspy.ai/api/optimizers/KNNFewShot/
"""

import time

import dspy

from dspy_prompt_lab.data import load_tickets
from dspy_prompt_lab.lm import configure_lm, get_embedder
from dspy_prompt_lab.metrics import label_match
from dspy_prompt_lab.programs import make_triage
from dspy_prompt_lab.report import print_result


def main():
    started = time.time()
    lm = configure_lm()
    train, _, test = load_tickets()
    evaluate = dspy.Evaluate(devset=test, metric=label_match, num_threads=4)
    baseline = evaluate(make_triage()).score

    optimizer = dspy.KNNFewShot(
        k=6,
        trainset=train,
        vectorizer=get_embedder(),
        max_bootstrapped_demos=0,  # labeled neighbours only: one LM call per query
        max_labeled_demos=6,
    )
    compiled = optimizer.compile(make_triage())  # no metric: it only retrieves, it does not search

    compiled(ticket=test[0].ticket)
    print("Demos chosen for one test ticket:")
    print("  ticket:", test[0].ticket)
    for message in lm.history[-1]["messages"]:
        if message["role"] == "user" and "[[ ## label" not in message["content"]:
            print("  demo input:", message["content"].splitlines()[1][:80])
    print_result("11_knn_fewshot", baseline, evaluate(compiled).score, lm, started)


if __name__ == "__main__":
    main()
