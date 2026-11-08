"""07. Optimize a retrieval-augmented program end to end.

The program (`RAG` in src/dspy_prompt_lab/programs.py) has two parts:
  * a retriever: a small in-memory keyword index over 15 help-center docs. No vector database.
  * a generator: `ChainOfThought("context, question -> answer")`.

DSPy optimizes the generator's prompt. The retriever is plain Python and is *not* tuned, which is
the limit of this example: if retrieval misses, no prompt fixes it. The script checks
retrieval separately so you can tell the two failure modes apart.

Run:  uv run python examples/07_rag_pipeline_optimization.py
Cost: 53,491 prompt + 4,272 completion tokens, 6 min on a local 30B model (Ollama).
      Multiply by your provider's token prices for a hosted model.
Debug: check retrieval before blaming the prompt (hit@3 below). To see the context the model got,
       call `dspy.inspect_history(n=1)` after one `program(question=...)`.
Docs: https://dspy.ai/tutorials/rag/
"""

import time

import dspy

from dspy_prompt_lab.data import load_help_qa
from dspy_prompt_lab.lm import configure_lm
from dspy_prompt_lab.metrics import key_in_answer
from dspy_prompt_lab.programs import RAG
from dspy_prompt_lab.report import print_result


def main():
    started = time.time()
    lm = configure_lm()
    train, dev = load_help_qa()
    program = RAG(k=3)

    # Separate the two failure modes: did retrieval find the right doc?
    hits = [ex.doc_id in [d["id"] for d in program.retriever(ex.question, 3)] for ex in dev]
    print(f"Retrieval hit@3 on dev: {sum(hits)}/{len(hits)}. Answers can only be right when this is high.")

    evaluate = dspy.Evaluate(devset=dev, metric=key_in_answer, num_threads=4)
    baseline = evaluate(program).score
    print(f"Baseline RAG answer accuracy: {baseline:.1f}%")

    compiled = dspy.BootstrapFewShot(metric=key_in_answer, max_bootstrapped_demos=4).compile(
        RAG(k=3), trainset=train
    )
    after = evaluate(compiled).score
    print_result("07_rag", baseline, after, lm, started)
    print("Note: 12 dev questions, so one question is 8 points. Treat small moves as noise.")


if __name__ == "__main__":
    main()
