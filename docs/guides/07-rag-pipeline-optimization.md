# Example 07: Optimize a RAG pipeline

You will check retrieval and generation separately, then optimize the generator prompt. The result shows why a small dev set cannot confirm a gain.

| | |
|---|---|
| Run | `uv run python examples/07_rag_pipeline_optimization.py` |
| Model calls | 40 |
| Reference run | answer accuracy 88.9% to 83.3% on 18 dev questions, 53,131 prompt + 3,956 completion tokens, 479 s |

## Step 1. Check retrieval first

The retriever is a plain keyword index. DSPy does not tune it, so check it before you blame the prompt.

```python
    train, dev = load_help_qa()
    program = RAG(k=3)
```

```python
    hits = [ex.doc_id in [d["id"] for d in program.retriever(ex.question, 3)] for ex in dev]
    print(f"Retrieval hit@3 on dev: {sum(hits)}/{len(hits)}. Answers can only be right when this is high.")
```

Output:

```text
Retrieval hit@3 on dev: 17/18. Answers can only be right when this is high.
```

**What to notice.** Retrieval finds the right doc in the top 3 for 17 of 18 questions. Retrieval is not the bottleneck here.

## Step 2. Score the baseline

```python
    evaluate = dspy.Evaluate(devset=dev, metric=key_in_answer, num_threads=4)
    baseline = evaluate(program).score
    print(f"Baseline RAG answer accuracy: {baseline:.1f}%")
```

Output:

```text
Average Metric: 16 / 18 (88.9%)
Baseline RAG answer accuracy: 88.9%
```

**What to notice.** The baseline is already 16 of 18, so there is little room to gain.

## Step 3. Optimize the generator

`BootstrapFewShot` adds demos to the generator. The retriever stays as it is.

```python
    compiled = dspy.BootstrapFewShot(metric=key_in_answer, max_bootstrapped_demos=4).compile(
        RAG(k=3), trainset=train
    )
    after = evaluate(compiled).score
    print_result("07_rag", baseline, after, lm, started, split="dev")
```

Output:

```text
Bootstrapped 4 full traces after 4 examples for up to 1 rounds, amounting to 4 attempts.
Average Metric: 15 / 18 (83.3%)
RESULT 07_rag: dev 88.9% -> 83.3%  (53,131 prompt + 3,956 completion tokens, 479s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The score went down by one question, from 16 to 15. That is within noise on 18 questions, so this run does not show that optimization helped or hurt. The RESULT line says `dev` because this script scores the dev questions, not the test set.

## Step 4. Read the sample-size note

The script ends with a note about sample size.

```python
    one = 100 / len(dev)
    print(f"Note: {len(dev)} dev questions, so one question is {one:.1f} points. Treat small moves as noise.")
```

Output:

```text
Note: 18 dev questions, so one question is 5.6 points. Treat small moves as noise.
```

**What to notice.** The script computes the question count and the points per question, so the note stays correct if the dev set changes. The local score reproduced across two runs: 88.9% to 83.3% both times.

## Try it

- Change `k=3` to `k=1` in both `RAG` calls and watch hit@3 and accuracy.
- Raise `max_bootstrapped_demos` and rerun.
- Call `dspy.inspect_history(n=1)` after one `program(question=...)` to see the context the model got.

## Read next

- [Step 15 in the walkthrough](../walkthrough.md#step-15-optimize-a-rag-program-07_rag_pipeline_optimizationpy)
- [Optimizers](../optimizers.md)
