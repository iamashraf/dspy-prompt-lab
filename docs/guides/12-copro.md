# Example 12: COPRO

You will let an LM rewrite the instruction and keep the best version. COPRO changes only the instruction, never the demos.

| | |
|---|---|
| Run | `uv run python examples/12_copro.py` |
| Model calls | 182 |
| Reference run | test 69.4% to 66.7%, 50,261 prompt + 2,613 completion tokens, 225 s |

## Step 1. Handle Ollama's single completion

COPRO asks for `breadth` completions in one call. Ollama returns only one, so the script tells LiteLLM to drop that parameter.

```python
    if lm.model.startswith("ollama"):
        # COPRO asks for `breadth` completions in one call (the `n` parameter). Ollama only returns
        # one, so LiteLLM raises unless told to drop it, and each round then tries a single candidate.
        # A provider that returns several completions per call gives the full breadth.
        litellm.drop_params = True
        print("Ollama: one completion per call, so COPRO tries 1 new instruction per round.")
```

Output:

```text
Ollama: one completion per call, so COPRO tries 1 new instruction per round.
```

**What to notice.** On Ollama the search is much narrower than `breadth=4` suggests.

## Step 2. Run the search

Each round writes new instructions and scores them on the trainset. `track_stats=True` keeps the scores.

```python
    optimizer = dspy.COPRO(
        metric=label_match, breadth=4, depth=2, init_temperature=1.0, track_stats=True
    )
    compiled = optimizer.compile(
        make_triage(), trainset=train, eval_kwargs=dict(num_threads=4, display_progress=False)
    )
```

Output:

```text
Iteration Depth: 1/2.
At Depth 1/2, Evaluating Prompt Candidate #1/2 for Predictor 1 of 1.
At Depth 1/2, Evaluating Prompt Candidate #2/2 for Predictor 1 of 1.
Iteration Depth: 2/2.
At Depth 2/2, Evaluating Prompt Candidate #1/1 for Predictor 1 of 1.
```

**What to notice.** Depth 1 scored 2 candidates and depth 2 scored 1, far fewer than 4 per round.

## Step 3. Read the new instruction

Print the instruction COPRO chose and the best training score at each depth.

```python
    print("\n--- optimized instruction ---")
    print(compiled.predictors()[0].signature.instructions)
    for stats in compiled.results_best.values():
        print("best scores by depth:", stats["max"])
```

Output:

```text
--- optimized instruction ---
You are an expert support ticket routing specialist. Your task is to analyze incoming support tickets and route them to the appropriate department or queue based on the ticket content, severity level, and specialized requirements. Consider factors such as technical complexity, customer impact, product category, and expertise needed. Evaluate the ticket description, any error messages, affected sys...
best scores by depth: [75.0, 75.0]
```

**What to notice.** The instruction is long and generic. It has no word of the house rules, such as which label wins when a ticket fits two.

## Step 4. Score on test

```python
    print_result("12_copro", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
RESULT 12_copro: test 69.4% -> 66.7%  (50,261 prompt + 2,613 completion tokens, 225s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** The best training score was 75.0%, but the test score is below the 69.4% baseline. The drop is one ticket, so treat it as no gain. This repo did not test Anthropic.

## Try it

- Raise `depth` to 3 and check whether the best score moves.
- Run with a provider that returns several completions per call and compare `breadth=4` results.
- Print `compiled.results_latest` to see min, max and average score per depth.

## Read next

- [Walkthrough, step 10](../walkthrough.md)
- [Optimizers](../optimizers.md)
