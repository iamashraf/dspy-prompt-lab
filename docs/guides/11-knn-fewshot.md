# Example 11: KNNFewShot

You will pick different demos for every ticket by retrieving the most similar training tickets. This helps when the right example depends on the input.

| | |
|---|---|
| Run | `uv run python examples/11_knn_fewshot.py` |
| Model calls | 73 |
| Reference run | test 69.4% to 83.3%, 26,119 prompt + 970 completion tokens, 131 s |

This example needs a local Ollama with `nomic-embed-text` for embeddings, even when the chat model is Anthropic.

## Step 1. Build the optimizer

`KNNFewShot` embeds the training tickets. It takes the trainset and the embedder up front.

```python
    optimizer = dspy.KNNFewShot(
        k=6,
        trainset=train,
        vectorizer=get_embedder(),
        max_bootstrapped_demos=0,  # labeled neighbours only: one LM call per query
        max_labeled_demos=6,
    )
    compiled = optimizer.compile(make_triage())  # no metric: it only retrieves, it does not search
```

**What to notice.** `compile` takes no metric and no trainset. With `max_bootstrapped_demos=0` each ticket costs one call. DSPy's default bootstraps per query, about k+1 calls per ticket, and took 687 s in an earlier run.

## Step 2. Inspect the retrieved demos

Call the program once and read the demos from the LM history.

```python
    compiled(ticket=test[0].ticket)
    print("Demos chosen for one test ticket:")
    print("  ticket:", test[0].ticket)
    for message in lm.history[-1]["messages"]:
        if message["role"] == "user" and "[[ ## label" not in message["content"]:
            print("  demo input:", message["content"].splitlines()[1][:80])
```

Output:

```text
Demos chosen for one test ticket:
  ticket: We want to downgrade from Business to Team before the next renewal.
  demo input: Can I get an audit log of changes on our boards?
  demo input: The audit log vanished after we downgraded. Is that a bug?
  demo input: Any plan to support Microsoft Teams notifications like you do for Slack?
  demo input: I was charged twice for my Team plan this month, please refund one.
  demo input: I can't sign in with Google since my company changed domains.
  demo input: Do you have a data processing agreement we can sign? Our legal team is asking.
```

**What to notice.** Some neighbours match (downgrade, Team plan). Others look loosely related (Microsoft Teams, Google sign-in). A weak embedding model or a small trainset can cause that.

## Step 3. Score the compiled program

```python
    print_result("11_knn_fewshot", baseline, evaluate(compiled).score, lm, started)
```

Output:

```text
RESULT 11_knn_fewshot: test 69.4% -> 83.3%  (26,119 prompt + 970 completion tokens, 131s, ollama_chat/qwen3-coder:30b)
```

**What to notice.** It matches example 09 (83.3%) with fewer prompt tokens (26,119 against 35,038). The difference in score is zero tickets.

On Anthropic `claude-haiku-5-5` (one run) the score went from 77.8% to 75.0%, using 40,199 prompt + 4,197 completion tokens in 17 s. That drop is one ticket, which is within noise.

## Try it

- Change `k=6` and `max_labeled_demos=6` to 3 and compare.
- Set `max_bootstrapped_demos=2` to see the slower default behaviour. Expect many more calls.
- Print the ticket and demos for `test[1]` and judge whether the neighbours look relevant.

## Read next

- [Walkthrough, step 6](../walkthrough.md)
- [Optimizers](../optimizers.md)
