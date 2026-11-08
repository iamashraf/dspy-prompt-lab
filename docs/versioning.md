# Versioning optimized prompts

An optimizer's output is a prompt: instructions plus few-shot demos. Change the model, the data or the optimizer settings and the prompt changes. In production you need to know which prompt is live, how it differs from the last one, and whether it is better.

This repo saves each compiled program as a JSON file and compares two of them on the same tickets.

## What gets saved

Example 03 calls `save_version` after it scores each compiled program:

```python
save_version(compiled, "03_" + name.split()[0], results[name][1], "test", lm)
```

Each call writes two things under `artifacts/prompts/`:

- `<name>/<time>.json`, the program from `program.save`: instructions, demos and signature. It holds no API key.
- One row in `index.jsonl` with the score, split, model, DSPy version, git commit and a hash of `data/tickets.jsonl`.

Two versions are comparable only when the model, split and data hash match. `artifacts/` is git-ignored, so these files stay local. The commit in each row ties a prompt to the code that made it.

## Compare two versions

```bash
uv run python scripts/compare_prompts.py \
  artifacts/prompts/03_bootstrapped/20261108-085029.json \
  artifacts/prompts/03_labeled/20261108-085029.json
```

The script prints the instruction diff and the demo overlap. It then scores both versions on the 36 dev tickets and lists every ticket where they disagree. Add `test` as a third argument to score on test instead.

The score in `index.jsonl` is the test score from the example run, so it differs from the dev score here. The two splits answer different questions. Dev picks a version, and test reports it once.

```text
Instructions: identical
Demos: A has 6, B has 16, 2 shared

Dev score: A 77.8%  B 83.3%
  A wrong, B right: [account] How do I leave a workspace I was invited to?
  A right, B wrong: [account] Invites to my teammates bounce back with an error.
```

## Why the flip list matters

The flip list shows which tickets changed, and the score gap alone cannot. It also shows how little one run proves.

The same two versions were compared twice on the same 36 dev tickets, on the same local model with the cache off for the second run. The runs disagreed.

| Run | A | B | Tickets that flipped |
|---|---|---|---|
| 1 | 66.7% | 66.7% | 8 (4 each way) |
| 2 | 77.8% | 83.3% | 4 (3 for B, 1 for A) |

Only one flipped ticket, "Invites to my teammates bounce back with an error", appears in both runs. The local model does not give the same answers on every run, even with the same prompt. One ticket is 2.8 points, so a gap of one or two tickets is noise.

This is the reason to keep versions. A single comparison cannot say which prompt is better. Saved files let you rerun the same two prompts as often as you need.

## Rules that keep the comparison fair

- Select a version on dev. Report test once, at the end (the index score is test, the compare default is dev).
- Compare only versions with the same model and data hash.
- Set `DSPY_LAB_CACHE=0` when you measure cost or time, because cached calls cost nothing.

## Limits

- The script scores each version once. Run it at least three times with `DSPY_LAB_CACHE=0` and trust only the flips that repeat. The cache returns the same answers, so a cached rerun proves nothing.
- The two runs above used `qwen3-coder:30b`. The cause of the run-to-run difference was not isolated.
- The saved files stay local. To track prompts in git, copy the ones you want to keep into a committed folder.
- Only example 03 saves versions. Add `save_version` after any `compile` call to save the others.
