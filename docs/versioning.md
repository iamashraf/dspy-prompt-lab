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

The script prints the instruction diff and the demo overlap. It then scores both versions on the 36 dev tickets and lists every ticket where they disagree.

```text
Instructions: identical
Demos: A has 6, B has 16, 2 shared

Dev score: A 66.7%  B 66.7%
  A right, B wrong: [account] A new teammate says their invitation link has expired.
  A wrong, B right: [feature] How do I add a custom field for budget on my cards?
```

## Why the flip list matters

Both versions score 66.7% on dev, so a score-only check reports no change. The two versions disagree on 8 of 36 tickets, 4 in each direction.

A new prompt can keep the score and still change what users see. It can also fix one ticket type and break another. The flip list shows that, and the score does not.

One ticket is 2.8 points on 36 tickets. Treat a gap of one or two tickets as noise, and read the flips to see whether the change is real.

## Rules that keep the comparison fair

- Select a version on dev. Report test once, at the end.
- Compare only versions with the same model and data hash.
- Set `DSPY_LAB_CACHE=0` when you measure cost or time, because cached calls cost nothing.

## Limits

- The 8 flips come from one run on `qwen3-coder:30b`. The run was not repeated.
- The saved files stay local. To track prompts in git, copy the ones you want to keep into a committed folder.
- Only example 03 saves versions. Add `save_version` after any `compile` call to save the others.
