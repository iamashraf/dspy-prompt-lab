# Contributing

This repo teaches DSPy prompt optimization. Keep each example short and easy to read.

## Set up

```bash
uv sync
cp .env.example .env       # only needed to run examples against a real model
make test                  # offline, no key or network
make lint
```

## Ground rules

- Tests must pass offline. Use `dspy.utils.DummyLM`, not a real model.
- Never commit a key, a `.env` file, or the contents of `artifacts/`.
- Each example is one idea in one file, with a docstring that has `Run`, `Cost`, `Debug` and `Docs` lines.
- If you change an example, run it and report the result it prints. Do not tune settings on the test split.
- Match the surrounding style. `ruff check .` must be clean.

## Adding an example

1. Copy the closest example in `examples/` and keep its shape (`main()`, `configure_lm()`, `print_result()`).
2. Use the shared data and programs in `src/dspy_prompt_lab/`.
3. Run it, read the log in `artifacts/logs/`, and put the real result in the PR description.

## Pull requests

Describe what changed and how you checked it. Small PRs are easier to review.
