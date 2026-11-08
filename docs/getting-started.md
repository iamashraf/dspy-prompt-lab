# Getting started

This page gets one example running, then explains the setup choices. It takes about 10 minutes plus model download time.

## Prerequisites

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- One language model, either a local one through [Ollama](https://ollama.com) or an OpenAI or Anthropic API key.
- About 20 GB of free disk if you run a local 30B model, or about 5 GB for `llama3.1:8b`.

## Install

```bash
git clone https://github.com/iamashraf/dspy-prompt-lab.git
cd dspy-prompt-lab
uv sync
make test        # offline tests. No model, key, or network needed.
```

If `make test` passes, the code works on your machine. The next step adds a model.

## Pick a model

Copy the example config and choose a provider.

```bash
cp .env.example .env
```

| Provider | Set in `.env` | Notes |
|---|---|---|
| Ollama (default) | `DSPY_LAB_PROVIDER=ollama` | Free. Runs on your machine. No key. |
| OpenAI | `DSPY_LAB_PROVIDER=openai` and `OPENAI_API_KEY=...` | Default model `gpt-4o-mini`. |
| Anthropic | `DSPY_LAB_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` | Default model `claude-haiku-5-5`. Has no embeddings, see below. |

Set `DSPY_LAB_MODEL` to use a different model from the same provider. `.env` is git-ignored. The repo never stores a key, and a missing key stops the run with a message that names the variable.

### Ollama with podman

If Ollama is not installed, start it in a container.

```bash
make ollama      # starts Ollama with podman (or docker) only if none is listening on :11434
```

`make ollama` runs `compose.ollama.yaml`, which also pulls `DSPY_LAB_MODEL` (default `llama3.1:8b`). With a native Ollama install, run `ollama pull llama3.1:8b` instead.

### Embeddings

Only example 11 needs embeddings. Ollama uses `nomic-embed-text` (`ollama pull nomic-embed-text`). OpenAI uses `text-embedding-3-small`. Anthropic has no embeddings API, so set `DSPY_LAB_EMBED_PROVIDER=ollama` or `openai` as well.

## Run an example

```bash
uv run python examples/02_evaluate_baseline.py
```

You should see the dataset sizes, a baseline score, and the first few mistakes. Every example ends with a line like this.

```text
RESULT 03_bootstrap_fewshot: test 69.4% -> 86.1%  (62,063 prompt + 1,566 completion tokens, 175s, ollama_chat/qwen3-coder:30b)
```

Your numbers will differ from the ones in these docs. See [results](results.md) for why. Then follow the [walkthrough](walkthrough.md), which runs the experiments in a useful order and explains what each one shows.

## What a run writes

Each run writes to `artifacts/` (git-ignored).

| File | Content |
|---|---|
| `artifacts/logs/<example>-<time>.log` | DSPy's log: scores, optimizer progress, warnings |
| `artifacts/logs/<example>-<time>.trace.jsonl` | Every model call: messages sent, reply, token counts |
| `artifacts/results.jsonl` | One line per finished example, for comparing runs |
| `artifacts/triage_compiled.json` | The saved program from example 05 |

See [debugging](debugging.md) for how to use them.

## Cost and time

Each example's docstring lists its measured tokens. The measured numbers in [results](results.md) come from a local model, so they cost nothing but time.

To estimate a hosted-model bill, multiply the prompt and completion tokens by your provider's current per-token prices. The cheap examples (02, 06, 09, 11) use under 40,000 prompt tokens. The search-based ones (04, 08, 10, 13, 15) use 130,000 to 250,000.

DSPy caches model replies on disk, so a repeat run of the same example is free and fast. Set `DSPY_LAB_CACHE=0` when you want to measure real cost or time.

## Run in a container

```bash
podman build -t dspy-prompt-lab -f Containerfile .
podman run --rm --env-file .env -e OLLAMA_API_BASE=http://host.containers.internal:11434 dspy-prompt-lab
```

The image runs example 02. The container reaches Ollama on your host through `OLLAMA_API_BASE`, or uses an API key from `.env`. Pass another example as the last argument, for example `python examples/03_bootstrap_fewshot.py`.

## Common problems

| Message | Fix |
|---|---|
| `Cannot reach Ollama at ...` | Run `make ollama`, or check `OLLAMA_API_BASE`. |
| `Ollama has no model '...'` | Run `ollama pull <model>`. |
| `DSPY_LAB_PROVIDER=openai needs OPENAI_API_KEY` | Set the key in `.env`. |
| A run takes very long | A 30B model on CPU is slow. Use a smaller model or a hosted one. |
| Scores differ from the docs | Expected. See [results](results.md#why-your-numbers-will-differ). |

More in [debugging](debugging.md).
