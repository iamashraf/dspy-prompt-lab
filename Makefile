.PHONY: sync test lint ollama examples

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

# Start a local Ollama only if none is listening (see scripts/ensure_ollama.sh)
ollama:
	./scripts/ensure_ollama.sh

# Run the cheap examples in order. Needs a model configured in .env
examples:
	@for f in examples/0[1-5]*.py; do echo "== $$f"; uv run python $$f || exit 1; done
