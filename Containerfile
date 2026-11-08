# syntax=docker/dockerfile:1

# Build: podman build -t dspy-prompt-lab -f Containerfile .
# Runs the examples in a container. The container reaches a host Ollama through
# OLLAMA_API_BASE, or an OpenAI/Anthropic key passed with --env-file .env.
FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 appuser \
    && mkdir /app \
    && chown appuser:appuser /app

WORKDIR /app
USER appuser

# The uv binary is mounted only for the install steps, so no build tool ends up in the image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer, so a code-only change does not re-download them.
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project --no-dev

COPY --chown=appuser:appuser . .
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

CMD ["python", "examples/02_evaluate_baseline.py"]
