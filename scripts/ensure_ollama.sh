#!/usr/bin/env bash
# Start a local Ollama only if nothing is already listening on its port.
set -euo pipefail

if command -v podman &> /dev/null; then
  COMPOSE="podman compose"
elif command -v docker &> /dev/null; then
  COMPOSE="docker compose"
else
  echo "Neither podman nor docker found. Install Ollama directly: https://ollama.com" >&2
  exit 1
fi

if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Ollama is already running at localhost:11434. Reusing it."
else
  echo "No Ollama at localhost:11434. Starting one with '$COMPOSE'."
  $COMPOSE -f compose.ollama.yaml up -d ollama ollama-init
fi
