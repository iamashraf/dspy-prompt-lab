"""Build a `dspy.LM` from environment variables (and an optional `.env` file).

Provider and model come from `DSPY_LAB_PROVIDER` and `DSPY_LAB_MODEL`. No key is
stored in the repo: keys are read from the environment only.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request

import dspy
import litellm  # noqa: F401  imported eagerly: lazy first import from 4 threads raced and crashed
from dotenv import find_dotenv, load_dotenv

from dspy_prompt_lab.debug import LMS, start_run_log


class LMConfigError(RuntimeError):
    """The environment does not describe a usable language model."""


# prefix is the LiteLLM provider prefix; key_env is None when no key is needed.
PROVIDERS = {
    "ollama": {"prefix": "ollama_chat", "key_env": None, "default_model": "llama3.1:8b"},
    "openai": {"prefix": "openai", "key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "anthropic": {
        "prefix": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-5-5",
    },
}


def _ant_profile_token() -> str | None:
    """Return the OAuth access token of the active `ant` CLI profile, or None.

    `ant auth login` stores a browser-login token. `ant auth print-credentials` refreshes it when it
    is close to expiry. The token is short-lived (hours), so it is read each time an LM is built.
    """
    try:
        out = subprocess.run(
            ["ant", "auth", "print-credentials"], capture_output=True, text=True, timeout=20
        )
        return json.loads(out.stdout).get("access_token") or None
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None


def get_lm(model: str | None = None, **kwargs) -> dspy.LM:
    """Return an LM for the configured provider. Pass `model` to override the model name.

    Extra keyword arguments go to `dspy.LM` (for example `temperature`, `max_tokens`).
    """
    load_dotenv(find_dotenv(usecwd=True))
    provider = os.getenv("DSPY_LAB_PROVIDER", "ollama").strip().lower()
    if provider not in PROVIDERS:
        raise LMConfigError(
            f"DSPY_LAB_PROVIDER={provider!r} is not supported. Use one of: {', '.join(PROVIDERS)}."
        )
    spec = PROVIDERS[provider]
    name = model or os.getenv("DSPY_LAB_MODEL", "").strip() or spec["default_model"]

    # DSPY_LAB_CACHE=0 turns off DSPy's response cache: use it to measure real cost and time.
    cache = os.getenv("DSPY_LAB_CACHE", "1") != "0"
    options = {"max_tokens": 1024, "cache": cache, **kwargs}
    if provider != "anthropic":
        options.setdefault("temperature", 0.0)
    if spec["key_env"]:
        key = os.getenv(spec["key_env"], "").strip()
        if provider == "anthropic":
            # Claude 5 models accept no sampling parameters, and the API rejects temperature.
            # Drop the ones DSPy or an example sets, so runs use the model's own default.
            litellm.drop_params = True
        if not key and provider == "anthropic":
            key = _ant_profile_token() or ""  # keyless: the logged-in `ant` CLI profile
        if not key:
            hint = " or log in with `ant auth login`" if provider == "anthropic" else ""
            raise LMConfigError(
                f"DSPY_LAB_PROVIDER={provider} needs {spec['key_env']}{hint}. "
                "Copy .env.example to .env and set it, or use DSPY_LAB_PROVIDER=ollama."
            )
        options["api_key"] = key
    else:
        options["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        options.setdefault("num_ctx", 8192)  # Ollama's default window is small for few-shot prompts

    # engine="litellm": DSPy 3.4 defaults to a native engine that returned HTTP 404 for Ollama.
    lm = dspy.LM(f"{spec['prefix']}/{name}", engine="litellm", **options)
    LMS.append(lm)  # so token counts and the trace cover this LM too
    return lm


def configure_lm(**kwargs) -> dspy.LM:
    """Build the LM, make it DSPy's default, start the run log (see debug.py), and return it."""
    lm = get_lm(**kwargs)
    if lm.model.startswith("ollama_chat/"):
        _check_ollama(lm.kwargs["api_base"], lm.model.removeprefix("ollama_chat/"))
    dspy.configure(lm=lm)
    start_run_log(lm)
    return lm


def _check_ollama(api_base: str, model: str) -> None:
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=3) as resp:
            tags = resp.read().decode()
    except (urllib.error.URLError, OSError) as exc:
        raise LMConfigError(
            f"Cannot reach Ollama at {api_base} ({exc}). Start it with `make ollama`, "
            "or set DSPY_LAB_PROVIDER=openai|anthropic in .env."
        ) from exc
    if f'"{model}"' not in tags:
        raise LMConfigError(f"Ollama has no model {model!r}. Run `ollama pull {model}`.")


EMBEDDERS = {
    "ollama": "ollama/nomic-embed-text",
    "openai": "openai/text-embedding-3-small",
}


def get_embedder() -> dspy.Embedder:
    """Return an embedder for the same provider. Anthropic has no embeddings API.

    With DSPY_LAB_PROVIDER=anthropic, set DSPY_LAB_EMBED_PROVIDER=ollama (or openai).
    """
    load_dotenv(find_dotenv(usecwd=True))
    provider = os.getenv("DSPY_LAB_EMBED_PROVIDER") or os.getenv("DSPY_LAB_PROVIDER", "ollama")
    if provider not in EMBEDDERS:
        raise LMConfigError(
            f"No embeddings for provider {provider!r}. Set DSPY_LAB_EMBED_PROVIDER to one of: "
            f"{', '.join(EMBEDDERS)}."
        )
    model = os.getenv("DSPY_LAB_EMBED_MODEL", "").strip() or EMBEDDERS[provider]
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise LMConfigError("Embeddings with openai need OPENAI_API_KEY in .env.")
        return dspy.Embedder(model, api_key=key)
    return dspy.Embedder(model, api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))


def token_usage(lm: dspy.LM | None = None) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) for every LM this run built with `get_lm`.

    A judge or reflection LM has its own history, so counting only the main LM undercounts.
    Calls served from DSPy's cache are not counted because no tokens were spent.
    """
    prompt = completion = 0
    for each in LMS if lm is None else [lm]:
        for entry in each.history:
            usage = entry.get("usage") or {}
            prompt += usage.get("prompt_tokens") or 0
            completion += usage.get("completion_tokens") or 0
    return prompt, completion


def print_usage(lm: dspy.LM) -> None:
    prompt, completion = token_usage()
    print(f"\nTokens used this run: {prompt:,} prompt + {completion:,} completion ({lm.model})")
    print("(calls answered from DSPy's cache cost nothing and are not counted)")
