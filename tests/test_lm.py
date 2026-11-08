import pytest

from dspy_prompt_lab.lm import LMConfigError, configure_lm, get_lm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up
    for var in ("DSPY_LAB_PROVIDER", "DSPY_LAB_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_default_is_ollama_through_litellm():
    lm = get_lm()
    assert lm.model == "ollama_chat/llama3.1:8b"
    assert lm.engine == "litellm"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_missing_key_fails_with_clear_message(monkeypatch, provider):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", provider)
    with pytest.raises(LMConfigError, match="_API_KEY"):
        get_lm()


def test_key_is_read_from_env_and_model_override(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DSPY_LAB_MODEL", "my-model")
    assert get_lm().model == "anthropic/my-model"
    assert get_lm(model="other").model == "anthropic/other"


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "nope")
    with pytest.raises(LMConfigError, match="not supported"):
        get_lm()


def test_unreachable_ollama_says_how_to_fix(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:9")
    with pytest.raises(LMConfigError, match="make ollama"):
        configure_lm()


def test_litellm_is_fully_imported_before_threads_can_race():
    import litellm

    assert hasattr(litellm, "completion")  # the lazy-import race left this attribute missing
