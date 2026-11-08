import pytest

from dspy_prompt_lab.lm import LMConfigError, configure_lm, get_lm


@pytest.fixture(autouse=True)
def no_ant_profile(monkeypatch):
    # Never read the developer's real `ant` login in tests.
    monkeypatch.setattr("dspy_prompt_lab.lm._ant_profile_token", lambda: None)


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


def test_anthropic_falls_back_to_the_ant_cli_profile(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "anthropic")
    monkeypatch.setattr("dspy_prompt_lab.lm._ant_profile_token", lambda: "sk-ant-oat01-fake")
    lm = get_lm()
    assert lm.model == "anthropic/claude-haiku-5-5"
    assert lm.kwargs["api_key"] == "sk-ant-oat01-fake"


def test_explicit_key_wins_over_the_ant_profile(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-explicit")
    monkeypatch.setattr("dspy_prompt_lab.lm._ant_profile_token", lambda: "sk-ant-oat01-fake")
    assert get_lm().kwargs["api_key"] == "sk-explicit"


def test_missing_anthropic_credentials_mention_ant_login(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "anthropic")
    with pytest.raises(LMConfigError, match="ant auth login"):
        get_lm()


def test_ant_profile_token_parses_cli_json_and_survives_a_missing_cli(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from dspy_prompt_lab import lm as lm_module

    monkeypatch.undo()  # drop the autouse stub so the real function runs
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout='{"access_token": "tok"}')
    )
    assert lm_module._ant_profile_token() == "tok"

    def boom(*a, **k):
        raise FileNotFoundError("ant")

    monkeypatch.setattr(subprocess, "run", boom)
    assert lm_module._ant_profile_token() is None


def test_anthropic_sends_no_temperature_because_claude_5_rejects_it(monkeypatch):
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert get_lm().kwargs["temperature"] is None  # None means DSPy does not send it
    monkeypatch.setenv("DSPY_LAB_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_lm().kwargs["temperature"] == 0.0
