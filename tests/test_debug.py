import json
from types import SimpleNamespace

import dspy

from dspy_prompt_lab.debug import show_failures, start_run_log, write_trace
from dspy_prompt_lab.lm import token_usage


def fake_lm():
    entry = {
        "timestamp": "t", "model": "openai/x", "messages": [{"role": "user", "content": "hi"}],
        "outputs": ["yo"], "usage": {"prompt_tokens": 3, "completion_tokens": 1}, "cost": 0.0,
        "kwargs": {"api_key": "sk-SECRET-123", "temperature": 0},  # real history entries have this
    }
    return SimpleNamespace(history=[entry], model="openai/x")


def test_trace_never_contains_the_api_key(tmp_path):
    path = tmp_path / "t.jsonl"
    assert write_trace(path, [fake_lm()]) == 1
    text = path.read_text()
    assert "sk-SECRET-123" not in text
    assert json.loads(text)["usage"]["prompt_tokens"] == 3


def test_start_run_log_writes_a_log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LAB_LOG_DIR", str(tmp_path))
    log = start_run_log(fake_lm(), name="demo")
    import logging

    logging.getLogger("dspy_prompt_lab").info("hello from test")
    for h in logging.getLogger("dspy").handlers:
        h.flush()
    assert log.parent == tmp_path and "demo-" in log.name
    assert "hello from test" in log.read_text()


def test_show_failures_prints_input_prediction_and_gold(capsys):
    ex = dspy.Example(ticket="refund me", label="billing").with_inputs("ticket")
    result = SimpleNamespace(results=[(ex, dspy.Prediction(label="bug"), False)])
    show_failures(result)
    out = capsys.readouterr().out
    assert "1 of 1 failed" in out and "predicted='bug'" in out and "gold='billing'" in out


def test_token_usage_sums_every_lm_not_just_the_main_one():
    assert token_usage(fake_lm()) == (3, 1)
    from dspy_prompt_lab import debug

    debug.LMS[:] = [fake_lm(), fake_lm()]  # e.g. main LM + judge LM
    try:
        assert token_usage() == (6, 2)
    finally:
        debug.LMS.clear()


def test_log_name_falls_back_when_not_run_from_a_script(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_LAB_LOG_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["-c"])
    assert start_run_log(fake_lm()).name.startswith("run-")
    monkeypatch.setattr("sys.argv", ["examples/04_miprov2.py"])
    assert start_run_log(fake_lm()).name.startswith("04_miprov2-")
