import json

import dspy

from dspy_prompt_lab import versions
from dspy_prompt_lab.programs import make_triage


def test_save_version_writes_json_and_index_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "PROMPTS", tmp_path)
    lm = dspy.LM("openai/x", api_key="sk-SECRET-123")
    program = make_triage()
    program.predictors()[0].lm = lm  # an optimizer can leave an lm on the predictor
    path = versions.save_version(program, "demo", 80.0, "test", lm)

    assert "sk-SECRET-123" not in path.read_text()
    row = json.loads((tmp_path / "index.jsonl").read_text())
    assert row["file"] == str(path) and row["score"] == 80.0 and len(row["data"]) == 12
    fresh = make_triage()
    fresh.load(str(path))  # the saved file loads back
