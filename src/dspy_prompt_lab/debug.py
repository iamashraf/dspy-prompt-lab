"""Run logs and debugging helpers.

`configure_lm()` calls `start_run_log()`, so every example writes two files under
`artifacts/logs/` (override with DSPY_LAB_LOG_DIR):

  <example>-<time>.log          what DSPy logged: scores, optimizer progress, warnings
  <example>-<time>.trace.jsonl  every LM call: the messages sent, the reply, token counts

Set DSPY_LAB_DEBUG=1 for DEBUG-level logs. The trace keeps an allow-listed set of fields only.
`lm.history` also holds the call kwargs, which include the API key, so never dump it whole.
"""

import atexit
import json
import logging
import os
import sys
import time
from pathlib import Path

import dspy

TRACE_FIELDS = ("timestamp", "model", "messages", "outputs", "usage", "cost")
_started: set[str] = set()
LMS: list[dspy.LM] = []  # every LM built by get_lm: judge, reflection and optimizer LMs included


def start_run_log(lm: dspy.LM, name: str | None = None) -> Path:
    """Attach a file log for this run and write the LM trace when the process exits."""
    script = Path(sys.argv[0]).stem  # "-c" for `python -c`, empty in some REPLs
    name = name or (script if script.isidentifier() or script[:1].isdigit() else "run")
    log_dir = Path(os.getenv("DSPY_LAB_LOG_DIR", "artifacts/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = log_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    level = logging.DEBUG if os.getenv("DSPY_LAB_DEBUG") == "1" else logging.INFO

    handler = logging.FileHandler(f"{stem}.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    for logger_name in ("dspy", "dspy_prompt_lab"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler)

    # One trace per process: atexit also fires when an example crashes, which is when you want it.
    if str(stem) not in _started:
        _started.add(str(stem))
        atexit.register(write_trace, Path(f"{stem}.trace.jsonl"))
    logging.getLogger("dspy_prompt_lab").info("run=%s model=%s", name, lm.model)
    print(f"Logs: {stem}.log  (LM trace: {stem}.trace.jsonl)")
    return Path(f"{stem}.log")


def write_trace(path: Path, lms: list | None = None) -> int:
    """Write one JSON line per LM call, keeping only TRACE_FIELDS. Returns the number of calls."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for lm in LMS if lms is None else lms:
            for entry in lm.history:
                if isinstance(entry, dict):
                    f.write(json.dumps({k: entry.get(k) for k in TRACE_FIELDS}, default=str) + "\n")
                    count += 1
    return count


def show_failures(result, n: int = 5, fields: tuple[str, ...] = ("label", "answer")) -> None:
    """Print the first `n` failed rows of a `dspy.Evaluate` result: input, predicted, gold.

    A score says how much is wrong. The failures say what to change.
    """
    failed = [(ex, pred) for ex, pred, ok in result.results if not ok]
    print(f"{len(failed)} of {len(result.results)} failed. First {min(n, len(failed))}:")
    for ex, pred in failed[:n]:
        text = str(next(iter(ex.inputs().values())))[:70]
        for field in fields:
            if field in ex:
                print(f"  {text!r}\n    predicted={getattr(pred, field, None)!r}  gold={ex[field]!r}")
                break
