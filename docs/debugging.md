# Debugging

When an example misbehaves, work through this page from the top. Most problems show up in the log or in the failed predictions.

## Where the evidence is

Every run writes two files under `artifacts/logs/`. Change the folder with `DSPY_LAB_LOG_DIR`.

| File | What it holds | Use it to |
|---|---|---|
| `<example>-<time>.log` | DSPy's log lines: scores, optimizer progress, warnings, errors | Find out what the optimizer did |
| `<example>-<time>.trace.jsonl` | One line per model call: messages sent, reply, token counts | See the exact prompt and reply |

Set `DSPY_LAB_DEBUG=1` for DEBUG-level log lines.

The trace keeps only the fields `timestamp`, `model`, `messages`, `outputs`, `usage` and `cost`. It leaves out the call arguments because those hold your API key. The files can still contain your data, so read them before sharing.

Read the trace with:

```bash
tail -n 1 artifacts/logs/<example>-<time>.trace.jsonl | python -m json.tool | head -40
```

## Tools in the code

| Tool | What it does |
|---|---|
| `dspy.inspect_history(n=1)` | Prints the last prompt and reply. Used in examples 01 and 03. |
| `show_failures(result)` | Prints the input, prediction and gold answer for failed rows. Used in example 02. |
| `dspy.Evaluate(..., provide_traceback=True, max_errors=N)` | Shows crashes inside the program. |
| `dspy.Evaluate(..., save_as_json="artifacts/eval.json")` | Keeps every input, prediction and score to compare runs. |
| `compiled.predictors()[0].demos` | The demos an optimizer chose. |
| `compiled.predictors()[0].signature.instructions` | The instruction an optimizer wrote. |
| `DSPY_LAB_CACHE=0` | Turns off the reply cache, so every call is real. |

## Read the failures, not the score

A score says how much is wrong. The failed rows say what to change. In example 02, the mistakes show the labeling policy. The model labeled a refund request about a Slack bug as `bug`, but the rule says billing wins. No optimizer found that rule.

## Symptoms and causes

| Symptom | Likely cause | What to check |
|---|---|---|
| `Bootstrapped 0 full traces` | The metric rejected every run | Test the metric by hand on one gold and prediction pair |
| An optimizer scores the same as the baseline | The best candidate was the original, or there was nothing to learn | Read the trial scores in the log and the printed instruction |
| Score drops after optimizing | Small dev set, or overfitting | Compare train and dev (example 08). One ticket is 2.8 points |
| Results change between runs | Cache, sampling, or a non-zero temperature | Rerun with `DSPY_LAB_CACHE=0` and temperature 0 |
| A run finishes in seconds with 0 tokens | Every reply came from the cache | Set `DSPY_LAB_CACHE=0` |
| SIMBA changes nothing | The model gives the same answer on every sample | Look for `max-to-min gap 0.0` in the log |
| COPRO does not change the instruction on Ollama | Ollama returns one completion per call, so there is one candidate | Use a hosted model |
| `UnsupportedParamsError ... 'n'` | The provider does not support several completions | Same as above. Example 12 sets `litellm.drop_params` |
| `HTTP 404` from Ollama | DSPy 3.4's native engine cannot reach Ollama this way | `lm.py` already uses `engine="litellm"` |
| `partially initialized module 'litellm'` | A race when several threads import it | `lm.py` imports it first. Update to the current code |
| `MIPROv2 requires optional dependency 'optuna'` | `optuna` is missing | `uv sync` installs it |

## Warnings that are safe to ignore

- `requires_permission_to_run is deprecated`. The argument no longer does anything. The examples do not pass it.
- `Input contains fields not in signature ... ['max_depth']` during `MIPROv2`. DSPy's own instruction proposer logs these. The run completes.

## Debugging a metric

An optimizer improves what the metric rewards. Check the metric on its own before blaming the optimizer.

1. Pick five gold and prediction pairs, including some wrong ones.
2. Call the metric on each and check the result by hand.
3. Include the nasty cases. A refusal such as "The context does not contain information" must score as wrong.

The containment metric once passed that refusal, because the key `n` appears inside "information". The fix was a whole-word match, plus a test.

## Debugging an LLM judge

Read ten verdicts next to the reference answers. Run it twice. A judge whose verdicts flip between runs adds noise to every score. When you can, judge with a different model from the one being judged. Set `DSPY_LAB_JUDGE_MODEL`.

## Logging and secrets

Do not turn on LiteLLM's own debug logging for shared logs. It can print request details, and this repo has not checked whether those include credentials. The code avoids writing keys to the run log and the trace, and a test checks that the trace never contains one.
