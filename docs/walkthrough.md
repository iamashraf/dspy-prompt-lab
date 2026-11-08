# Walkthrough: run the experiments and understand them

This page runs the sixteen examples in a useful order. Each step says what to run, what to look for in the output, what happened in the reference run, and one change to try.

All reference numbers come from one local model, `qwen3-coder:30b`, with the cache off. They are the test score on 36 tickets, where one ticket is 2.8 points. Your numbers will differ. [Results](results.md) explains how much.

Setup is in [getting started](getting-started.md). Terms are in [concepts](concepts.md).

## The task

Every ticket example routes a support ticket for a made-up product, Tasklane, to one of six queues: `billing`, `bug`, `howto`, `account`, `security`, `feature`.

The labels follow a house policy that the model cannot guess. When a ticket mixes topics, the first matching rule wins (`security`, then `billing`, `account`, `bug`, `feature`, `howto`). `howto` means Tasklane can do it, and `feature` means it cannot. The full policy is in [`data/README.md`](../data/README.md). The baseline prompt says none of this, so an optimizer has something to find.

## Part 1. Foundations

### Step 1. See what DSPy sends (`01_signatures_and_modules.py`)

Run it and read the last prompt it prints. You never wrote that prompt. DSPy built it from a signature (`ticket -> label`) and a module (`ChainOfThought`).

- Look for the field names, the allowed labels, and the instruction line "Route a support ticket to the right queue."
- Try changing `ChainOfThought` to `Predict` in `CleanThenTriage` and compare the prompts.

### Step 2. Measure a baseline (`02_evaluate_baseline.py`)

Run it before anything else. Every later result means nothing without this number.

- Look for the baseline score and the list of mistakes.
- Reference run. Dev 72.2%, test 69.4%. 10 of 36 dev tickets failed.
- The mistakes are the policy. The model labeled a refund request about a double-posting Slack bot as `bug`, but billing wins. It labeled a question about inviting a client to one board as `feature`, but guests exist.
- Try printing 10 mistakes instead of 5 and write down the rule you would give a new teammate.

## Part 2. Optimizing the demos

These optimizers change the examples shown in the prompt and leave the instruction alone.

### Step 3. Copy examples in (`09_labeled_fewshot.py`)

`LabeledFewShot` copies 12 training tickets into the prompt. It makes no model calls to compile. Treat it as the floor.

- Reference run. 83.3% from 69.4%, using 35,000 prompt tokens.
- Look for how a free step beats the baseline by 14 points.
- Try `k=4` and `k=24`.

### Step 4. Let the model write its own examples (`03_bootstrap_fewshot.py`)

`BootstrapFewShot` runs the program on training tickets and keeps the runs the metric marks correct. The script compares two settings.

- Reference run. 6 bootstrapped demos gave 80.6%. 16 labeled demos gave 86.1%.
- Look for why. A bootstrapped demo is a run the model already got right, so it teaches what the model knew. The policy lives in the tickets it gets wrong, and only labeled demos carry those.
- Try `max_bootstrapped_demos=6, max_labeled_demos=6` to mix them.

### Step 5. Save and reload (`05_save_load_compiled.py`)

Optimizing costs time and money, so do it once. The script saves a compiled program as JSON, builds a fresh program, loads the file, and scores it.

- Look for the saved file in `artifacts/triage_compiled.json`. It is plain text.
- Look for the line "Same as the saved program: True". Loaded demos come back as plain dicts instead of `dspy.Example` objects, with the same content.
- Reference run. 83.3%.
- Try opening the JSON and deleting a demo, then reload and rescore.

### Step 6. Pick examples per ticket (`11_knn_fewshot.py`)

`KNNFewShot` embeds the training tickets and shows the 6 most similar ones for each new ticket. This needs an embedding model.

- Reference run. 83.3% using 26,000 prompt tokens in 131 seconds.
- Look at the demos printed for one ticket. They should look like it.
- The default setting bootstraps demos for every query. That took 687 seconds for 36 tickets in an earlier run and scored 77.8%. This script asks for labeled neighbors only.

### Step 7. Try many demo sets (`10_bootstrap_random_search.py`)

`BootstrapFewShotWithRandomSearch` builds several candidate programs and keeps the one that scores best on the dev set.

- Reference run. The nine candidates scored 72.2% to 83.3% on dev. The winner scored 77.8% on test.
- Look for the gap between the best dev score and the test score. That gap is selection noise, and it is why you report on a split the optimizer never saw.
- Try `num_candidate_programs=2` to see how few candidates you need.

### Step 8. Vote across programs (`16_ensemble.py`)

`Ensemble` runs several programs and takes a majority vote. The script builds three `LabeledFewShot` programs, each from a different third of the training set.

- Reference run. Members scored 75.0%, 80.6% and 75.0%. The vote scored 80.6%.
- Each ticket costs three model calls.
- Try five members.

## Part 3. Optimizing the instruction

These optimizers rewrite the instruction text. They cost more and show you what they changed.

### Step 9. Search for instructions and demos (`04_miprov2.py`)

`MIPROv2` bootstraps demos, asks a model to propose instructions, and searches the combinations. `auto="light"` keeps the search small.

- Look for the printed instruction and the validation scores of each trial.
- Reference run. The trials scored between 72.2% and 83.3% on dev. The final program scored 77.8% on test, using 208,000 prompt tokens in 833 seconds.
- The instruction it wrote describes five of the six labels in general terms. It did not find the precedence rule.
- Try `auto="medium"` if you have the budget.

### Step 10. Rewrite the instruction (`12_copro.py`)

`COPRO` writes new instructions and keeps the best. It asks for several candidates in one call using the `n` parameter, which Ollama does not support.

- On Ollama the script prints that it tries one new instruction per round.
- Reference run. The best training score was 75.0%, but the test score was 66.7%, below the 69.4% baseline. The new instruction was a long, generic "expert routing specialist" prompt.
- On OpenAI or Anthropic you get the full `breadth`. This repo did not run it there.

### Step 11. Induce rules (`15_infer_rules.py`)

`InferRules` writes natural-language rules from examples and appends them to the instruction.

- Reference run. 80.6%. It wrote five rules, one definition per label, such as "Classify tickets related to payments, invoices, billing cycles, or subscription issues as 'billing'". None covers `security`.
- Look for whether any rule mentions precedence. None did.

### Step 12. Learn from feedback text (`14_gepa.py`)

`GEPA` reads text feedback from your metric and rewrites the instruction from the mistakes. The metric here says which label the model chose, which was correct, and shows the ticket.

- Reference run. 72.2%, up from 69.4%. The best of seven candidates scored 78% on dev, which again overstates test. It used 163,000 prompt tokens.
- The instruction it wrote holds one real piece of the policy, learned from a mistake: a question about SAML SSO with Azure AD is `howto`, not `feature`, because Tasklane supports it. It did not find the precedence rule.
- Try changing the feedback text to include a hint about precedence and see whether the result changes.

### Step 13. Learn from unstable examples (`13_simba.py`)

`SIMBA` samples the program several times per ticket and learns from tickets where the outputs vary.

- Reference run. 69.4%, no change. The final program kept the original instruction and added no demos. It used 131,000 prompt tokens in 1,237 seconds.
- Look in the log for `max-to-min gap`. In 19 of 20 sampling buckets it was 0.0, so the model gave the same answer every time and there was nothing to learn from. SIMBA fits tasks where the model is unstable. This classifier is stable.
- Try a higher temperature in `get_lm(temperature=...)`, or a smaller model that makes more varied mistakes.

## Part 4. Metrics and data

### Step 14. Judge with a model (`06_custom_metric_with_llm_judge.py`)

Exact match fails on free text. The script scores 18 RAG answers with a containment check and with an LLM judge.

- Reference run. Containment 89%, judge 94%. They disagreed on one answer, where the model said "one of the 10 recovery codes" and the key was the singular "recovery code".
- An earlier version of the check used a plain substring test. It marked "The context provided does not contain information about which key adds a new card" as correct, because the key `n` appears inside "information". The judge caught it. The metric now needs a whole word.
- Try judging with a different model by setting `DSPY_LAB_JUDGE_MODEL`.

### Step 15. Optimize a RAG program (`07_rag_pipeline_optimization.py`)

A keyword retriever feeds 3 help-center documents to a `ChainOfThought` answerer. DSPy tunes the answerer only.

- Look at the retrieval line first. Reference run, the right document was in the top 3 for 17 of 18 questions.
- Reference run. 88.9% baseline and 83.3% after optimizing. That is one question, so it is noise. Optimization does not always help.
- Try `k=1` and watch the retrieval line drop.

### Step 16. Catch overfitting (`08_overfitting_and_splits.py`)

The script trains on 6, 18 and 36 tickets and prints the score on the training set and on dev.

| Training tickets | Train | Dev |
|---|---|---|
| 6 | 100.0% | 80.6% |
| 18 | 88.9% | 86.1% |
| 36 | 91.7% | 83.3% |

- Look at the gap. With 6 tickets the prompt contains all of them, so train is perfect and dev is 19 points lower.
- It picks the best size by dev score and reports test once: 83.3%.

## What to take away

- Examples beat instructions here. The best results came from labeled demos (step 4, 86.1%). The instruction optimizers missed the precedence order.
- Free beats expensive. `LabeledFewShot` (step 3) used no compile calls and landed within 3 points of the best.
- Read the instruction and the failures, not only the score. See [debugging](debugging.md). For how far to trust the numbers, see [results](results.md).

Next, see [optimizers](optimizers.md) for when to pick each one.
