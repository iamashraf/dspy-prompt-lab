# Further reading

## DSPy

- [DSPy documentation](https://dspy.ai/) and the [source on GitHub](https://github.com/stanfordnlp/dspy)
- [Learning DSPy](https://dspy.ai/learn/) and the [tutorials](https://dspy.ai/tutorials/)
- [Optimizers overview](https://dspy.ai/learn/optimization/optimizers/)
- [Signatures](https://dspy.ai/learn/programming/signatures/), [modules](https://dspy.ai/learn/programming/modules/), [data](https://dspy.ai/learn/evaluation/data/), [metrics](https://dspy.ai/learn/evaluation/metrics/)
- [Saving and loading programs](https://dspy.ai/tutorials/saving/)
- [Retrieval-augmented generation tutorial](https://dspy.ai/tutorials/rag/)
- [Community use cases](https://dspy.ai/community/use-cases/)

## Papers

| Topic | Paper |
|---|---|
| The DSPy framework | [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) |
| The ancestor of DSPy | [Demonstrate-Search-Predict](https://arxiv.org/abs/2212.14024) |
| `MIPROv2` | [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695) |
| `GEPA` | [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) |
| LLM as judge | [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) |

## Optimizer pages

[LabeledFewShot](https://dspy.ai/api/optimizers/LabeledFewShot/), [BootstrapFewShot](https://dspy.ai/api/optimizers/BootstrapFewShot/), [BootstrapFewShotWithRandomSearch](https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch/), [KNNFewShot](https://dspy.ai/api/optimizers/KNNFewShot/), [Ensemble](https://dspy.ai/api/optimizers/Ensemble/), [COPRO](https://dspy.ai/api/optimizers/COPRO/), [InferRules](https://dspy.ai/api/optimizers/InferRules/), [MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/), [SIMBA](https://dspy.ai/api/optimizers/SIMBA/), [GEPA](https://dspy.ai/api/optimizers/GEPA/overview/), [GEPA advanced](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/)

## Tools this repo uses

- [uv](https://docs.astral.sh/uv/) for Python environments
- [Ollama](https://ollama.com) for local models, and the [`llama3.1`](https://ollama.com/library/llama3.1) and [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) models
- [LiteLLM providers](https://docs.litellm.ai/docs/providers) and its [Ollama page](https://docs.litellm.ai/docs/providers/ollama), which DSPy uses to reach models
- [Podman](https://podman.io/) for containers
