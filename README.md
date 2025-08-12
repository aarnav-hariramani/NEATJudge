
# llm-judge

Learned few-shot **retriever + judge** using NEAT for the selector policy and a JSON-only LLM judge.
This repo implements query-conditioned K-shot selection from a **disjoint Bank** and evaluates
on resampled fitness batches, with validation-based model selection to reduce overfitting.
