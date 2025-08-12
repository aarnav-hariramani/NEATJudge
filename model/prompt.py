
from textwrap import dedent
import json
DEFAULT_HEADER = """
You are a strict evaluator. Given a user query and a proposed response (e.g., product),
rate the **relevance** on a numeric scale compatible with the dataset.
- Output MUST be JSON only: {"rating": X}
- No explanations or extra keys.
- Be consistent and deterministic.
"""
def format_examples(examples):
    lines = []
    for e in examples:
        obj = {"query": e["query"], "response": e["response"], "label": e["label"]}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)
def assemble_prompt(header: str, query: str, examples: list[dict]) -> str:
    ex_json = format_examples(examples)
    return dedent(f"""{header.strip()}

Query:
{query}

Few-shot examples (JSON lines):
{ex_json}

Respond ONLY as: {{"rating": X}}""").strip()
