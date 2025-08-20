# llms/mutate.py
# Prompt mutation using a local Ollama model via langchain-ollama.

from __future__ import annotations
import os
import re
from typing import Optional
from langchain_ollama import ChatOllama

_MUTATION_INSTR = """
Rewrite the following e-commerce relevance RUBRIC to make it more SPECIFIC and USEFUL while staying concise.

Requirements (HARD):
- Keep the same task framing (judge QUERY→TITLE relevance).
- Keep a crisp 1..5 INTEGER scale with concrete criteria for each number.
- Explicitly mention brand/model, variant/size/color, explicit constraints/exclusions, and device compatibility as criteria.
- Include the exact line: Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].
- DO NOT include any meta-discussion, examples, markdown, or the word "ORIGINAL".
- Output ONLY the rewritten rubric text (no code fences, no quotes).

RUBRIC:
""".strip()


def _postprocess(text: str) -> str:
    """Strip meta/echo and validate the mutated rubric."""
    if not text:
        return ""

    t = text.strip()
    # drop surrounding code fences / quotes
    t = t.strip().strip("`").strip()
    t = re.sub(r"^['\"]+|['\"]+$", "", t).strip()

    # If the model accidentally echoed our instruction, try to cut to the rubric after a blank line.
    lower = t.lower()
    if "rewrite the following" in lower or "requirements (hard)" in lower or "rubric:" in lower:
        parts = t.split("\n\n")
        t = "\n\n".join(p for p in parts if p.strip())[-1].strip()

    return t


def _looks_like_rubric(text: str) -> bool:
    """Heuristic check to ensure the output is an actual rubric, not the instruction."""
    if not text:
        return False
    bad_markers = [
        "rewrite the following", "requirements (hard)", "output only the rewritten",
        "original:", "no explanations", "you refine a judging rubric",
    ]
    tl = text.lower()
    if any(m in tl for m in bad_markers):
        return False
    need = ["Return ONLY JSON", "INTEGER", "1", "2", "3", "4", "5"]
    return all(n.lower() in tl for n in need) and (("QUERY" in text and "TITLE" in text) or ("query" in tl and "title" in tl))


def mutate_prompt(original: str, base_url: str, model: str = "llama3.2:3B", temperature: float = 0.8) -> str:
    """Return a mutated rubric string. If mutation fails validation, fall back to the original."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    prompt = f"{_MUTATION_INSTR}\n\n{original.strip()}\n"
    out = llm.invoke(prompt)
    candidate = _postprocess(out.content if hasattr(out, "content") else str(out))
    if not _looks_like_rubric(candidate):
        # fallback
        return original
    return candidate
