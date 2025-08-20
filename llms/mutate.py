# llms/mutate.py
from __future__ import annotations

import os
import re
from typing import Optional

from langchain_ollama import ChatOllama

BEGIN = "<<<BEGIN_PROMPT>>>"
END = "<<<END_PROMPT>>>"

_CANON_JSON_LINE = 'Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].'

# a minimal, strong anchor we want to preserve in every mutation
_ANCHOR_FIRST_LINE = (
    "You are a strict evaluator for e-commerce relevance. Given a user QUERY and a candidate product TITLE, "
    "rate how well the TITLE satisfies the QUERY on an integer scale 1–5:"
)

# Anything in this set will be stripped if the LLM leaks our editing instructions.
_FORBIDDEN_LEAK_MARKERS = (
    "ORIGINAL:",
    "Goal:",
    "MUST include:",
    "tie-breakers",
    "Output ONLY JSON:",
    "No explanations",
    "You refine a judging rubric",
    "Edit goals:",
    "BEGIN_PROMPT",
    "END_PROMPT",
)

def _extract_between_markers(text: str) -> Optional[str]:
    m = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), text, flags=re.S)
    return (m.group(1).strip() if m else None)

def _strip_editorial_leak(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if any(tok in line for tok in _FORBIDDEN_LEAK_MARKERS):
            # drop instruction/meta lines that leaked
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()

def _ensure_anchor_and_json_line(mutated: str) -> str:
    lines = [ln.strip() for ln in mutated.splitlines() if ln.strip()]
    if not lines:
        return mutated

    # Force the first line to the canonical anchor
    if _ANCHOR_FIRST_LINE not in lines[0]:
        lines[0] = _ANCHOR_FIRST_LINE

    text = "\n\n".join(lines).strip()

    # Ensure JSON-only line exists exactly once and is well-formed
    if "Return ONLY JSON:" not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + _CANON_JSON_LINE
    else:
        # Normalize the JSON-only line to our canonical version
        text = re.sub(
            r'Return ONLY JSON:.*',
            _CANON_JSON_LINE,
            text
        )

    # Light lint: collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _fallback_small_edit(original: str) -> str:
    """
    If the LLM fails to produce a clean mutation, make a tiny deterministic tweak
    so NEAT still sees a distinct genome.
    """
    # Example tweak: enforce a slightly more specific definition for 4 and 3.
    text = re.sub(
        r"4\s*=\s*.*",
        "4 = Strong match (minor missing attribute; still clearly suitable).",
        original
    )
    text = re.sub(
        r"3\s*=\s*.*",
        "3 = Partial match (related or wrong variant; may suit some intents but not a clear fit).",
        text
    )
    return _ensure_anchor_and_json_line(text)

def mutate_prompt(
    original: str,
    base_url: Optional[str] = None,
    model: str = "llama3.2:3b",
    temperature: float = 0.7,
) -> str:
    """
    Uses Ollama (via LangChain ChatOllama) to produce a bounded mutation.
    We fence the output, then sanitize and validate it so only a judge prompt survives.
    """
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    instr = f"""
You are editing the following rubric/prompt text. Make a concise improvement for QUERY→TITLE relevance in e-commerce.

Rules:
- Keep it the same *kind* of rubric (strict evaluator, 1..5 integer scale).
- Strengthen concreteness: brand/model compatibility; variant/size/color; explicit constraints/exclusions; device compatibility.
- Keep the JSON-only output instruction present and correct.
- NO meta talk, NO explanations, NO markdown, NO quotes.
- Return ONLY the edited prompt between the markers {BEGIN} and {END}.
- Keep length within ±20% of the original.

Original:
{BEGIN}
{original}
{END}
""".strip()

    try:
        raw = llm.invoke(instr).content.strip()
    except Exception:
        # If Ollama is down or any failure happens, do a deterministic tiny edit
        return _fallback_small_edit(original)

    extracted = _extract_between_markers(raw)
    if not extracted:
        # model ignored markers; best-effort salvage
        candidate = _strip_editorial_leak(raw)
    else:
        candidate = _strip_editorial_leak(extracted)

    # Validate & repair
    candidate = _ensure_anchor_and_json_line(candidate)

    # If the mutation is effectively identical, apply a tiny deterministic edit
    if candidate.strip() == original.strip():
        candidate = _fallback_small_edit(original)

    # Aggressive final safety: never let instruction scaffolding leak
    candidate = candidate.replace(BEGIN, "").replace(END, "").strip()
    return candidate or original
