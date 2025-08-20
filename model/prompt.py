# model/prompt.py
# Utility to build the system prompt shown to the LLM judge.

from __future__ import annotations
from typing import List, Dict

# Canonical header/rubric. Keep this SHORT and specific.
DEFAULT_HEADER = """You are a strict evaluator for e-commerce relevance. Given a user QUERY and a candidate product TITLE, rate how well the TITLE satisfies the QUERY on an integer scale 1–5:

5 = Exact/ideal match (brand/model/variant correct; meets all constraints like size/color/device compatibility)
4 = Strong match (minor mismatch or missing secondary attribute; still clearly suitable)
3 = Partial match (related product or wrong variant; may suit some intents but not a clear fit)
2 = Tangential (same broad category but unlikely to satisfy the query)
1 = Irrelevant (different category or violates explicit constraints)

Tie-breakers for near cases: exact > close > related > tangential > irrelevant.
Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].""".strip()


def _format_examples(examples: List[Dict[str, str]]) -> str:
    """Format few-shot examples for the judge."""
    lines = []
    for ex in examples:
        q = ex.get("query", "").strip()
        t = ex.get("title", "").strip()
        y = int(round(float(ex.get("label", 0))))
        # clamp to [1..5]
        y = max(1, min(5, y))
        lines.append(f"QUERY: {q}\nTITLE: {t}\nGOLD: {y}")
    return "\n\n".join(lines)


def _dedupe_return_only_json(text: str) -> str:
    """Ensure there's exactly one 'Return ONLY JSON' instruction at the end."""
    import re
    # remove duplicate lines that start with Return ONLY JSON
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith("return only json")]
    # append the canonical one
    lines.append('Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].')
    # squeeze blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned


def sanitize_prompt(text: str) -> str:
    """Strip meta-instructions that sometimes leak in from the mutation LLM."""
    import re
    bad_markers = [
        "You refine a judging rubric",
        "Goal: make the rubric MORE SPECIFIC",
        "MUST include: \"Return ONLY JSON",
        "ORIGINAL:",
        "Output only the FULL prompt text",
        "No explanations or markdown",
    ]
    t = text.strip()
    for marker in bad_markers:
        idx = t.lower().find(marker.lower())
        if idx != -1 and idx < 100:
            after = t[idx:].splitlines()
            try:
                first_empty = next(i for i, ln in enumerate(after) if ln.strip() == "")
                t = "\n".join(after[first_empty+1:]).strip()
            except StopIteration:
                if "ORIGINAL:" in t:
                    t = t.split("ORIGINAL:")[-1].strip()
            break

    # Strip stray code fences / quotes
    t = t.strip().strip("`").strip()
    t = re.sub(r"^['\"]+|['\"]+$", "", t).strip()

    # Ensure the canonical return instruction appears once, at the end.
    t = _dedupe_return_only_json(t)
    return t


def assemble_prompt(header: str, query: str, title: str, examples: List[Dict[str, str]]) -> str:
    """Build the full message text that we send to the judge model.

    NOTE: train.py imports *this* signature: (header, query, title, examples)
    """
    header = sanitize_prompt(header or DEFAULT_HEADER)
    fewshot = _format_examples(examples or [])
    parts = [header]
    if fewshot:
        parts.append("\nHere are labeled examples:\n\n" + fewshot)
    parts.append(f"\nNow rate this case:\nQUERY: {query}\nTITLE: {title}\nReturn ONLY JSON: {{\"rating\": N}} where N is an INTEGER in [1,2,3,4,5].")
    return "\n".join(parts).strip()
