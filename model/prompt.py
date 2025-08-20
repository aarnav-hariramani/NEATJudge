# model/prompt.py
from __future__ import annotations

import re

_CANON_JSON_LINE = 'Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].'

def default_header() -> str:
    """
    Canonical seed rubric. You can evolve away from this, but we always
    keep the general shape and the JSON-only line.
    """
    return (
        "You are a strict evaluator for e-commerce relevance. Given a user QUERY and a candidate product TITLE, "
        "rate how well the TITLE satisfies the QUERY on an integer scale 1–5:\n\n"
        "5 = Exact/ideal match (brand/model/variant correct; meets all constraints like size/color/device compatibility)\n"
        "4 = Strong match (minor mismatch or missing secondary attribute; still clearly suitable)\n"
        "3 = Partial match (related product or wrong variant; may suit some intents but not a clear fit)\n"
        "2 = Tangential (same broad category but unlikely to satisfy the query)\n"
        "1 = Irrelevant (different category or violates explicit constraints)\n\n"
        "Tie-breakers for near cases: exact > close > related > tangential > irrelevant.\n"
        + _CANON_JSON_LINE
    )

def is_valid_header(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    # must include JSON-only line and integer scale markers 1..5
    if "Return ONLY JSON:" not in text:
        return False
    if not re.search(r"\b1\b.*\b2\b.*\b3\b.*\b4\b.*\b5\b", text, flags=re.S):
        return False
    # forbid obvious editor meta
    forbidden = ("<<<BEGIN_PROMPT>>>", "<<<END_PROMPT>>>", "ORIGINAL:", "Goal:", "MUST include:")
    if any(tok in text for tok in forbidden):
        return False
    return True

def repair_header(text: str) -> str:
    """
    Best-effort to clean a mutated header into a valid rubric.
    """
    if not text:
        return default_header()

    # Remove any stray fences or meta
    text = re.sub(r"<<<?BEGIN_PROMPT>>>?|<<<?END_PROMPT>>>?", "", text)
    for tok in ("ORIGINAL:", "Goal:", "MUST include:", "Edit goals:"):
        text = text.replace(tok, "")

    # Enforce one JSON-only line, normalized
    if "Return ONLY JSON:" not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += _CANON_JSON_LINE
    else:
        text = re.sub(r'Return ONLY JSON:.*', _CANON_JSON_LINE, text)

    # Gentle formatting
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if is_valid_header(text) else default_header()
