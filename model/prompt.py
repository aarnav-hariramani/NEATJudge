# model/prompt.py
from __future__ import annotations
from textwrap import dedent
from typing import Iterable, Dict, Any, Tuple, List, Optional
import json
import re

# Sectioned default header — remains a single string so other code paths are unchanged
DEFAULT_HEADER = dedent("""
### ROLE
You are a strict evaluator for e-commerce relevance. Given a user QUERY and a candidate product TITLE, you will rate how well the TITLE satisfies the QUERY.

### SCALE
5 = Exact/ideal match (brand/model/variant correct; meets explicit constraints like size/color/device compatibility)
4 = Strong match (minor mismatch or missing secondary attribute; still clearly suitable)
3 = Partial match (related product or wrong variant; may suit some intents but not a clear fit)
2 = Tangential (same broad category but unlikely to satisfy the query)
1 = Irrelevant (different category or violates explicit constraints)

### TIEBREAKERS
Exact > close variant > related accessory > tangential > irrelevant.

### OUTPUT
Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5]. No extra text.
""").strip()


def _format_example_dict(d: Dict[str, Any]) -> str:
    """
    Flexible example normalizer. Supports common keys your codebase likely uses:
    - {"query", "title", "label"}  -> makes {"rating": label}
    - {"input", "output"}          -> raw IO
    - {"prompt", "completion"}     -> raw IO
    - {"query", "title", "pred"}   -> uses 'pred' if no label
    Falls back to a simple pretty-print if unknown.
    """
    # Normalize keys (case-insensitive)
    lower = {k.lower(): k for k in d.keys()}
    def get(key): return d.get(lower.get(key, key))

    query = get("query")
    title = get("title")

    # Prefer label; fall back to rating/pred/completion/output
    label = get("label")
    rating = get("rating")
    pred   = get("pred")
    output = get("output") or get("completion")

    # If we have explicit input/output style, just print it
    if get("input") is not None and output is not None:
        return f"### EXAMPLE\nINPUT:\n{get('input')}\nOUTPUT:\n{output}"

    # If we have free-form prompt/completion
    if get("prompt") is not None and output is not None:
        return f"### EXAMPLE\nPROMPT:\n{get('prompt')}\nOUTPUT:\n{output}"

    # Canonical QUERY/TITLE with JSON rating
    lines = ["### EXAMPLE"]
    if query is not None:
        lines.append(f"QUERY: {query}")
    if title is not None:
        lines.append(f"TITLE: {title}")

    # Decide the numeric rating to print
    y = None
    for cand in (label, rating, pred):
        if cand is None:
            continue
        try:
            y = int(cand)
            break
        except Exception:
            pass

    if y is not None:
        lines.append("OUTPUT:")
        lines.append(json.dumps({"rating": int(y)}, ensure_ascii=False))
    elif output is not None:
        lines.append("OUTPUT:")
        # Ensure it's JSON object or raw text
        out = str(output).strip()
        lines.append(out)
    else:
        # Last resort: dump the dict for visibility
        dumped = json.dumps(d, ensure_ascii=False)
        lines.append(f"DATA: {dumped}")

    return "\n".join(lines).strip()


def _format_example_tuple(t: Tuple[Any, Any]) -> str:
    """
    If examples are given as (input, output) tuples.
    """
    x, y = t
    return f"### EXAMPLE\nINPUT:\n{str(x)}\nOUTPUT:\n{str(y)}"


def _format_few_shots(few_shots: Any) -> str:
    """
    Accepts:
      - list[dict], list[tuple], tuple, dict
      - single dict or tuple
    Returns a normalized examples block or empty string.
    """
    if few_shots is None:
        return ""

    block_items: List[str] = []

    # Single dict/tuple
    if isinstance(few_shots, dict):
        block_items.append(_format_example_dict(few_shots))
    elif isinstance(few_shots, tuple) and len(few_shots) == 2:
        block_items.append(_format_example_tuple(few_shots))  # type: ignore[arg-type]
    elif isinstance(few_shots, (list, tuple)):
        for ex in few_shots:
            if isinstance(ex, dict):
                block_items.append(_format_example_dict(ex))
            elif isinstance(ex, tuple) and len(ex) == 2:
                block_items.append(_format_example_tuple(ex))
            else:
                # If it's a bare string, include as-is
                block_items.append(str(ex))
    else:
        # Fallback to string
        block_items.append(str(few_shots))

    if not block_items:
        return ""

    return "\n\n".join(block_items).strip()


def assemble_prompt(
    header: str,
    few_shots: Optional[Any] = None,
    tail: Optional[str] = None,
) -> str:
    """
    Backward-compatible prompt assembler.

    Common existing call patterns this supports:
      - assemble_prompt(header)
      - assemble_prompt(header, few_shots)
      - assemble_prompt(header, few_shots, tail)

    Behavior:
      1) Starts with header (as-is).
      2) If few_shots provided, appends normalized examples.
      3) If tail provided, appends tail verbatim (e.g., final IO instruction).

    Returns a single string; no change to downstream usage.
    """
    parts: List[str] = []
    if header:
        parts.append(header.strip())

    ex_block = _format_few_shots(few_shots)
    if ex_block:
        parts.append(ex_block)

    if tail:
        parts.append(tail.strip())

    prompt = "\n\n".join([p for p in parts if p]).strip()

    # Safety: ensure the OUTPUT JSON rule remains visible if header had it.
    # (Does not add new policy; just preserves if user header had it.)
    if re.search(r'{"\s*rating\s*"\s*:', prompt, flags=re.I) is None:
        # If header didn’t contain it, we don’t force it.
        pass

    return prompt


__all__ = ["DEFAULT_HEADER", "assemble_prompt"]
