# llms/mutate.py
# Drop-in replacement: counterexample-driven prompt augmentation.
# - Keeps your base rubric intact.
# - Appends small "Calibration rules" + few-shot examples (from recent mistakes if present).
# - Falls back to generic rules/examples if no mistakes file is available.
#
# This module prints the familiar PROMPT MUTATED block:
# --- PROMPT MUTATED ---
# OLD ...
# NEW ...
# --- END PROMPT MUTATION ---

from __future__ import annotations
import json
import os
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import re
_JSON_LINE_RE = re.compile(r'(?mi)^\s*Return\s+ONLY\s+JSON:\s*\{"rating":\s*N\}.*$')

# Where we *optionally* pull counterexamples from (written by your training loop if available).
_DEFAULT_MISTAKES_PATH = os.path.join("runs", "last_mistakes.jsonl")

# Common color/size tokens used for simple heuristic cues in rules/examples.
_COLOR_WORDS = {
    "black","white","red","blue","green","yellow","purple","pink","silver","gold","gray","grey",
    "navy","beige","brown","teal","orange","rose","space gray","midnight","starlight"
}
_SIZE_PATTERNS = [
    r"\b(6|6\.1|6\.5|7|8|9|10|11|12|13|14)(?:\"| inch|in)\b",
    r"\b(64|128|256|512)\s?GB\b",
    r"\b(1|2|3|4|6|8|12|16)\s?Pack\b",
    r"\b(XXS|XS|S|M|L|XL|XXL|3XL)\b",
]
_DEVICE_HINTS = [
    "iphone","ipad","samsung","galaxy","pixel","oneplus","airpods","switch","playstation","xbox",
    "macbook","surface","kindle","fitbit","garmin"
]

def _read_mistakes(path: str) -> List[Dict[str, Any]]:
    """Read mistakes JSONL if it exists. Each line may include:
       {"query": str, "title": str, "label": int, "pred": int}
       Additional keys are ignored. Returns empty list if file missing/empty.
    """
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "query" in obj and "title" in obj:
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return out

def _extract_brand_candidates(text: str) -> List[str]:
    """Very light heuristic: first capitalized tokens (non-stopwords) as brand candidates."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-]+", text)
    cands: List[str] = []
    for tok in tokens[:6]:  # don’t overdo it
        if tok.isupper():
            # e.g., "USB", skip acronyms that are too generic
            continue
        if tok[0].isupper():
            cands.append(tok.lower())
    return list(dict.fromkeys(cands))  # dedup, preserve order

def _has_any(haystack: str, needles: List[str]) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles if n)

def _find_colors(text: str) -> List[str]:
    t = text.lower()
    found = [c for c in _COLOR_WORDS if c in t]
    return list(dict.fromkeys(found))

def _find_sizes(text: str) -> List[str]:
    res: List[str] = []
    for pat in _SIZE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            res.append(m.group(0))
    return list(dict.fromkeys(res))

def _device_mentions(text: str) -> List[str]:
    t = text.lower()
    out = [d for d in _DEVICE_HINTS if d in t]
    return list(dict.fromkeys(out))

def _categorize_mistake(ex: Dict[str, Any]) -> str:
    """Rough bucket to diversify few-shot selection."""
    q, t = str(ex.get("query","")), str(ex.get("title",""))
    q_brands = _extract_brand_candidates(q)
    t_brands = _extract_brand_candidates(t)

    if q_brands and not _has_any(t, q_brands):
        return "brand_mismatch"
    if _find_sizes(q) and not _has_any(t, _find_sizes(q)):
        return "size_missing"
    if _find_colors(q) and not _has_any(t, _find_colors(q)):
        return "color_missing"
    if _device_mentions(q) and not _has_any(t, _device_mentions(q)):
        return "device_incompat"
    # Explicit negation like "not for" in title vs query for that device
    if "not for" in t.lower() and _device_mentions(q):
        return "violates_constraint"
    return "other"

def _pick_fewshot(mistakes: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Pick up to k mistakes, trying to diversify categories and take the biggest errors."""
    if not mistakes:
        return []
    # Sort by absolute error if present
    def errscore(m: Dict[str, Any]) -> int:
        try:
            return abs(int(m.get("label", 0)) - int(m.get("pred", 0)))
        except Exception:
            return 1
    sorted_m = sorted(mistakes, key=errscore, reverse=True)

    # Bucket by category
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for m in sorted_m:
        c = _categorize_mistake(m)
        buckets.setdefault(c, []).append(m)

    # Round-robin sample across categories
    cats = list(buckets.keys())
    rng.shuffle(cats)
    out: List[Dict[str, Any]] = []
    while len(out) < k and cats:
        for c in list(cats):
            if not buckets[c]:
                cats.remove(c)
                continue
            out.append(buckets[c].pop(0))
            if len(out) >= k:
                break
    if len(out) < k:
        # Top up randomly from remaining
        pool = [m for group in buckets.values() for m in group]
        rng.shuffle(pool)
        out.extend(pool[: max(0, k - len(out))])
    return out

def _build_rules_from_mistakes(mistakes: List[Dict[str, Any]]) -> List[str]:
    """Emit short cap-style rules that actually change decisions deterministically."""
    rules = [
        "- If title violates an explicit constraint in QUERY (e.g., 'not for X'), rate **1**.",
        "- If QUERY specifies a brand/model and TITLE lacks that brand/model, cap rating at **2**.",
        "- If QUERY sets a device family (e.g., iPhone 14 Pro Max) and TITLE is generic/universal with no proof of compatibility, cap at **3**.",
        "- If QUERY includes required size or capacity and TITLE lacks it, cap at **3**.",
        "- Exact brand+model (+variant/color/size) with no contradictions → **5**.",
        "- When borderline: exact > close > related > tangential > irrelevant.",
    ]
    # Light specialization if we see consistent patterns
    if any(_categorize_mistake(m) == "brand_mismatch" for m in mistakes):
        rules.append("- Brand/model mismatch observed frequently; enforce cap **≤2** when brand or model absent from TITLE.")
    if any(_categorize_mistake(m) == "device_incompat" for m in mistakes):
        rules.append("- Device mismatch: if QUERY device not mentioned in TITLE, and TITLE claims 'universal', cap **≤3** unless explicit compatibility is shown.")
    if any(_categorize_mistake(m) == "color_missing" for m in mistakes):
        rules.append("- Missing required color from QUERY → cap **≤4** (do not assign 5).")
    if any(_categorize_mistake(m) == "size_missing" for m in mistakes):
        rules.append("- Missing required size/capacity from QUERY → cap **≤3**.")
    return rules

def _format_fewshot(examples: List[Dict[str, Any]], rng: random.Random) -> str:
    """Format few-shot block with JSON-only answers to steer the judge deterministically."""
    lines = []
    for i, ex in enumerate(examples, 1):
        q = str(ex.get("query", "")).strip()
        t = str(ex.get("title", "")).strip()
        label = ex.get("label", None)
        # If we don't have a gold label, infer a conservative target from heuristics
        if isinstance(label, int) and 1 <= label <= 5:
            target = label
        else:
            # Heuristic fallback
            target = 3
            q_brands = _extract_brand_candidates(q)
            if q_brands and not _has_any(t, q_brands):
                target = 2
            if "not for" in t.lower():
                target = 1
            if q and t and all(x.lower() in t.lower() for x in _extract_brand_candidates(q)[:1]):
                # brand match at least → bump
                target = max(target, 3)
        # Keep the pattern exactly like the judge will see it:
        shot = textwrap.dedent(f"""
        QUERY: {q}
        TITLE: {t}
        Output ONLY JSON: {{"rating": {int(target)}}}
        """).strip()
        lines.append(shot)
    return "\n\n".join(lines)

def _split_at_json_line(text: str):
    m = _JSON_LINE_RE.search(text or "")
    if not m:
        return text.strip(), ''
    return text[:m.start()].rstrip(), text[m.start():].strip()

def mutate_prompt(
    base_prompt: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    *,
    mistakes_path: str = _DEFAULT_MISTAKES_PATH,
    num_fewshot: int = 4,
    seed: Optional[int] = None
) -> str:
    """
    Mutate by inserting a compact calibration block *before* the strict JSON line.
    Do NOT append anything after the JSON line to preserve parsing discipline.
    Signature remains backward-compatible with existing callers.
    """
    pre, json_line = _split_at_json_line(base_prompt)
    if not json_line:
        json_line = 'Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].'

    # Keep it short to limit length penalty — two to three high-precision caps
    rules = [
        "Calibration rules (apply before scoring):",
        "- If QUERY specifies a brand/model that is missing in TITLE, cap rating at 2.",
        "- If TITLE violates an explicit constraint in QUERY (e.g., size, device fit, or “not for X”), rate 1."
    ]
    cal_block = "---\n" + "\n".join(rules) + "\n---"

    # Reassemble with JSON line as the final header instruction
    new_prompt = (pre + "\n\n" + cal_block + "\n\n" + json_line).strip() + "\n"

    # Familiar mutation log format (first 15 lines preview)
    old_head = "\n".join((base_prompt or "").strip().splitlines()[:15])
    new_head = "\n".join(new_prompt.strip().splitlines()[:15])
    print("\n--- PROMPT MUTATED ---")
    print("OLD (first 15 lines):")
    print(old_head)
    print("\nNEW (first 15 lines):")
    print(new_head)
    print("--- END PROMPT MUTATION ---\n")

    return new_prompt