# llms/mutate.py
# Slot-wise, LLM-driven prompt mutation (no rubric bloat).
# - Parses the header into sections (ROLE, SCALE, TIEBREAKERS, OUTPUT, EXTRA).
# - Applies *local* edits: rewrite, prune, swap, or compress one section at a time.
# - Uses the same Ollama model you already configure for the judge (config/default.yaml),
#   but you can override via env: LLM_MUTATOR_MODEL, LLM_BASE_URL, LLM_MUTATOR_TEMPERATURE.

from __future__ import annotations
import os, re, random, yaml
from typing import Dict

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

# ---------- config helpers ----------

def _load_cfg() -> dict:
    cfg_path = os.getenv("LLM_JUDGE_CONFIG", "config/default.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}

def _get_llm():
    cfg = _load_cfg()
    judge = cfg.get("judge", {}) if isinstance(cfg, dict) else {}
    model = os.getenv("LLM_MUTATOR_MODEL") or judge.get("model") or "llama3.2:3b"
    base  = os.getenv("LLM_BASE_URL") or judge.get("base_url") or "http://localhost:11434/"
    temp  = float(os.getenv("LLM_MUTATOR_TEMPERATURE") or 0.0)
    if ChatOllama is None:
        raise RuntimeError("ChatOllama not available; install langchain_ollama.")
    return ChatOllama(model=model, base_url=base, temperature=temp)

# ---------- section parsing ----------

_SECTION_ORDER = ["ROLE", "SCALE", "TIEBREAKERS", "OUTPUT", "EXTRA"]
_SECTION_RE = re.compile(r'^\s*#{2,3}\s*(ROLE|SCALE|TIEBREAKERS|OUTPUT|EXTRA)\s*$', re.I | re.M)

def _ensure_markers(header: str) -> str:
    if _SECTION_RE.search(header):
        return header

    # Heuristically split your current rubric into sections
    lines = header.strip().splitlines()
    role, scale, tie, out, extra = [], [], [], [], []
    for ln in lines:
        low = ln.lower()
        if re.match(r'\s*[1-5]\s*=', low) or ("5 =" in low and "1 =" in low) or ("rating" in low and any(x in low for x in ["1", "2", "3", "4", "5"])):
            scale.append(ln)
        elif "tie-break" in low or "tiebreaker" in low or "near cases" in low:
            tie.append(ln)
        elif "return only json" in low or '{"rating"' in low:
            out.append(ln)
        else:
            role.append(ln)

    def _block(tag, content, fallback=""):
        txt = "\n".join(content).strip() or fallback
        return f"### {tag}\n{txt}".strip()

    parts = [
        _block("ROLE", role, "You are an impartial, terse relevance judge for product titles."),
        _block("SCALE", scale, "5 = Exact match\n4 = Strong match\n3 = Partial match\n2 = Tangential\n1 = Irrelevant"),
        _block("TIEBREAKERS", tie, "Exact > close variant > related accessory > tangential > irrelevant."),
        _block("OUTPUT", out, 'Return ONLY JSON: {"rating": N} where N is an integer 1..5. No extra text.')
    ]
    return ("\n\n".join(parts)).strip() + "\n"

def _split(header: str) -> Dict[str, str]:
    header = _ensure_markers(header)
    pieces: Dict[str, str] = {k: "" for k in _SECTION_ORDER}
    current = None
    for ln in header.splitlines():
        m = re.match(r'^\s*#{2,3}\s*(\w+)\s*$', ln.strip(), re.I)
        if m:
            key = m.group(1).upper()
            current = key if key in pieces else None
            continue
        if current:
            pieces[current] += ln + "\n"
    for k in pieces:
        pieces[k] = pieces[k].rstrip()
    return pieces

def _render(pieces: Dict[str, str]) -> str:
    blocks = []
    for k in _SECTION_ORDER:
        v = (pieces.get(k) or "").strip()
        if v:
            blocks.append(f"### {k}\n{v}")
    return ("\n\n".join(blocks)).strip() + "\n"

# ---------- LLM helpers ----------

_REWRITE_SYSTEM = "You are editing one section of a grading rubric. Preserve meaning, reduce redundancy, be concise."

def _llm_rewrite(text: str, style_hint: str) -> str:
    llm = _get_llm()
    user = (
        "Rewrite the following section. Keep the same intent but make it sharper and shorter.\n"
        "Constraints:\n"
        "- Do not add new policy.\n"
        "- Keep total length to ~3–7 short lines.\n"
        "- Avoid synonyms that change semantics.\n"
        "- Output ONLY the rewritten section (plain text, no JSON/prefix).\n\n"
        f"STYLE: {style_hint}\n\nSECTION:\n{text}"
    )
    out = llm.invoke(user).content.strip()
    # Remove stray markdown headers if any
    out = re.sub(r'^\s*#{1,6}\s*', '', out).strip()
    return out

def _llm_summarize_bullets(text: str, max_bullets: int = 6) -> str:
    llm = _get_llm()
    user = (
        f"Compress the bullet list below into at most {max_bullets} bullets. "
        "Keep the most predictive/decisive rules, merge duplicates, remove vague items. "
        "Output ONLY the bullet list (one bullet per line, starting with '-' or a digit). No extra text.\n\n"
        f"{text}"
    )
    out = llm.invoke(user).content.strip()
    return out

# ---------- mutation ops ----------

def _op_prune_extra(p: Dict[str, str]) -> None:
    if not p.get("EXTRA"):
        return
    lines = [ln for ln in p["EXTRA"].splitlines() if ln.strip()]
    if len(lines) > 10:
        p["EXTRA"] = "\n".join(lines[:10])

def _op_rewrite_scale(p: Dict[str, str]) -> None:
    if p.get("SCALE"):
        p["SCALE"] = _llm_rewrite(p["SCALE"], "Make each level more discriminative; 3 vs 4 should be clearly distinct.")

def _op_rewrite_role(p: Dict[str, str]) -> None:
    if p.get("ROLE"):
        p["ROLE"] = _llm_rewrite(p["ROLE"], "Direct, formal, non-chatty. No moralizing.")

def _op_rewrite_ties(p: Dict[str, str]) -> None:
    if p.get("TIEBREAKERS"):
        p["TIEBREAKERS"] = _llm_rewrite(p["TIEBREAKERS"], "One sentence priority ordering; no examples.")

def _op_harden_output(p: Dict[str, str]) -> None:
    base = 'Return ONLY JSON: {"rating": N} where N is an integer 1..5. No extra text.'
    cur  = p.get("OUTPUT", "")
    p["OUTPUT"] = _llm_rewrite(cur or base, "Be strict. Forbid any text besides the JSON object.")

def _op_summarize_extra(p: Dict[str, str]) -> None:
    extra = p.get("EXTRA", "")
    if not extra:
        return
    bullet_lines = [ln for ln in extra.splitlines() if re.match(r'\s*(-|\d+[\.\)])', ln)]
    if len(bullet_lines) >= 6:
        p["EXTRA"] = _llm_summarize_bullets("\n".join(bullet_lines), max_bullets=6)

def _op_delete_random_rule(p: Dict[str, str]) -> None:
    # Delete one bullet from SCALE or EXTRA to combat verbosity drift
    target = "EXTRA" if (p.get("EXTRA") and "-" in p["EXTRA"]) else "SCALE"
    text = p.get(target, "")
    lines = text.splitlines()
    idxs = [i for i, ln in enumerate(lines) if re.match(r'\s*(-|\d+[\.\)])', ln)]
    if idxs:
        i = random.choice(idxs)
        del lines[i]
        p[target] = "\n".join(lines).strip()

_MUT_OPS = [
    _op_prune_extra,
    _op_rewrite_scale,
    _op_rewrite_role,
    _op_rewrite_ties,
    _op_harden_output,
    _op_summarize_extra,
    _op_delete_random_rule,
]

# ---------- public API ----------

def mutate_prompt(base_prompt: str) -> str:
    pieces = _split(base_prompt)
    before = _render(pieces)

    # pick 1–2 local ops
    ops = random.sample(_MUT_OPS, k=min(2, len(_MUT_OPS)))
    for op in ops:
        try:
            op(pieces)
        except Exception:
            pass

    after = _render(pieces)

    # guardrails
    if "rating" not in after.lower():      # must preserve JSON instruction
        return base_prompt
    if not pieces.get("SCALE") or not pieces.get("OUTPUT"):
        return base_prompt
    if len(after) > int(len(before) * 1.25):  # do not bloat
        return base_prompt

    print("\n--- PROMPT MUTATED ---")
    print("OLD (first 15 lines):")
    print("\n".join(before.strip().splitlines()[:15]))
    print("\nNEW (first 15 lines):")
    print("\n".join(after.strip().splitlines()[:15]))
    print("--- END PROMPT MUTATION ---\n")

    return after
