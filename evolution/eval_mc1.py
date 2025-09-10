
import os, pickle
import numpy as np
from typing import Any, Dict, List, Optional
from neatjudge.data.truthfulqa_hf import load_truthfulqa_hf
from neatjudge.data.splits import make_splits
from neatjudge.data.bank_index import BankIndex
from neatjudge.selector.features import build_features
from neatjudge.selector.network import Selector
from neatjudge.selector.mmr import mmr_select
from neatjudge.prompts.mutation import PromptGenome, BASE_HEADER
from neatjudge.metrics.truthfulqa import mc1_accuracy
from neatjudge.judge.ollama import Judge

def _get_embed_model(cfg: Dict[str, Any]) -> str:
    return cfg.get("selector", {}).get("embedding", "sentence-transformers/all-MiniLM-L6-v2")

def _maybe_load_genome(path: Optional[str]) -> Optional[object]:
    p = path or "runs/best_selector.pkl"
    if os.path.exists(p):
        with open(p, "rb") as f: return pickle.load(f)
    return None

def _maybe_load_prompt(path: Optional[str]) -> PromptGenome:
    p = path or "runs/best_prompt.json"
    if os.path.exists(p):
        import json
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return PromptGenome(**d)
    return PromptGenome(header=BASE_HEADER)

def eval_truthfulqa_mc1(cfg: Dict[str, Any],
                        genome_path: Optional[str] = None,
                        prompt_path: Optional[str] = None,
                        neat_config_path: Optional[str] = None) -> float:
    rows = load_truthfulqa_hf(cfg)
    splits = make_splits(rows, cfg)
    bank, test = splits["bank"], splits["test"]

    bank_index = BankIndex(bank, _get_embed_model(cfg))
    # Load selector + prompt
    genome = _maybe_load_genome(genome_path)
    selector = Selector(genome, neat_config_path=neat_config_path)
    prompt_g = _maybe_load_prompt(prompt_path)
    judge = Judge(model=cfg["judge"]["model"], base_url=cfg["judge"]["base_url"], temperature=cfg["judge"]["temperature"])

    preds: List[float] = []
    eval_rows: List[Dict[str, Any]] = []

    K = int(cfg["selector"]["K"])
    shortlist = int(cfg["selector"]["shortlist"])
    sim_cap = int(cfg["selector"]["sim_cap"])
    alpha = float(cfg["selector"].get("alpha", 0.5))
    mmr_lambda = float(cfg["selector"]["mmr_lambda"])

    for row in test:
        q = row["question"]
        q_emb = bank_index.encode_query(q)

        examples = []
        if K > 0 and len(bank_index.bank_emb) > 0:
            idxs = bank_index.shortlist(q_emb, shortlist)
            idxs = bank_index.filter_similar(idxs, q_emb, cap=sim_cap)
            cand = np.asarray([bank_index.bank_emb[i] for i in idxs])

            feats = build_features(q_emb, cand)
            sel = selector.score(feats)
            sims = cand @ q_emb
            sims_n = (sims - sims.mean()) / (sims.std() + 1e-9)
            sel_n  = (sel  - sel.mean())  / (sel.std()  + 1e-9) if np.std(sel) > 1e-9 else sel
            comb   = alpha * sims_n + (1.0 - alpha) * sel_n

            order = np.argsort(-comb)
            mmr_local = mmr_select(cand[order], q_emb, K, lam=mmr_lambda)
            chosen_idx = [idxs[int(order[i])] for i in mmr_local]
            examples   = bank_index.examples(chosen_idx)

        prompt = prompt_g.render(q, row["choice"], examples)
        scores = judge.score(prompt, repeats=int(cfg["judge"]["repeats"]))
        finite = [s for s in scores if s==s]
        if finite:
            preds.append(float(sum(finite)/len(finite)))
            eval_rows.append(row)

    mc1 = mc1_accuracy(eval_rows, preds)
    print(f"TruthfulQA MC1 = {mc1:.2f}%")
    return mc1
