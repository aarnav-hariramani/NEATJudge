
import numpy as np
from typing import Any, Dict, List
from neatjudge.data.truthfulqa_hf import load_truthfulqa_hf
from neatjudge.data.splits import make_splits
from neatjudge.data.bank_index import BankIndex
from neatjudge.selector.features import build_features
from neatjudge.selector.network import Selector
from neatjudge.selector.mmr import mmr_select
from neatjudge.prompts.header import assemble_prompt, DEFAULT_HEADER
from neatjudge.metrics.truthfulqa import mc1_accuracy
from neatjudge.judge.ollama import Judge

def _get_embed_model(cfg: Dict[str, Any]) -> str:
    return cfg.get("selector", {}).get("embedding", "sentence-transformers/all-MiniLM-L6-v2")

def eval_truthfulqa_mc1(cfg: Dict[str, Any], genome=None, neat_config_path: str | None = None) -> float:
    rows = load_truthfulqa_hf(cfg)
    splits = make_splits(rows, cfg)
    bank, test = splits["bank"], splits["test"]

    bank_index = BankIndex(bank, _get_embed_model(cfg))
    selector   = Selector(genome, neat_config_path)
    judge      = Judge(model=cfg["judge"]["model"], base_url=cfg["judge"]["base_url"], temperature=cfg["judge"]["temperature"])

    preds: List[float] = []
    eval_rows: List[Dict[str, Any]] = []

    for row in test:
        q = row["question"]
        q_emb = bank_index.encode_query(q)
        shortlist = bank_index.shortlist(q_emb, int(cfg["selector"]["shortlist"]))
        filtered  = bank_index.filter_similar(shortlist, q_emb, cap=int(cfg["selector"]["sim_cap"]))
        cand_emb  = np.asarray([bank_index.bank_emb[i] for i in filtered])

        feats = build_features(q_emb, cand_emb)
        sel   = selector.score(feats)
        sims  = cand_emb @ q_emb
        alpha = float(cfg["selector"].get("alpha", 0.5))
        sims_n = (sims - sims.mean()) / (sims.std() + 1e-9)
        sel_n  = (sel  - sel.mean())  / (sel.std()  + 1e-9) if np.std(sel) > 1e-9 else sel
        comb   = alpha * sims_n + (1.0 - alpha) * sel_n

        order = np.argsort(-comb)
        mmr_local = mmr_select(cand_emb[order], q_emb, int(cfg["selector"]["K"]), lam=float(cfg["selector"]["mmr_lambda"]))
        chosen_idx = [filtered[int(order[i])] for i in mmr_local]
        examples   = bank_index.examples(chosen_idx)

        prompt = assemble_prompt(DEFAULT_HEADER, q, row["choice"], examples)
        scores = judge.score(prompt, repeats=int(cfg["judge"]["repeats"]))
        m = [s for s in scores if s==s]
        if m:
            preds.append(float(sum(m)/len(m)))
            eval_rows.append(row)

    mc1 = mc1_accuracy(eval_rows, preds)
    print(f"TruthfulQA MC1 = {mc1:.2f}%")
    return mc1
