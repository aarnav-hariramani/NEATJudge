import argparse, numpy as np
from utils.io import read_yaml, load_pickle
from data.loaders import load_dataset
from data.splits import make_splits
from data.bank_index import BankIndex
from model.selector import build_features, Selector
from model.prompt import assemble_prompt
from model.metrics import mae_accuracy, extra_classification_metrics
from llms.judge import Judge
from model.mmr import mmr_select
from utils.logging import tqdm_gen
import os

def _get_embed_model(cfg: dict) -> str:
    sel = cfg.get("selector", {}) or {}
    return sel.get("embed_model", "all-MiniLM-L6-v2")

def main():
    cfg = read_yaml("config/default.yaml")
    _, _, test_rows = make_splits(load_dataset(cfg), cfg)
    bank_index = BankIndex.from_dataset(load_dataset(cfg["data"]), _get_embed_model(cfg))
    best = load_pickle("runs/ckpts/best.pkl")
    selector = Selector(best["genome"])
    header = best["header"]
    judge = Judge(cfg["judge"]["model"], cfg["judge"]["base_url"], cfg["judge"]["temperature"])
    label_range = float(cfg["data"]["label_range"])

    preds, labs, stabs = [], [], []
    for row in tqdm_gen(test_rows, desc="test"):
        q = row["history"]
        q_emb = bank_index.encode_query(q)
        shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
        filtered_idx = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])
        cand_emb = bank_index.bank_emb[filtered_idx]
        feats = build_features(q_emb, cand_emb)
        sel = selector.score(feats)
        sims = cand_emb @ q_emb
        alpha = float(cfg["selector"].get("alpha", 0.5))
        sims_n = (sims - sims.mean()) / (sims.std() + 1e-9)
        sel_n  = (sel  - sel.mean())  / (sel.std()  + 1e-9)
        comb   = alpha * sims_n + (1.0 - alpha) * sel_n
        order  = np.argsort(-comb)
        cand_sorted = cand_emb[order]
        chosen_local = mmr_select(cand_sorted, q_emb, cfg["selector"]["K"], lam=cfg["selector"]["mmr_lambda"])
        chosen_idx = [filtered_idx[int(order[i])] for i in chosen_local]
        examples = bank_index.examples(chosen_idx)

        # >>> Candidate TITLE included <<<
        prompt = assemble_prompt(header, q, row["response"], examples, cfg["selector"]["max_prompt_chars"])
        scores = judge.score(prompt, cfg["judge"]["repeats"])
        if not scores:
            continue
        mean_pred = float(np.mean(scores))
        stab = float(np.std(scores) < 1e-8) * 100.0
        if not np.isfinite(mean_pred):
            continue
        preds.append(mean_pred); labs.append(float(row["label"])); stabs.append(stab)

    mae, acc = mae_accuracy(preds, labs, label_range)
    stab = float(np.mean(stabs)) if stabs else 0.0
    extras = extra_classification_metrics(preds, labs)
    print(f"Test MAE={mae:.3f}  Acc={acc:.2f}%  Stability={stab:.2f}%")
    print(f"Exact={extras['acc_em']:.2f}%  Within±1={extras['acc_within1']:.2f}%  QWK={extras['qwk']:.3f}")

if __name__ == "__main__":
    main()
