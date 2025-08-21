import argparse, numpy as np
from utils.io import read_yaml, load_pickle
from data.loaders import load_dataset
from data.splits import make_splits
from data.bank_index import BankIndex
from model.selector import build_features, Selector
from model.prompt import assemble_prompt
from model.metrics import mae_accuracy, extra_classification_metrics
from llms.judge import Judge
from utils.logging import tqdm_gen
from model.mmr import mmr_select
import os

def _get_embed_model(cfg: dict) -> str:
    sel = cfg.get("selector", {}) or {}
    return (
        sel.get("embed_model")
        or sel.get("embedder")
        or os.getenv("EMBED_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    rows = load_dataset(cfg)
    splits = make_splits(rows, cfg)
    test = splits["test"]

    ckpt = load_pickle(args.ckpt)
    genome = ckpt["genome"]
    neat_cfg = ckpt["config"]
    # Support both keys for safety
    header = ckpt.get("header_text") or ckpt.get("header") or ""

    embed_model = _get_embed_model(cfg)
    bank = splits["bank"]
    bank_index = BankIndex(bank, embed_model)

    judge = Judge(
        model=cfg["judge"]["model"],
        base_url=cfg["judge"]["base_url"],
        temperature=cfg["judge"]["temperature"]
    )
    selector = Selector(genome, neat_cfg)

    preds, labs, stabs = [], [], []
    label_range = float(cfg["data"]["label_range"])

    for row in tqdm_gen(test, desc="val g1", leave=False):
        q = row["query"]
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
        prompt = assemble_prompt(header, q, row["response"], examples)
        sc = judge.score(prompt, repeats=cfg["judge"]["repeats"])
        mean_pred, stab = judge.aggregate(sc)
        if not np.isfinite(mean_pred):
            continue
        preds.append(float(mean_pred))
        labs.append(float(row["label"]))
        stabs.append(float(stab))

    mae, acc = mae_accuracy(preds, labs, label_range)
    stab = float(np.mean(stabs)) if stabs else 0.0
    extras = extra_classification_metrics(preds, labs)
    print(f"Test MAE={mae:.3f}  Acc={acc:.2f}%  Stability={stab:.2f}%")
    print(f"Exact={extras['acc_em']:.2f}%  Within±1={extras['acc_within1']:.2f}%  QWK={extras['qwk']:.3f}")

if __name__ == "__main__":
    main()
