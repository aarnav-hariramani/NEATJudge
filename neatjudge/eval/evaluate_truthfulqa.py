import argparse, neat, numpy as np
from ..utils.io import read_yaml
from ..data.loaders import load_dataset_from_cfg
from ..data.splits import split_bank_val
from ..data.bank_index import BankIndex
from ..evolution.prompt import DEFAULT_HEADER, assemble_prompt
from ..llms.judge import TransformersJudge
from .metrics import mc1, mc2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = read_yaml(args.config)

    data = load_dataset_from_cfg(cfg)
    bank, val = split_bank_val(data, cfg["data"]["val_frac"], cfg["data"]["seed"])
    bank_index = BankIndex(bank, cfg["selector"]["embed_model"])

    judge = TransformersJudge(cfg["llm"]["hf_model_name"], device=cfg["llm"]["device"])

    preds = []
    mc2_correct = []
    mc2_total = []

    for ex in val:
        q_emb = bank_index.encode_query(ex.question)
        idxs = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
        idxs = bank_index.filter_similar(idxs, q_emb, cap=cfg["selector"]["K"])
        few = [bank[i] for i in idxs]
        header = DEFAULT_HEADER
        few = [bank[i] for i in idxs]; prompt = assemble_prompt(header, ex.question, ex.options, few_shots=few)
        # MC1 via LM logprobs
        pred_idx, probs = judge.score_choices(prompt, ex.options)
        preds.append(pred_idx)
        mc2_correct.append(probs[ex.correct_idx])
        mc2_total.append(sum(probs))

    score_mc1 = mc1(preds, [ex.correct_idx for ex in val])
    score_mc2 = mc2(mc2_correct, mc2_total)
    print(f"TruthfulQA (split={cfg['data']['split']})  MC1={score_mc1:.2f}%  MC2={score_mc2:.2f}%")

if __name__ == '__main__':
    main()
