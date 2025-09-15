# neatjudge/eval/evaluate_truthfulqa.py
import argparse, neat, numpy as np, os
from ..utils.io import read_yaml
from ..data.loaders import load_dataset_from_cfg
from ..data.splits import split_bank_val
from ..data.bank_index import BankIndex
from ..evolution.prompt import DEFAULT_HEADER, assemble_prompt
from ..evolution.selector import build_features
from ..llms.judge import TransformersJudge
from .metrics import mc1, mc2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--header_path", default=None, help="Optional path to a champion_header.txt")
    ap.add_argument("--selector_genome", default=None, help="Optional path to a champion_genome.pkl")
    args = ap.parse_args()
    cfg = read_yaml(args.config)

    # dataset & split (this uses the full configured split — not the tiny training slice)
    all_examples = load_dataset_from_cfg(cfg)
    bank, val = split_bank_val(
    all_examples,
    val_frac=cfg["data"]["val_frac"],
    seed=cfg["data"]["seed"]
)
    bank_index = BankIndex(bank, embed_model=cfg["selector"]["embed_model"])


    # header: default or champion override
    header = DEFAULT_HEADER
    if args.header_path and os.path.exists(args.header_path):
        with open(args.header_path, "r") as fh:
            header = fh.read()

    # judge (HF model with option logprobs for MC1/MC2)
    judge = TransformersJudge(
        model_name=cfg["judge"]["model_name"],
        device=cfg["judge"].get("device","auto"),
        max_new_tokens=1,
        temperature=cfg["judge"]["temperature"]
    )

    # optional: champion selector genome for re-ranking the shortlist
    selector = None
    if args.selector_genome and os.path.exists(args.selector_genome):
        import pickle
        from ..evolution.selector import Selector
        neat_cfg = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            cfg["neat_config"]
        )
        with open(args.selector_genome, "rb") as fh:
            genome = pickle.load(fh)
        selector = Selector(genome, neat_cfg)

    preds, mc2_correct, mc2_total = [], [], []

    for ex in val:
        q_emb = bank_index.encode_query(ex.question)
        shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
        idxs = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])

        # if a champion selector is provided, re-rank by NEAT scores
        if selector is not None and len(idxs) > 0:
            feats = build_features(q_emb, bank_index.bank_emb[idxs])
            scores = selector.score(feats)
            order = np.argsort(-scores)
            idxs = [idxs[i] for i in order]

        few = [bank[i] for i in idxs[:cfg["selector"]["K"]]]
        prompt = assemble_prompt(header, ex.question, ex.options, few_shots=few)

        pred_idx, probs = judge.score_choices(prompt, ex.options)
        preds.append(pred_idx)
        mc2_correct.append(probs[ex.correct_idx])
        mc2_total.append(sum(probs))

    score_mc1 = mc1(preds, [ex.correct_idx for ex in val])
    score_mc2 = mc2(mc2_correct, mc2_total)
    print(f"TruthfulQA (split={cfg['data']['split']})  MC1={score_mc1:.2f}%  MC2={score_mc2:.2f}%")

if __name__ == '__main__':
    main()
