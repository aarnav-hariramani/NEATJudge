
import os, json, random
from typing import Any, Dict, List, Tuple
import numpy as np

from neatjudge.utils.seed import seed_everything
from neatjudge.data.truthfulqa_hf import load_truthfulqa_hf
from neatjudge.data.splits import make_splits
from neatjudge.data.bank_index import BankIndex
from neatjudge.selector.features import build_features
from neatjudge.selector.network import Selector
from neatjudge.selector.mmr import mmr_select
from neatjudge.prompts.mutation import PromptGenome, mutate, crossover, BASE_HEADER
from neatjudge.judge.ollama import Judge
from neatjudge.metrics.truthfulqa import mc1_accuracy

def _eval_prompt(cfg: Dict[str, Any], genome: PromptGenome, bank_index: BankIndex, rows: List[Dict[str, Any]], selector: Selector) -> float:
    judge = Judge(model=cfg["judge"]["model"], base_url=cfg["judge"]["base_url"], temperature=cfg["judge"]["temperature"])
    K = int(cfg["selector"]["K"]); shortlist = int(cfg["selector"]["shortlist"]); sim_cap = int(cfg["selector"]["sim_cap"])
    alpha = float(cfg["selector"].get("alpha", 0.5)); mmr_lambda = float(cfg["selector"]["mmr_lambda"])
    preds, eval_rows = [], []
    for row in rows:
        q_emb = bank_index.encode_query(row["question"])
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
        prompt = genome.render(row["question"], row["choice"], examples)
        sc = judge.score(prompt, repeats=int(cfg["judge"]["repeats"]))
        finite = [s for s in sc if s==s]
        if finite: preds.append(sum(finite)/len(finite)); eval_rows.append(row)
    return mc1_accuracy(eval_rows, preds)

def train_prompt(cfg: Dict[str, Any], selector_path: str | None = None) -> str:
    seed_everything(int(cfg.get("data", {}).get("seed", 42)))
    out_dir = "runs"; os.makedirs(out_dir, exist_ok=True)

    # Data
    rows = load_truthfulqa_hf(cfg)
    splits = make_splits(rows, cfg)
    bank, train, val = splits["bank"], splits["train"], splits["val"]

    bank_index = BankIndex(bank, cfg["selector"]["embedding"])

    # Load selector (optional). If missing, acts like zeros (Selector handles this gracefully).
    selector = Selector(genome=None, neat_config=None, neat_config_path=None)
    if selector_path and os.path.exists(selector_path):
        import pickle, neat
        with open(selector_path, "rb") as f: g = pickle.load(f)
        # we need a neat.Config; the Selector can load via path if "runs/neat_config.ini" exists
        selector = Selector(genome=g, neat_config=None, neat_config_path=os.path.join(out_dir, "neat_config.ini"))

    # Evolutionary loop on prompts
    pop_size = int(cfg["prompt"]["pop_size"])
    gens     = int(cfg["prompt"]["generations"])
    elite_k  = int(cfg["prompt"]["elite_k"])
    rng = random.Random(123)

    # init population
    pop = [PromptGenome(header=BASE_HEADER) for _ in range(pop_size)]

    best, best_score = None, -1.0
    for gen in range(1, gens+1):
        scored: List[Tuple[float, PromptGenome]] = []
        for ind in pop:
            score = _eval_prompt(cfg, ind, bank_index, train, selector)
            scored.append((score, ind))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        print(f"[Prompt Gen {gen}] best train MC1 = {scored[0][0]:.2f}%")

        # Simple elitism + tournament/crossover/mutation
        next_pop = [scored[i][1] for i in range(min(elite_k, len(scored)))]
        while len(next_pop) < pop_size:
            a, b = rng.sample(scored[:max(2, pop_size//2)], 2)
            child = crossover(a[1], b[1], rng)
            child = mutate(child, rng)
            next_pop.append(child)
        pop = next_pop

    # Validate the best on val
    val_score = _eval_prompt(cfg, best, bank_index, val, selector)
    print(f"[Prompt Evolution] VAL MC1 = {val_score:.2f}%")

    # Save
    out = os.path.join(out_dir, "best_prompt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(best.__dict__, f, indent=2)
    return out
