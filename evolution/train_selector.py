
import os, pickle, random
from typing import Any, Dict, List
import numpy as np

try:
    import neat
except Exception:
    neat = None

from neatjudge.utils.seed import seed_everything
from neatjudge.data.truthfulqa_hf import load_truthfulqa_hf
from neatjudge.data.splits import make_splits
from neatjudge.data.bank_index import BankIndex
from neatjudge.selector.features import build_features
from neatjudge.selector.network import Selector
from neatjudge.selector.mmr import mmr_select
from neatjudge.prompts.mutation import PromptGenome, BASE_HEADER
from neatjudge.judge.ollama import Judge
from neatjudge.metrics.truthfulqa import mc1_accuracy

def _neat_ini(num_inputs: int) -> str:
    return f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 100.0
pop_size              = 30
reset_on_extinction   = False
[DefaultGenome]
num_inputs            = {num_inputs}
num_outputs           = 1
num_hidden            = 0
activation_default    = sigmoid
activation_options    = sigmoid
aggregation_default   = sum
bias_mutate_rate      = 0.5
conn_add_prob         = 0.5
conn_delete_prob      = 0.3
node_add_prob         = 0.1
node_delete_prob      = 0.05
weight_mutate_rate    = 0.8
[DefaultSpeciesSet]
compatibility_threshold = 3.0
[DefaultStagnation]
max_stagnation       = 5
species_elitism      = 1
[DefaultReproduction]
elitism               = 2
survival_threshold    = 0.2
"""

def train_selector(cfg: Dict[str, Any]) -> str:
    if neat is None:
        raise RuntimeError("neat-python is not installed. pip install neat-python")

    out_dir = "runs"; os.makedirs(out_dir, exist_ok=True)
    seed_everything(int(cfg.get("data", {}).get("seed", 42)))

    all_rows = load_truthfulqa_hf(cfg)
    splits = make_splits(all_rows, cfg)
    bank_rows, train_rows, val_rows = splits["bank"], splits["train"], splits["val"]

    bank_index = BankIndex(bank_rows, cfg["selector"]["embedding"])
    D = bank_index.embed_dim(); num_inputs = 4 * D
    neat_path = os.path.join(out_dir, "neat_config.ini")
    with open(neat_path, "w", encoding="utf-8") as f: f.write(_neat_ini(num_inputs))

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_path)
    p = neat.Population(config); p.add_reporter(neat.StdOutReporter(True)); stats = neat.StatisticsReporter(); p.add_reporter(stats)

    judge = Judge(model=cfg["judge"]["model"], base_url=cfg["judge"]["base_url"], temperature=cfg["judge"]["temperature"])
    K = int(cfg["selector"]["K"]); shortlist = int(cfg["selector"]["shortlist"]); sim_cap = int(cfg["selector"]["sim_cap"])
    alpha = float(cfg["selector"].get("alpha", 0.5)); mmr_lambda = float(cfg["selector"]["mmr_lambda"])
    micro_batches = int(cfg["evolution"]["micro_batches"]); batch_size = int(cfg["evolution"]["batch_size"]); patience = int(cfg["evolution"]["early_stop_patience"])

    best_val, best_genome, no_improve = -1.0, None, 0

    def eval_genome(genome, config) -> float:
        selector = Selector(genome, neat_config=config)
        preds, rows = [], []
        rng = random.Random(1234)
        for b in range(micro_batches):
            batch_qs = rng.sample(train_rows, min(batch_size, len(train_rows)))
            for row in batch_qs:
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
                prompt = PromptGenome(header=BASE_HEADER).render(row["question"], row["choice"], examples)
                sc = judge.score(prompt, repeats=int(cfg["judge"]["repeats"]))
                finite = [s for s in sc if s==s]
                if finite: preds.append(sum(finite)/len(finite)); rows.append(row)
        return mc1_accuracy(rows, preds)

    generation = 0
    while True:
        generation += 1
        winner = p.run(eval_genome, n=1)
        # validate
        sel = Selector(winner, neat_config=config)
        preds, rows = [], []
        for row in val_rows:
            q_emb = bank_index.encode_query(row["question"])
            examples = []
            if K > 0 and len(bank_index.bank_emb) > 0:
                idxs = bank_index.shortlist(q_emb, shortlist)
                idxs = bank_index.filter_similar(idxs, q_emb, cap=sim_cap)
                cand = np.asarray([bank_index.bank_emb[i] for i in idxs])
                feats = build_features(q_emb, cand)
                sel_s = sel.score(feats)
                sims = cand @ q_emb
                sims_n = (sims - sims.mean()) / (sims.std() + 1e-9)
                sel_n  = (sel_s  - sel_s.mean())  / (sel_s.std()  + 1e-9) if np.std(sel_s) > 1e-9 else sel_s
                comb   = alpha * sims_n + (1.0 - alpha) * sel_n
                order = np.argsort(-comb)
                mmr_local = mmr_select(cand[order], q_emb, K, lam=mmr_lambda)
                chosen_idx = [idxs[int(order[i])] for i in mmr_local]
                examples   = bank_index.examples(chosen_idx)
            prompt = PromptGenome(header=BASE_HEADER).render(row["question"], row["choice"], examples)
            sc = judge.score(prompt, repeats=int(cfg["judge"]["repeats"]))
            finite = [s for s in sc if s==s]
            if finite: preds.append(sum(finite)/len(finite)); rows.append(row)
        val_mc1 = mc1_accuracy(rows, preds)
        print(f"[Gen {generation}] VAL MC1 = {val_mc1:.2f}%")
        if val_mc1 > best_val:
            best_val, best_genome, no_improve = val_mc1, winner, 0
            with open(os.path.join(out_dir, "best_selector.pkl"), "wb") as f: pickle.dump(best_genome, f)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (patience={patience}). Best VAL MC1 = {best_val:.2f}%")
                break
    return os.path.join(out_dir, "best_selector.pkl")
