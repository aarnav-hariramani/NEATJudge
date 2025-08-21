import argparse, os, time
import numpy as np
import neat
from utils.io import read_yaml, save_pickle
from utils.seed import set_global_seed
from utils.logging import tqdm_gen
from data.loaders import load_dataset
from data.splits import make_splits, resample_microbatches
from data.bank_index import BankIndex
from model.mmr import mmr_select
from model.prompt import assemble_prompt, DEFAULT_HEADER
from model.metrics import mae_accuracy
from model.fitness import fitness_score
from model.selector import build_features, Selector
from llms.judge import Judge

def _get_embed_model(cfg: dict) -> str:
    sel = cfg.get("selector", {}) or {}
    return sel.get("embed_model", "all-MiniLM-L6-v2")

def evaluate_genome(genome, bank_index, train_rows, val_rows, cfg, header, judge):
    selector = Selector(genome)
    label_range = float(cfg["data"]["label_range"])
    mb = cfg["evolution"]["micro_batches"]
    bs = cfg["evolution"]["batch_size"]
    total_fit = 0.0
    total_acc = 0.0
    total_div = 0.0
    total_len_pen = 0.0
    total_stab = 0.0
    count = 0

    for _ in range(mb):
        batch = resample_microbatches(train_rows, bs)
        preds, labs, stabs = [], [], []
        for row in tqdm_gen(batch, desc="queries g1", leave=False):
            q = row["history"]
            q_emb = bank_index.encode_query(q)
            shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
            filtered_idx = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])

            # --- Edit #1: USE selector scores to pre-rank before MMR (TRAIN) ---
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

            prompt = assemble_prompt(header, q, row["response"], examples, cfg["selector"]["max_prompt_chars"])
            scores = judge.score(prompt, cfg["judge"]["repeats"])
            if not scores:
                continue
            mean_pred, stab = float(np.mean(scores)), float(np.std(scores) < 1e-8) * 100.0
            preds.append(mean_pred)
            labs.append(float(row["label"]))
            stabs.append(stab)

        mae, acc = mae_accuracy(preds, labs, label_range)
        total_acc += acc
        total_stab += float(np.mean(stabs)) if stabs else 0.0
        # diversity & length penalty computed elsewhere or mocked
        total_div += 0.0
        total_len_pen += 0.0
        count += 1

    avg_acc = total_acc / max(count, 1)
    avg_stab = total_stab / max(count, 1)
    avg_div = total_div / max(count, 1)
    avg_len = total_len_pen / max(count, 1)
    fit = fitness_score(avg_acc, avg_stab, avg_div, avg_len, cfg["fitness"])
    return fit, avg_acc, avg_stab, avg_div, avg_len

def train(cfg):
    set_global_seed(cfg["data"]["seed"])
    ds = load_dataset(cfg)
    train_rows, val_rows, test_rows = make_splits(ds, cfg)
    bank_index = BankIndex.from_dataset(ds, _get_embed_model(cfg))
    judge = Judge(cfg["judge"]["model"], cfg["judge"]["base_url"], cfg["judge"]["temperature"])
    header = DEFAULT_HEADER

    # NEAT setup
    neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            cfg["neat_config"])
    pop = neat.Population(neat_conf)

    best = {"fitness": -1e9, "genome": None, "header": header}

    generations = cfg["evolution"]["generations"]
    for gen in range(generations):
        print(f" ****** Running generation {gen} ****** \n")
        genomes = list(pop.population.items())
        fits = []

        for gid, genome in tqdm_gen(genomes, desc="genomes"):
            fit, acc, stab, div, lpen = evaluate_genome(genome, bank_index, train_rows, val_rows, cfg, header, judge)
            fits.append((gid, fit))
            print(f"\n                                               [genome {len(fits)}/{len(genomes)}] fit={fit:.2f} acc={acc:.2f} stab={stab:.2f} div={div:.2f} len_pen={lpen:.2f}\n")

        # evolve population
        pop.reporters.post_evaluate(neat_conf, pop.population, pop.species, dict(fits))
        pop.population = neat.DefaultReproduction().reproduce(neat_conf, pop.species, pop.config.pop_size, pop.generation)
        pop.species.speciate(neat_conf, pop.population, pop.generation)

        # validation with current best genome
        gid_best, best_fit = max(fits, key=lambda x: x[1])
        genome_best = pop.population.get(gid_best, None)
        if genome_best is None:
            # fall back to previous
            genome_best = best["genome"]

        # VALIDATION pass to compute real metrics and update best checkpoint
        selector = Selector(genome_best) if genome_best is not None else None
        preds, labs, stabs = [], [], []
        for row in tqdm_gen(val_rows, desc="val g1", leave=False):
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

            prompt = assemble_prompt(header, q, row["response"], examples, cfg["selector"]["max_prompt_chars"])
            scores = judge.score(prompt, cfg["judge"]["repeats"])
            if not scores:
                continue
            mean_pred, stab = float(np.mean(scores)), float(np.std(scores) < 1e-8) * 100.0
            preds.append(mean_pred); labs.append(float(row["label"])); stabs.append(stab)

        mae, acc = mae_accuracy(preds, labs, float(cfg["data"]["label_range"]))
        stab = float(np.mean(stabs)) if stabs else 0.0

        # save best
        if acc > best.get("val_acc", -1):
            best.update({
                "val_acc": acc,
                "genome": genome_best,
                "header": header
            })
            save_pickle(best, "runs/ckpts/best.pkl")

    print("Training done. Best checkpoint at runs/ckpts/best.pkl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    os.makedirs("runs/ckpts", exist_ok=True)
    train(cfg)

if __name__ == "__main__":
    main()
