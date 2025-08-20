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
from model.prompt_genome import PromptGenome
from llms.judge import Judge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/default.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    set_global_seed(cfg["data"]["seed"])

    # Load data + splits
    rows = load_dataset(cfg)
    splits = make_splits(rows, cfg)
    bank, pool, val = splits["bank"], splits["fitness_pool"], splits["val"]

    # Bank index + judge
    bank_index = BankIndex(bank, cfg["selector"]["embedder"])
    label_range = float(cfg["data"]["label_range"])
    judge = Judge(
        model=cfg["judge"]["model"],
        base_url=cfg["judge"]["base_url"],
        temperature=cfg["judge"]["temperature"]
    )

    # Configure neat
    neat_cfg = neat.Config(
        PromptGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg["neat_config"],
    )

    # Hyperparams
    generations = int(cfg["evolution"]["generations"])
    pop = neat.Population(neat_cfg)
    pop.add_reporter(neat.StdOutReporter(False))

    # Track best
    best_val = -1e9
    best_ckpt = None
    no_improve = 0

    def eval_genomes(genomes, config):
        nonlocal best_val, best_ckpt, no_improve

        # Ensure we can get a len() for tqdm
        genomes = list(genomes)

        for g_idx, (gid, genome) in enumerate(tqdm_gen(genomes, desc="genomes", leave=True)):
            # Resample per-genome so you see a fresh query bar each time
            micro = resample_microbatches(
                pool,
                batch_size=cfg["evolution"]["batch_size"],
                micro_batches=cfg["evolution"]["micro_batches"],
                seed=np.random.randint(0, 10_000_000),
            )

            # Total queries for this genome across all micro-batches
            total_q = sum(len(b) for b in micro)
            qbar = tqdm_gen(range(total_q), desc=f"queries g{g_idx+1}", leave=False)

            selector = Selector(genome, config)
            header = getattr(genome, "header_text", DEFAULT_HEADER)

            preds, labs, stabs = [], [], []
            divs, length_pens = [], []
            progressed = 0

            for batch in micro:
                for row in batch:
                    q = row["query"]
                    q_emb = bank_index.encode_query(q)
                    shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
                    filtered_idx = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])

                    cand_emb = bank_index.bank_emb[filtered_idx]
                    feats = build_features(q_emb, cand_emb)
                    sel_scores = selector.score(feats)
                    chosen_local = mmr_select(cand_emb, q_emb, cfg["selector"]["K"], lam=cfg["selector"]["mmr_lambda"])
                    chosen_idx = [filtered_idx[i] for i in chosen_local]

                    examples = bank_index.examples(chosen_idx)
                    # >>> include the candidate TITLE being rated <<<
                    prompt = assemble_prompt(header, q, row["response"], examples)

                    if len(prompt) > cfg["selector"]["max_prompt_chars"]:
                        length_pens.append((len(prompt) - cfg["selector"]["max_prompt_chars"]) / 1000.0)

                    scores = judge.score(prompt, repeats=cfg["judge"]["repeats"])
                    mean_pred, stab = judge.aggregate(scores)
                    if np.isfinite(mean_pred):
                        preds.append(float(mean_pred))
                        labs.append(float(row["label"]))
                        stabs.append(float(stab))
                        divs.append(bank_index.diversity(chosen_idx))

                    progressed += 1
                    qbar.update(1)

            qbar.close()

            if not preds:
                genome.fitness = -1e9
                continue

            _, acc = mae_accuracy(preds, labs, label_range)
            stab = float(np.mean(stabs)) if stabs else 0.0
            div  = float(np.mean(divs)) if divs else 0.0
            lpen = float(np.mean(length_pens)) if length_pens else 0.0
            fit  = fitness_score(acc, stab, div, lpen, cfg["fitness"])
            genome.fitness = fit

            # One-line per-genome summary so you *see* it advance.
            print(f"[genome {g_idx+1}/{len(genomes)}] fit={fit:.2f} acc={acc:.2f} stab={stab:.2f} div={div:.2f} len_pen={lpen:.2f}")

        # One generation step
        return

    for g in tqdm_gen(range(generations), desc="generations"):
        winner = pop.run(eval_genomes, 1)
        selector = Selector(winner, neat_cfg)
        header = getattr(winner, "header_text", DEFAULT_HEADER)
        preds, labs = [], []
        val_subset = val[: min(len(val), cfg["evolution"]["batch_size"])]
        for row in tqdm_gen(val_subset, desc=f"val g{g+1}", leave=False):
            q = row["query"]
            q_emb = bank_index.encode_query(q)
            shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
            filtered_idx = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])
            cand_emb = bank_index.bank_emb[filtered_idx]
            feats = build_features(q_emb, cand_emb)
            sel_scores = selector.score(feats)
            chosen_local = mmr_select(cand_emb, q_emb, cfg["selector"]["K"], lam=cfg["selector"]["mmr_lambda"])
            chosen_idx = [filtered_idx[i] for i in chosen_local]
            examples = bank_index.examples(chosen_idx)
            # >>> include the candidate TITLE being rated <<<
            prompt = assemble_prompt(header, q, row["response"], examples)
            sc = judge.score(prompt, repeats=cfg["judge"]["repeats"])
            mean_pred, _ = judge.aggregate(sc)
            if np.isfinite(mean_pred):
                preds.append(float(mean_pred))
                labs.append(float(row["label"]))

        _, val_acc = mae_accuracy(preds, labs, label_range)

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            best_ckpt = {
                "genome": winner,
                "config": neat_cfg,
                "header_text": header,
                "timestamp": time.time(),
                "val_acc": float(val_acc),
            }
            os.makedirs("runs/ckpts", exist_ok=True)
            save_pickle(best_ckpt, "runs/ckpts/best.pkl")

            print("\n=== NEW BEST PROMPT (first 30 lines) ===")
            header_text = header
            print("\n".join(header_text.splitlines()[:30]))
            print("=== END PROMPT ===\n")

            os.makedirs("runs/prompts", exist_ok=True)
            with open(f"runs/prompts/gen{g+1}_val{val_acc:.2f}.txt", "w", encoding="utf-8") as f:
                f.write(header_text)
        else:
            no_improve += 1

        if no_improve >= cfg["evolution"]["early_stop_patience"]:
            print(f"Early stopping at gen {g+1}. Best val_acc={best_val:.2f}")
            break
    print("Training done. Best checkpoint at runs/ckpts/best.pkl")

if __name__ == "__main__":
    main()
