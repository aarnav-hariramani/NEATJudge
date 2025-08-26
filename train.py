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

from llms.mutate import mix_prompts
import random

HEADER_BANK = []               # list of (fitness, header_text)
BANK_SIZE = 20
BANK_INJECT_P = 0.05

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
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    set_global_seed(cfg["data"]["seed"])

    # Load data + splits
    rows = load_dataset(cfg)
    splits = make_splits(rows, cfg)
    bank, pool, val = splits["bank"], splits["fitness_pool"], splits["val"]

    # Bank index + judge
    embed_model = _get_embed_model(cfg)
    bank_index = BankIndex(bank, embed_model)
    label_range = float(cfg["data"]["label_range"])

    judge = Judge(
        model=cfg["judge"]["model"],
        base_url=cfg["judge"]["base_url"],
        temperature=cfg["judge"]["temperature"]
    )

    # Configure NEAT
    neat_cfg = neat.Config(
        PromptGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg["neat_config"],
    )

    generations = int(cfg["evolution"]["generations"])
    pop = neat.Population(neat_cfg)
    pop.add_reporter(neat.StdOutReporter(False))

    best_val = -1e9
    best_ckpt = None
    no_improve = 0

    def eval_genomes(genomes, config):
        nonlocal best_val, best_ckpt, no_improve
        global HEADER_BANK
        genomes = list(genomes)

        gen_records = []  # collect (fitness, header_text) for this generation

        for g_idx, (gid, genome) in enumerate(tqdm_gen(genomes, desc="genomes", leave=True)):
            # --- Step 2A: probabilistic recombination with bank BEFORE scoring ---
            if HEADER_BANK and random.random() < BANK_INJECT_P:
                # pick a donor header from the bank
                _, donor_header = random.choice(HEADER_BANK)
                # mix current genome's header with donor (ROLE/TIEBREAKERS mix; SCALE/OUTPUT preserved)
                mixed = mix_prompts(getattr(genome, "header_text", DEFAULT_HEADER), donor_header)
                genome.header_text = mixed

            micro = resample_microbatches(
                pool,
                batch_size=cfg["evolution"]["batch_size"],
                micro_batches=cfg["evolution"]["micro_batches"],
                seed=np.random.randint(0, 10_000_000),
            )

            total_q = sum(len(b) for b in micro)
            qbar = tqdm_gen(range(total_q), desc=f"queries g{g_idx+1}", leave=False)

            selector = Selector(genome, config)
            header = getattr(genome, "header_text", DEFAULT_HEADER)
            BASE_LEN = len(DEFAULT_HEADER)
            _header_overhead_chars = max(0, len(header) - BASE_LEN)
            print(f"[header len={len(header)} overhead={_header_overhead_chars}]")

            preds, labs, stabs, divs, length_pens = [], [], [], [], []
            progressed = 0

            for batch in micro:
                for row in batch:
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

                    order = np.argsort(-comb)
                    cand_sorted = cand_emb[order]
                    chosen_local = mmr_select(cand_sorted, q_emb, cfg["selector"]["K"], lam=cfg["selector"]["mmr_lambda"])
                    chosen_idx = [filtered_idx[int(order[i])] for i in chosen_local]
                    examples = bank_index.examples(chosen_idx)

                    max_chars = int(cfg["selector"].get("max_prompt_chars", 6000))
                    ex = list(examples)
                    while True:
                        prompt = assemble_prompt(header, q, row["response"], ex)
                        if len(prompt) <= max_chars or len(ex) == 0:
                            break
                        ex = ex[:-1]  # drop one example and re-check

                    length_pens.append(_header_overhead_chars / 200.0)

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
                # still record header (with very low fitness) so we can keep bank non-empty
                gen_records.append((-1e9, getattr(genome, "header_text", DEFAULT_HEADER)))
                continue

            _, acc = mae_accuracy(preds, labs, label_range)
            stab = float(np.mean(stabs)) if stabs else 0.0
            div  = float(np.mean(divs)) if divs else 0.0
            lpen = float(np.mean(length_pens)) if length_pens else 0.0
            fit  = fitness_score(acc, stab, div, lpen, cfg["fitness"])
            genome.fitness = fit

            print(f"[genome {g_idx+1}/{len(genomes)}] fit={fit:.2f} acc={acc:.2f} stab={stab:.2f} div={div:.2f} len_pen={lpen:.2f}")

            # --- collect for the bank update ---
            gen_records.append((genome.fitness, getattr(genome, "header_text", DEFAULT_HEADER)))

        # --- Step 2B: update HEADER_BANK with this generation’s best headers ---
        merged = {}
        for f, h in HEADER_BANK + gen_records:
            key = (h or "").strip()
            if not key:
                continue
            if key not in merged or f > merged[key]:
                merged[key] = f
        HEADER_BANK[:] = sorted(((f, h) for h, f in merged.items()), key=lambda x: x[0], reverse=True)[:BANK_SIZE]

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
            sel = selector.score(feats)
            sims = cand_emb @ q_emb
            alpha = float(cfg["selector"].get("alpha", 0.5))
            sims_n = (sims - sims.mean()) / (sims.std() + 1e-9)
            sel_n  = (sel  - sel.mean())  / (sel.std()  + 1e-9)
            comb   = alpha * sims_n + (1.0 - alpha) * sel_n
            order  = np.argsort(-comb)
            cand_sorted = cand_emb[order]
            chosen_local = mmr_select(cand_sorted, q_emb, cfg["selector"]["K"], lam=cfg["selector"]["mmr_lambda"])
            chosen_idx = [filtered_idx[i] for i in chosen_local]
            examples = bank_index.examples(chosen_idx)

            # >>> Candidate TITLE included <<<
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
                "header": header,        # keep original key for compatibility
                "header_text": header,   # also store under new name
                "timestamp": time.time(),
                "val_acc": float(val_acc),
            }
            os.makedirs("runs/ckpts", exist_ok=True)
            save_pickle(best_ckpt, "runs/ckpts/best.pkl")

            print("\n=== NEW BEST PROMPT (first 30 lines) ===")
            print("\n".join(header.splitlines()[:30]))
            print("=== END PROMPT ===\n")
            os.makedirs("runs/prompts", exist_ok=True)
            with open(f"runs/prompts/gen{g+1}_val{val_acc:.2f}.txt", "w", encoding="utf-8") as f:
                f.write(header)
        else:
            no_improve += 1

        if no_improve >= cfg["evolution"]["early_stop_patience"]:
            print(f"Early stopping at gen {g+1}. Best val_acc={best_val:.2f}")
            break

    print("Training done. Best checkpoint at runs/ckpts/best.pkl")

if __name__ == "__main__":
    main()
