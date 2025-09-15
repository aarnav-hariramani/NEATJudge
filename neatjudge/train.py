# neatjudge/train.py
import argparse, neat, numpy as np, random, os, time, pickle, json
from .utils.io import read_yaml
from .utils.logging import tqdm_gen
from .utils.seed import set_global_seed
from .data.loaders import load_dataset_from_cfg
from .data.splits import split_bank_val
from .data.bank_index import BankIndex
from .evolution.selector import build_features, Selector
from .evolution.prompt import DEFAULT_HEADER, assemble_prompt
from .evolution.mutate import mutate_header
from .llms.judge import TransformersJudge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    set_global_seed(cfg.get("data", {}).get("seed", 42))

    all_examples = load_dataset_from_cfg(cfg)               # TruthfulQA (HF)
    bank, val = split_bank_val(
    all_examples,
    val_frac=cfg["data"]["val_frac"],
    seed=cfg["data"]["seed"]
)
    bank_index = BankIndex(bank, embed_model=cfg["selector"]["embed_model"])


    # ==== LLM ====
    judge = TransformersJudge(
        model_name=cfg["judge"]["model_name"],
        device=cfg["judge"].get("device", "auto"),
        max_new_tokens=1,
        temperature=cfg["judge"]["temperature"]
    )

    # ==== NEAT SETUP ====
    neat_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg["neat_config"]  # path to neat.ini
    )
    pop = neat.Population(neat_cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Start with default header; mutate over generations
    HEADER = DEFAULT_HEADER

    def eval_genomes(genomes, neat_config):
        nonlocal HEADER
        # evaluate each genome on a small validation SLICE for speed (fitness = MC1%)
        # NOTE: This is *not* the full-split score; use evaluator script for that.
        # Take a small, deterministic slice so values are discrete (e.g., 8, 16 items)
        slice_size = int(cfg["evolution"].get("val_slice_size", 16))
        rng = np.random.default_rng(cfg["data"]["seed"])
        if len(val) == 0:
            for _, genome in genomes:
                genome.fitness = 0.0
            return
        idxs_val = rng.choice(len(val), size=min(slice_size, len(val)), replace=False)
        val_slice = [val[i] for i in idxs_val]

        # Precompute query embeddings for slice queries
        q_embs = [bank_index.encode_query(ex.question) for ex in val_slice]

        for _, genome in genomes:
            selector = Selector(genome, neat_config)
            mc1_hits = []

            for ex, q_emb in zip(val_slice, q_embs):
                # shortlist by semantic similarity, then optionally re-rank by NEAT selector
                shortlist_idx = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
                idxs = bank_index.filter_similar(shortlist_idx, q_emb, cap=cfg["selector"]["sim_cap"])
                if len(idxs) == 0:
                    mc1_hits.append(0.0)
                    continue

                feats = build_features(q_emb, bank_index.bank_emb[idxs])
                scores = selector.score(feats)
                order = np.argsort(-scores)
                chosen = [idxs[i] for i in order[:cfg["selector"]["K"]]]

                few = [bank[i] for i in chosen]
                prompt = assemble_prompt(HEADER, ex.question, ex.options, few_shots=few)

                pred_idx, _ = judge.score_choices(prompt, ex.options)  # MC1 by argmax logprob
                mc1_hits.append(1.0 if pred_idx == ex.correct_idx else 0.0)

            genome.fitness = 100.0 * float(np.mean(mc1_hits)) if mc1_hits else 0.0

    # Track best-overall across generations, then save a champion bundle for full evaluation
    best_overall = None
    best_overall_fit = -1.0

    for _ in tqdm_gen(range(cfg["evolution"]["generations"]), "generations"):
        pop.run(eval_genomes, 1)

        # update best-overall
        current_best = max(pop.population.values(), key=lambda gn: (gn.fitness or 0.0))
        if (current_best.fitness or 0.0) > best_overall_fit:
            best_overall = current_best
            best_overall_fit = float(current_best.fitness or 0.0)

        # mutate prompt header occasionally
        if random.random() < cfg["evolution"].get("prompt_mutate_rate", 0.4):
            HEADER = mutate_header(HEADER)

    # ===== SAVE CHAMPION ARTIFACTS =====
    run_dir = os.path.join(cfg["runs"]["out_dir"], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # save evolved prompt header
    with open(os.path.join(run_dir, "champion_header.txt"), "w") as f:
        f.write(HEADER)

    # save evolved genome (selector)
    with open(os.path.join(run_dir, "champion_genome.pkl"), "wb") as f:
        pickle.dump(best_overall, f)

    # small manifest (records the small-slice fitness used during training)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "fitness_mc1_val_slice": best_overall_fit,
                "config": cfg,
                "neat_config": cfg["neat_config"],
            },
            f,
            indent=2,
        )

    print(f"[SAVED] Champion → {run_dir}")

if __name__ == "__main__":
    main()
