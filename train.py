import argparse, neat, numpy as np, random
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
from .eval.metrics import mc1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = read_yaml(args.config)
    set_global_seed(cfg["data"]["seed"])

    # Data
    rows = load_dataset_from_cfg(cfg)
    bank, val = split_bank_val(rows, cfg["data"]["val_frac"], cfg["data"]["seed"])
    bank_index = BankIndex(bank, cfg["selector"]["embed_model"])

    # NEAT config
    neat_cfg = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        cfg["neat_config"]
    )
    pop = neat.Population(neat_cfg)
    pop.add_reporter(neat.StdOutReporter(False))

    judge = TransformersJudge(cfg["llm"]["hf_model_name"], device=cfg["llm"]["device"])

    HEADER = DEFAULT_HEADER

    def eval_genomes(genomes, config):
        # choose a micro-batch of validation questions
        rnd = random.Random(cfg["data"]["seed"])
        sample = rnd.sample(val, min(len(val), cfg["evolution"]["batch_size"]))
        for gid, genome in genomes:
            sel = Selector(genome, config)
            mc1_scores = []
            for ex in sample:
                q_emb = bank_index.encode_query(ex.question)
                idxs = bank_index.shortlist(q_emb, cfg["selector"]["shortlist"])
                idxs = bank_index.filter_similar(idxs, q_emb, cfg["selector"]["K"])
                feats = build_features(q_emb, bank_index.bank_emb[idxs])
                s = sel.score(feats)
                order = list(np.argsort(-s))
                chosen = [idxs[i] for i in order[:cfg["selector"]["K"]]]
                few = [bank[i] for i in chosen]; prompt = assemble_prompt(HEADER, ex.question, ex.options, few_shots=few)
                pred_idx, _ = judge.score_choices(prompt, ex.options)
                mc1_scores.append(1.0 if pred_idx==ex.correct_idx else 0.0)
            genome.fitness = 100.0 * float(np.mean(mc1_scores)) if mc1_scores else 0.0

    for g in tqdm_gen(range(cfg["evolution"]["generations"]), "generations"):
        pop.run(eval_genomes, 1)
        # mutate prompt header occasionally
        if random.random() < 0.4:
            HEADER = mutate_header(HEADER)

if __name__ == "__main__":
    main()
