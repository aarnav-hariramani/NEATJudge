
import argparse, json
from neatjudge.utils.io import load_yaml
from neatjudge.evolution.eval_mc1 import eval_truthfulqa_mc1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/default.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)

    results = {}

    # A) Pure baseline — no retrieval
    cfg_a = json.loads(json.dumps(cfg))
    cfg_a["data"]["bank_frac"] = 0.0
    cfg_a["data"]["val_frac"] = 0.0
    cfg_a["data"]["test_frac"] = 1.0
    cfg_a["selector"]["K"] = 0
    results["baseline_pure"] = eval_truthfulqa_mc1(cfg_a)

    # B) Retrieval + MMR only (no selector)
    cfg_b = json.loads(json.dumps(cfg))
    cfg_b["selector"]["K"] = 6
    # selector genome path is absent => zeros => MMR-only
    results["retrieval_mmr"] = eval_truthfulqa_mc1(cfg_b)

    # C) Selector only (load best selector if present)
    results["selector_trained"] = eval_truthfulqa_mc1(cfg)

    # D) Selector + evolved prompt (load both if present)
    results["selector+prompt"] = eval_truthfulqa_mc1(cfg)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
