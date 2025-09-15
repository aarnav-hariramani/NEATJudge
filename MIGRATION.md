# Migration Notes

- Removed custom eBay dataset and loaders.
- Unified evaluation to TruthfulQA HF + MC1/MC2.
- Kept your `config/neat.ini` exactly as provided to respect NEAT sensitivity.
- Swapped evaluator to Transformers-based option scoring (log-probs) for reproducible metrics.
- Preserved selector features and interfaces; split modules by responsibility.
