# GRATH Alignment Notes

- TruthfulQA via Hugging Face (`truthful_qa`) MC tasks are used here to avoid custom metrics drift. The refactor aligns evaluation to leaderboard conventions (MC1/MC2).

- The selector + prompt evolution integrates with a standard NEAT feed-forward net. Keep `config/neat.ini` parameters unchanged per your requirement.

- Empirical findings from GRATH you may wish to adopt in later phases:
  (i) minimize domain gap between training pairs and test domain;
  (ii) increase distributional distance between correct/incorrect answers in pairwise data;
  (iii) 1–2 DPO passes often suffice, watch for overfit in MC2.

See the paper for details.
