
# NEATJudge (TruthfulQA MC1, HF-only)
- HF loader: `EleutherAI/truthful_qa_mc` (multiple_choice)
- MC1 metric computed from judge scores (group-by-question argmax)
- Selection: embeddings + optional NEAT + MMR
- Judge: Ollama via LangChain
- No eBay code

## Run
pip install -r requirements.txt
bash scripts/eval_truthfulqa.sh
