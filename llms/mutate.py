from langchain_ollama import ChatOllama

def mutate_prompt(original: str, base_url: str, model: str = "llama3.2:3b", temperature: float = 0.7) -> str:
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    instr = f"""
Rewrite the rubric **slightly** to improve clarity and conciseness for rating QUERY→TITLE relevance in e-commerce.
Rules:
- Preserve the task and structure. Do NOT add meta-instructions.
- Keep the meanings of the 1–5 scale the same; tighten wording only.
- Keep this line EXACTLY (unchanged): Return ONLY JSON: {{\"rating\": N}} where N is an INTEGER in [1,2,3,4,5].
- Do not add extra numbered lists beyond the 1–5 rubric.
- Output ONLY the full rubric text (no markdown, no commentary).

Original rubric:
{original}
""".strip()
    new_prompt = llm.invoke(instr).content.strip()
    return new_prompt or original
