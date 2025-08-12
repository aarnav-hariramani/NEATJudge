from langchain_ollama import ChatOllama

def mutate_prompt(
    original: str,
    base_url: str,
    model: str = "llama3.2:3b",
    temperature: float = 1.0,
) -> str:
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    instr = f"""
You are refining a judging rubric used to rate QUERY→TITLE relevance for e-commerce.

GOAL: Make the rubric MORE SPECIFIC and USEFUL, but keep it compact. You may:
- Add concrete rules (brand/model compatibility, variant/size/color, constraints/exclusions)
- Tighten the 1..5 grading rubric with crisp criteria
- Add tie-breakers (exact match > close match > related > tangential > irrelevant)
- Keep JSON-only output rule: return exactly {{"rating": N}} where N is 1..5 (integer)

Do NOT include any explanations or markdown in your output.
Just output the FULL prompt text that should appear above the few-shot examples.

ORIGINAL:
{original}
""".strip()
    new_prompt = llm.invoke(instr).content.strip()
    return new_prompt if new_prompt else original
