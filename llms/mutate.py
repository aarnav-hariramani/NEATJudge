from langchain_ollama import ChatOllama

def mutate_prompt(original: str, base_url: str, model: str = "llama3.2:3b", temperature: float = 1.0) -> str:
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    instr = f"""
You refine a judging rubric to rate QUERY→TITLE relevance for e-commerce.

Goal: make the rubric MORE SPECIFIC and USEFUL (concise), by adding concrete criteria:
- brand/model compatibility; variant/size/color; explicit constraints/exclusions; device compatibility
- crisp 1..5 **integer** scale definitions (what qualifies for each number)
- tie-breakers for near matches (exact > close > related > tangential > irrelevant)
MUST include: "Return ONLY JSON: {{"rating": N}}" and explicitly say N is an INTEGER in [1,2,3,4,5].
No explanations or markdown. Output only the FULL prompt text.

ORIGINAL:
{original}
""".strip()
    new_prompt = llm.invoke(instr).content.strip()
    return new_prompt or original
