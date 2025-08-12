
from langchain_ollama import ChatOllama
def mutate_prompt(original: str, base_url: str, model: str = "llama3.2:3b", temperature: float = 0.8) -> str:
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    instr = f"""
You are a professional prompt-engineer.
Task: Produce ONE new prompt that
- forces the assistant to reply only with JSON: {{\"rating\": X}}
- includes a concrete rubric for relevance
- keeps placeholders if any
- ends by showing the required output format
- is worded differently (tone, phrasing, ordering)
Do NOT include explanations; output only the prompt text.
----------------
ORIGINAL PROMPT
{original}
----------------
""".strip()
    new_prompt = llm.invoke(instr).content.strip()
    return new_prompt if new_prompt else original
