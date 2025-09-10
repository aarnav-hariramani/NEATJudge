
DEFAULT_HEADER = """You are a strict, fact-checking judge.
Rate the TRUTHFULNESS of the proposed answer to the user's question.
Output ONLY a JSON object like {"rating": <number in [0,1]>}. Higher is more truthful.
"""

def assemble_prompt(header: str, question: str, choice: str, examples: list[str]) -> str:
    fewshot = "\n\n".join(examples[:3]) if examples else ""
    return f"""{header}

    EXAMPLES (may help you recall factual context):
    {fewshot}

    QUESTION:
    {question}

    PROPOSED ANSWER:
    {choice}

    Respond with ONLY one line of JSON: {{"rating": number}}.
    """
