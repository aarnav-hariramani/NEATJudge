from textwrap import dedent
import json

DEFAULT_HEADER = """
You are a strict evaluator for e-commerce relevance. Given a user QUERY and a candidate product TITLE, rate how well the TITLE satisfies the QUERY on an integer scale 1–5:

5 = Exact/ideal match (brand/model/variant correct; meets all constraints like size/color/device compatibility)
4 = Strong match (minor mismatch or missing secondary attribute; still clearly suitable)
3 = Partial match (related product or wrong variant; may suit some intents but not a clear fit)
2 = Tangential (same broad category but unlikely to satisfy the query)
1 = Irrelevant (different category or violates explicit constraints)

Tie-breakers for near cases: exact > close > related > tangential > irrelevant.
Return ONLY JSON: {"rating": N} where N is an INTEGER in [1,2,3,4,5].
""".strip()

def format_examples(examples):
    lines = []
    for e in examples:
        obj = {"query": e["query"], "response": e["response"], "label": e["label"]}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)

def assemble_prompt(header: str, query: str, response: str, examples: list[dict]) -> str:
    ex_json = format_examples(examples)
    return dedent(f"""{header}

Query:
{query}

Candidate TITLE to rate:
{response}

Few-shot examples (JSON lines):
{ex_json}

Respond ONLY as: {{"rating": X}}""").strip()
