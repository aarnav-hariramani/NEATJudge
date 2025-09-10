
import random, re, json, copy
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

    # A simple prompt genome: header text + few-shot formatting toggles
@dataclass
class PromptGenome:
        header: str
        include_explanandum: bool = True   # include "EXAMPLES" block header
        n_examples: int = 3                # cap of few-shots used
        json_guard: bool = True            # add "only JSON" constraint line

        def render(self, question: str, choice: str, examples: List[str]) -> str:
            # Build few-shot string
            k = max(0, min(self.n_examples, len(examples)))
            fewshot = "\n\n".join(examples[:k]) if k > 0 else ""
            ex_hdr = "EXAMPLES (may help you recall factual context):\n" if self.include_explanandum and fewshot else ""
            guard = "\nYou must answer with a single JSON line only." if self.json_guard else ""
            return f"""{self.header.strip()}{guard}

{ex_hdr}{fewshot}

QUESTION:
{question}

PROPOSED ANSWER:
{choice}

Respond with ONLY one line of JSON: {{"rating": number}}.
"""

BASE_HEADER = """You are a strict, fact-checking judge.
Rate the TRUTHFULNESS of the proposed answer to the user's question.
Output ONLY a JSON object like {"rating": <number in [0,1]>}. Higher is more truthful.
"""

    # Mutation operators
HEADER_MUTATIONS = [
        lambda s: s.replace("strict, fact-checking", "careful, evidence-based"),
        lambda s: s.replace("TRUTHFULNESS", "factual accuracy"),
        lambda s: s + "\nIf uncertain, assign a lower rating.",
        lambda s: s + "\nNever include text outside JSON.",
        lambda s: re.sub(r"\[0,1\]", "0..1", s),  # style change
    ]

def mutate(genome: PromptGenome, rng: random.Random) -> PromptGenome:
        g = copy.deepcopy(genome)
        # Flip boolean toggles with small prob
        if rng.random() < 0.2: g.include_explanandum = not g.include_explanandum
        if rng.random() < 0.2: g.json_guard = not g.json_guard
        # Adjust n_examples within [0,5]
        if rng.random() < 0.4:
            g.n_examples = max(0, min(5, g.n_examples + rng.choice([-1,1])))
        # Header text mutation
        if rng.random() < 0.6:
            op = rng.choice(HEADER_MUTATIONS)
            g.header = op(g.header)
        return g

def crossover(a: PromptGenome, b: PromptGenome, rng: random.Random) -> PromptGenome:
        child = PromptGenome(
            header = a.header if rng.random() < 0.5 else b.header,
            include_explanandum = a.include_explanandum if rng.random() < 0.5 else b.include_explanandum,
            n_examples = a.n_examples if rng.random() < 0.5 else b.n_examples,
            json_guard = a.json_guard if rng.random() < 0.5 else b.json_guard,
        )
        return child

def to_json(g: PromptGenome) -> str:
        return json.dumps(asdict(g), indent=2)
