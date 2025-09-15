import random, re
from .prompt import DEFAULT_HEADER

_ops = [
    lambda s: re.sub(r"careful truthfulness evaluator", "diligent fact-checker", s, flags=re.I),
    lambda s: re.sub(r"Return only a single capital letter.*", "Output just one capital letter (A/B/C/...).", s),
    lambda s: s.replace("with no explanation","and do not explain your choice"),
]

def mutate_header(header: str) -> str:
    s = header
    if random.random() < 0.5:
        op = random.choice(_ops)
        s = op(s)
    return s
