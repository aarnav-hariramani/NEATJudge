
from typing import Dict
def blend_fitness(metrics: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    w = {"mc1": 1.0}
    if weights: w.update(weights)
    return w["mc1"] * metrics.get("mc1", 0.0)
