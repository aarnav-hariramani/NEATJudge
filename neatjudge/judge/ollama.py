
from typing import List, Tuple
import json, math
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatOllama = None  # type: ignore
    SystemMessage = HumanMessage = None  # type: ignore

class Judge:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434/", temperature: float = 0.0):
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature) if ChatOllama else None

    def score(self, prompt: str, repeats: int = 1) -> List[float]:
        scores: List[float] = []
        for _ in range(max(1, repeats)):
            txt = self._invoke(prompt)
            try:
                obj = json.loads(txt.strip())
                rating = float(obj.get("rating", "nan"))
            except Exception:
                rating = float("nan")
            scores.append(rating)
        return scores

    def _invoke(self, prompt: str) -> str:
        if not self.llm:
            return '{"rating": 0.5}'
        msgs = [SystemMessage(content="You output only JSON."), HumanMessage(content=prompt)]
        out = self.llm.invoke(msgs)
        return out.content if hasattr(out, "content") else str(out)

    @staticmethod
    def aggregate(scores: List[float]) -> Tuple[float, float]:
        finite = [s for s in scores if s == s and math.isfinite(s)]
        if not finite:
            return float("nan"), float("nan")
        m = sum(finite) / len(finite)
        v = sum((s - m) ** 2 for s in finite) / len(finite)
        return m, v ** 0.5
