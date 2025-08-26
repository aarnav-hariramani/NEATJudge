
import json, re, numpy as np
from typing import List, Tuple
from langchain_ollama import ChatOllama

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeoutError


_num_re = re.compile(r'(-?\d+(\.\d+)?)')
class Judge:
    def __init__(self, model: str, base_url: str, temperature: float = 0.0):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    def _invoke_with_timeout(self, prompt: str, timeout_s: float):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self.llm.invoke, prompt)
            try:
                res = fut.result(timeout=timeout_s)
                return res.content.strip()
            except FTimeoutError:
                return None
            except Exception:
                return None
            
    def score(self, prompt: str, repeats: int, timeout_s: float = 12.0, retries: int = 1) -> list[float]:
        out = []
        for _ in range(repeats):
            raw = None
            for _try in range(retries + 1):
                raw = self._invoke_with_timeout(prompt, timeout_s)
                if raw:
                    break
            if not raw:
                continue
            val = self._parse(raw)
            if val is not None:
                out.append(val)
        return out
    def aggregate(self, scores: List[float]) -> Tuple[float, float]:
        if not scores:
            return float("nan"), 0.0
        arr = np.array(scores, dtype=float)
        vals, counts = np.unique(arr, return_counts=True)
        stability = float(np.max(counts)) / len(arr) * 100.0
        return float(np.mean(arr)), stability
    @staticmethod
    def _parse(raw: str):
    # 1) Prefer strict JSON with a 'rating' key
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "rating" in obj:
                return float(obj["rating"])
        except Exception:
            pass
        # 2) Accept 'rating: N' or 'rating = N' outside JSON
        m = re.search(r'["\']?rating["\']?\s*[:=]\s*(\d+(\.\d+)?)', raw, flags=re.I)
        if m:
            return float(m.group(1))
        # 3) Otherwise refuse to guess
        return None
