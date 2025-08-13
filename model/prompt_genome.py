import neat, random, os
from .prompt import DEFAULT_HEADER

try:
    from llms.mutate import mutate_prompt as _mutate
except Exception:
    _mutate = None

def _default_header():
    # Allow overriding base header via env if you want
    return os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER)

def _is_valid_header(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.lower()
    # Must enforce JSON-only and 1..5 integer rubric (tighten as needed)
    needs = [
        '{"rating":',          # JSON shape hint
        "only", "json",        # JSON-only instruction
        "1", "2", "3", "4", "5"  # mentions the 1..5 rubric
    ]
    return all(s in t for s in needs)

class PromptGenome(neat.DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        genome_conf = super().parse_config(param_dict)
        pmr = param_dict.get("prompt_mutate_rate", None)
        genome_conf.prompt_mutate_rate = float(pmr) if pmr is not None else 0.0
        return genome_conf

    def _ensure_header(self):
        # Make sure the attribute exists on every path (new, clone, crossover)
        if not hasattr(self, "header_text") or not self.header_text:
            self.header_text = _default_header()

    def configure_new(self, genome_conf):
        super().configure_new(genome_conf)
        self._ensure_header()

    def crossover(self, other, key, cfg):
        child = super().crossover(other, key, cfg)
        # Inherit one parent's header; ensure it exists
        self._ensure_header()
        other._ensure_header()
        child.header_text = random.choice([self.header_text, other.header_text])
        return child

    def mutate(self, genome_conf):
        super().mutate(genome_conf)
        self._ensure_header()

        pmr = getattr(genome_conf, "prompt_mutate_rate", 0.0)
        if pmr <= 0.0 or _mutate is None:
            return

        if random.random() < pmr:
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
                old = self.header_text
                new = _mutate(old, base_url=base_url)

                if not _is_valid_header(new):
                    # Optional: uncomment to see rejects
                    # print("[prompt mutation rejected] invalid header; keeping old.")
                    return

                if new != old:
                    print("\n--- PROMPT MUTATED ---")
                    print("OLD (first 15 lines):")
                    print("\n".join(old.splitlines()[:15]))
                    print("\nNEW (first 15 lines):")
                    print("\n".join(new.splitlines()[:15]))
                    print("--- END PROMPT MUTATION ---\n")
                    self.header_text = new
            except Exception as e:
                print(f"[prompt mutation failed] {e}")
