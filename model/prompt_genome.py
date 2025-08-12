
import neat, random, os
from .prompt import DEFAULT_HEADER
try:
    from llms.mutate import mutate_prompt as _mutate
except Exception:
    _mutate = None
class PromptGenome(neat.DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        genome_conf = super().parse_config(param_dict)
        pmr = param_dict.get("prompt_mutate_rate", None)
        genome_conf.prompt_mutate_rate = float(pmr) if pmr is not None else 0.0
        return genome_conf
    def configure_new(self, genome_conf):
        super().configure_new(genome_conf)
        self.header_text = os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER)
    def crossover(self, other, key, cfg):
        child = super().crossover(other, key, cfg)
        child.header_text = random.choice([self.header_text, other.header_text])
        return child
    def mutate(self, genome_conf):
        super().mutate(genome_conf)
        if getattr(genome_conf, "prompt_mutate_rate", 0.0) <= 0.0:
            return
        if _mutate is None:
            return
        if random.random() < genome_conf.prompt_mutate_rate:
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
                self.header_text = _mutate(self.header_text, base_url=base_url)
            except Exception:
                pass
