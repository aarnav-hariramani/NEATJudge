import neat
import random
import os
import re
from .prompt import DEFAULT_HEADER

# Optional mutator (safe if missing)
try:
    from llms.mutate import mutate_prompt as _mutate
except Exception:
    _mutate = None


def _default_header() -> str:
    """Allow overriding the base header via env var; otherwise use DEFAULT_HEADER."""
    return os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER)


def _is_valid_header(text: str) -> bool:
    """Hard validation so mutated prompts remain usable."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    tl = t.lower()
    # Must include the core concepts & JSON return clause
    must_have = ['return only json', '{"rating":', 'query', 'title']
    if not all(s in tl for s in must_have):
        return False
    # Must explicitly define all 1..5 levels as "N = ..."
    for k in ['1', '2', '3', '4', '5']:
        if not re.search(rf'\b{k}\s*=\s*', t, flags=re.IGNORECASE):
            return False
    return True


class PromptGenome(neat.DefaultGenome):
    """Extends NEAT's DefaultGenome to carry an evolvable prompt header string."""

    @classmethod
    def parse_config(cls, param_dict):
        """
        Called by neat.Config when loading the [DefaultGenome] section.
        We piggyback to add a custom field: prompt_mutate_rate.
        """
        genome_conf = super().parse_config(param_dict)
        pmr = param_dict.get("prompt_mutate_rate", None)
        genome_conf.prompt_mutate_rate = float(pmr) if pmr is not None else 0.0
        return genome_conf

    def _ensure_header(self):
        """Guarantee presence of header_text attribute on this genome."""
        if not hasattr(self, "header_text") or not self.header_text:
            self.header_text = _default_header()

    # --- NEAT lifecycle hooks -------------------------------------------------

    def configure_new(self, genome_config):
        """Initialize a new, random genome."""
        super().configure_new(genome_config)
        self._ensure_header()

    def crossover(self, other, key):
        """
        NOTE: Signature must be (other, key) to match neat.DefaultGenome.
        """
        child = super().crossover(other, key)
        # Ensure all parties have a header; child inherits internal state from parents
        self._ensure_header()
        other._ensure_header()
        # child may not have header_text if parents didn't — enforce it
        if not hasattr(child, "header_text") or not child.header_text:
            # Inherit from one parent at random
            child.header_text = random.choice([self.header_text, other.header_text])
        return child

    def mutate(self, config):
        """
        NEAT calls this with the *genome config* (not the global config).
        We read the custom mutate rate directly off `config`, not `config.genome_config`.
        """
        from llms.mutate import mutate_prompt
        from model.prompt import repair_header

        # If your neat.ini holds this in [DefaultGenome] as 'prompt_mutate_rate'
        rate = float(getattr(config, "prompt_mutate_rate", 1.0))

        # Run the base genome mutation first (weights/topology etc.), if you are using it
        try:
            super().mutate(config)
        except Exception:
            # if you're not using weights/topology, ignore
            pass

        # Mutate the prompt header with some probability
        import random, os
        if random.random() < rate:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            try:
                self.header = mutate_prompt(self.header, base_url=base_url, model=model, temperature=0.7)
            except Exception:
                # If Ollama is unavailable, keep the current header
                pass

            # Always sanitize/repair what we store
            self.header = repair_header(self.header)
