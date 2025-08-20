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
        Our custom rate lives on that object as 'prompt_mutate_rate'.
        """
        super().mutate(config)
        self._ensure_header()

        # Pull the custom mutation rate directly from the genome config
        rate = getattr(config, "prompt_mutate_rate", 0.0)
        if rate <= 0.0 or _mutate is None:
            return

        if random.random() < rate:
            try:
                old = self.header_text
                new = _mutate(old, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"))

                # Validate mutated header
                if not _is_valid_header(new):
                    # Rejected — keep the old header
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
