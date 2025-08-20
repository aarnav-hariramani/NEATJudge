# model/prompt_genome.py
from __future__ import annotations
import neat
import random
import os
from typing import Any, Dict, Iterable, List, Tuple

from .prompt import DEFAULT_HEADER
try:
    from llms.mutate import mutate_prompt as _mutate
except Exception:
    _mutate = None  # mutation optional


class PromptGenomeConfig:
    """Custom config for PromptGenome to let us read 'prompt_mutate_rate' from neat.ini."""
    def __init__(self, config_parser):
        section_name = self.__class__.__name__.replace("Config", "")
        if not config_parser.has_section(section_name):
            section_name = "DefaultGenome"

        get = config_parser.get
        getfloat = config_parser.getfloat

        # how often to rewrite the rubric with the LLM (0..1)
        try:
            self.prompt_mutate_rate = getfloat(section_name, "prompt_mutate_rate")
        except Exception:
            self.prompt_mutate_rate = 0.5

        # Base header to start from (env can override)
        self.base_header = os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER)

        # LLM settings
        self.mutate_model = get(section_name, "mutate_model", fallback=os.getenv("MUTATE_MODEL", "llama3.2:3b"))
        self.mutate_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            self.mutate_temperature = getfloat(section_name, "mutate_temperature")
        except Exception:
            self.mutate_temperature = 0.8

    @staticmethod
    def parse_config(config_parser):
        return PromptGenomeConfig(config_parser)


class PromptGenome(neat.genome.DefaultGenome):
    """
    Minimal genome that only evolves a prompt header string alongside a tiny NN (kept for NEAT bookkeeping).
    NEAT still expects network genes; we inherit DefaultGenome so compatibility stats still work,
    but we add `header_text` and mutate it separately.
    """

    def __init__(self, key):
        super().__init__(key)
        self.header_text: str = os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER)

    # --- Serialization (so checkpoints include the header) ---
    def __getstate__(self):
        state = super().__getstate__()
        state["header_text"] = self.header_text
        return state

    def __setstate__(self, state):
        self.header_text = state.pop("header_text", os.getenv("LLM_JUDGE_HEADER", DEFAULT_HEADER))
        super().__setstate__(state)

    # --- NEAT hooks ---
    def mutate(self, config: PromptGenomeConfig) -> None:
        """Run standard NEAT weight/node mutations AND (probabilistically) rewrite the header with an LLM."""
        # First do the normal genome mutation so NEAT plumbing stays valid
        super().mutate(config)

        # Then optionally mutate the prompt header
        rate = getattr(config, "prompt_mutate_rate", 0.0)
        if rate <= 0 or _mutate is None:
            return
        if random.random() > rate:
            return

        try:
            new_text = _mutate(
                self.header_text,
                base_url=getattr(config, "mutate_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
                model=getattr(config, "mutate_model", os.getenv("MUTATE_MODEL", "llama3.2:3b")),
                temperature=getattr(config, "mutate_temperature", 0.8),
            )
            ok = isinstance(new_text, str) and "Return ONLY JSON" in new_text and "rating" in new_text
            ok = ok and ("1" in new_text and "5" in new_text)
            if ok and new_text.strip() != self.header_text.strip():
                print("\n--- PROMPT MUTATED ---")
                print("OLD (first 15 lines):")
                print("\n".join(self.header_text.splitlines()[:15]))
                print("\nNEW (first 15 lines):")
                print("\n".join(new_text.splitlines()[:15]))
                print("--- END PROMPT MUTATION ---\n")
                self.header_text = new_text
        except Exception as e:
            print(f"[prompt mutation failed] {e}")

    # neat-python calls configure_* using genome_config which is our PromptGenomeConfig
    def configure_new(self, config: PromptGenomeConfig):
        super().configure_new(config)
        if not getattr(self, "header_text", None):
            self.header_text = config.base_header
