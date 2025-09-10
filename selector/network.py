
from typing import Optional
import numpy as np
try:
    import neat
except Exception:
    neat = None  # type: ignore

class Selector:
    """
    Wraps a NEAT feed-forward network to score feature rows.
    If NEAT/genome/config is unavailable, returns zeros (falls back to similarity-only path).
    """
    def __init__(self, genome=None, neat_config=None, neat_config_path: Optional[str] = None):
        self.net = None
        self.ready = False
        if genome is not None and neat is not None and (neat_config or neat_config_path):
            try:
                if neat_config is None:
                    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                              neat_config_path)
                self.net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
                self.ready = True
            except Exception:
                self.net = None
                self.ready = False

    def score(self, feats: np.ndarray) -> np.ndarray:
        if not self.ready or self.net is None:
            return np.zeros((feats.shape[0],), dtype="float32")
        out = [self.net.activate(row.tolist())[0] for row in feats]
        return np.asarray(out, dtype="float32")
