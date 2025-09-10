import neat, numpy as np

def build_features(q_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(q_emb, cand_embs.shape)
    return np.concatenate([cand_embs, q, np.abs(cand_embs-q), cand_embs*q], axis=1)

class Selector:
    def __init__(self, genome, cfg):
        self.net = neat.nn.FeedForwardNetwork.create(genome, cfg)
    def score(self, feats: np.ndarray) -> np.ndarray:
        return np.array([self.net.activate(x.astype(float))[0] for x in feats], dtype=float)
