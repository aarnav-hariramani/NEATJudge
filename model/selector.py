
import neat, numpy as np
def build_features(query_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(query_emb, cand_embs.shape)
    feats = np.concatenate([
        cand_embs,
        q,
        np.abs(cand_embs - q),
        cand_embs * q
    ], axis=1)
    return feats
class Selector:
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    def score(self, feats: np.ndarray) -> np.ndarray:
        out = [self.net.activate(x.astype(float))[0] for x in feats]
        return np.array(out, dtype=float)
