
import numpy as np
from neatjudge.selector.mmr import mmr_select
def test_mmr_shapes():
    rng = np.random.RandomState(0)
    cand = rng.randn(10, 8).astype("float32")
    q = rng.randn(8).astype("float32")
    idxs = mmr_select(cand, q, k=3, lam=0.3)
    assert len(idxs) == 3
    assert len(set(idxs)) == 3
