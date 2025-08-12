
import numpy as np
from model.mmr import mmr_select
def test_mmr_shapes():
    q = np.random.randn(384); q = q/np.linalg.norm(q)
    C = np.random.randn(10,384); C = C/np.linalg.norm(C,axis=1,keepdims=True)
    idx = mmr_select(C, q, K=5, lam=0.3)
    assert 0 < len(idx) <= 5
