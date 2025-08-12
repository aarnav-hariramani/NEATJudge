
from data.splits import make_splits
def test_splits_simple():
    rows = [{"query": "q", "response":"r", "label":1.0} for _ in range(100)]
    cfg = {"data":{"seed":1,"test_frac":0.1,"val_frac":0.1,"bank_frac":0.6}}
    sp = make_splits(rows, cfg)
    n = len(rows)
    assert len(sp["test"]) == int(0.1*n)
    assert len(sp["val"]) == int(0.1*(n-int(0.1*n)))
