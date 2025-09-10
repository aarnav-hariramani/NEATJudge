
from neatjudge.data.splits import make_splits
def test_splits_simple():
    rows = []
    for i in range(20):
        q = f"Q{i//4}"
        rows.append({"question": q, "choice": f"C{i%4}", "label": 1 if (i%4)==0 else 0})
    cfg = {"data": {"seed": 0, "bank_frac": 0.6, "val_frac": 0.1, "test_frac": 0.1}}
    splits = make_splits(rows, cfg)
    assert set(splits.keys()) == {"bank", "val", "test", "train"}
    def qs(part): return set(r["question"] for r in splits[part])
    assert qs("bank").isdisjoint(qs("val"))
    assert qs("bank").isdisjoint(qs("test"))
    assert qs("val").isdisjoint(qs("test"))
