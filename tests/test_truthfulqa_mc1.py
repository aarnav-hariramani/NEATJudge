
from neatjudge.metrics.truthfulqa import mc1_accuracy
def test_mc1_grouping_and_argmax():
    rows, scores = [], []
    for i in range(4):
        rows.append({"question": "Q0", "choice": f"A{i}", "label": 1 if i==2 else 0})
        scores.append(0.1 * i)  # argmax at i=3 (wrong)
    for i in range(4):
        rows.append({"question": "Q1", "choice": f"A{i}", "label": 1 if i==1 else 0})
        scores.append(1.0 if i==1 else 0.0)  # correct
    mc1 = mc1_accuracy(rows, scores)
    assert abs(mc1 - 50.0) < 1e-6
