from neatjudge.eval.metrics import mc1, mc2
def test_mc1_mc2():
    assert mc1([0,1,2],[0,1,2])==100.0
    assert mc1([0,1,2],[1,1,2])== 100.0 * 2/3
    assert mc2([0.9,0.6],[1.0,1.0])>0
