
def fitness_score(acc, stab, diversity, length_pen, w):
    wa = w.get("w_accuracy", 0.6)
    ws = w.get("w_stability", 0.2)
    wd = w.get("w_diversity", 0.1)
    wl = w.get("w_length_penalty", 0.1)
    return wa*acc + ws*stab + wd*diversity - wl*length_pen
