import random, numpy as np

def set_global_seed(s:int):
    random.seed(s); np.random.seed(s)
