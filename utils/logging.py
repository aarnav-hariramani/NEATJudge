from tqdm import tqdm

def tqdm_gen(it, desc, leave=False):
    return tqdm(it, desc=desc, leave=leave)
