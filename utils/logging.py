
from tqdm import tqdm
def tqdm_gen(iterable, desc, leave=False):
    return tqdm(iterable, desc=desc, leave=leave)
