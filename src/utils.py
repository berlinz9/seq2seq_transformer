import torch, json, os
from math import exp
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    torch.save(state, path)

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def idxs_to_sentence(idxs, itos, stop_token='<eos>'):
    words = []
    for i in idxs:
        # support itos keys as str or int
        if isinstance(i, int):
            key = str(i)
        else:
            key = str(i)
        w = itos.get(key) or itos.get(int(key)) or itos.get(i)
        if w is None:
            continue
        if w == stop_token:
            break
        words.append(w)
    return ' '.join(words)
