import json, torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.src = obj['src']
        self.tgt = obj['tgt']
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return {'src': torch.tensor(self.src[idx], dtype=torch.long),
                'tgt': torch.tensor(self.tgt[idx], dtype=torch.long)}

def collate_fn(batch, pad_idx=0):
    srcs = [b['src'] for b in batch]
    tgts = [b['tgt'] for b in batch]
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    # for decoder input, we will feed all tokens except last (standard teacher forcing)
    tgt_input = tgt_pad[:, :-1]
    tgt_output = tgt_pad[:, 1:]
    return {'src': src_pad, 'tgt_input': tgt_input, 'tgt_output': tgt_output}
