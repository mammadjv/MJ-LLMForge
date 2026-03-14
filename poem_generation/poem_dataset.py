import numpy as np
import torch

class PoemDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text,
                 chars,
                 seq_len=8,
                ):

        vocab_size = len(chars)
        self.seq_len = seq_len

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        self.data = self.encode(text)

    def __len__(self):
        return int(len(self.data)/2) - self.seq_len

    def __getitem__(self, idx):
        idx_ = idx * 2
        return np.array(self.data[idx_:idx_+self.seq_len]), np.array(self.data[idx_+1:idx_+self.seq_len+1])