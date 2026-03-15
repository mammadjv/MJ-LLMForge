import numpy as np
import torch
import tiktoken


def craete_dataset(dataset_folder, seq_len):
    train_data = np.memmap(f'{dataset_folder}/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(f'{dataset_folder}/val.bin', dtype=np.uint16, mode='r')

    enc = tiktoken.get_encoding("gpt2")
    vocab_len = enc.n_vocab
    gpt_train_dataset = GPTDataset(train_data, vocab_len, seq_len)
    gpt_val_dataset = GPTDataset(val_data, vocab_len, seq_len)

    return gpt_train_dataset, gpt_val_dataset


class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text,
                 vocab_len,
                 seq_len=8,
                ):

        # data is already encoded
        self.text = text
        self.seq_len = seq_len
        self.vocab_len = vocab_len

    def __len__(self):
        return self.text.shape[0] - self.seq_len - 1

    def __getitem__(self, idx):
        idx_ = idx
        return np.array(self.text[idx_:idx_+self.seq_len]), np.array(self.text[idx_+1:idx_+self.seq_len+1])

