import numpy as np
import torch
import tiktoken


def create_dataset(dataset_folder, seq_len):

    gpt_train_dataset = GPTDataset(f'{dataset_folder}/train.bin', seq_len)
    gpt_val_dataset = GPTDataset(f'{dataset_folder}/val.bin', seq_len)

    return gpt_train_dataset, gpt_val_dataset


class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 seq_len=8,
                ):

        self.data_path = data_path
        print('data path:', self.data_path)
        text = np.memmap(self.data_path, dtype=np.uint16, mode='r')        
        self.data_len = text.shape[0]
        del text

        enc = tiktoken.get_encoding("gpt2")
        vocab_len = enc.n_vocab
        self.seq_len = seq_len
        self.vocab_len = vocab_len

    def __len__(self):
        return self.data_len - self.seq_len - 1

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.data_len - self.seq_len - 1)
        idx_ = idx
        data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        x = torch.from_numpy(np.array(data[idx : idx + self.seq_len]).astype(np.int64))
        y = torch.from_numpy(np.array(data[idx+1 : idx + self.seq_len + 1]).astype(np.int64))
        return x, y

