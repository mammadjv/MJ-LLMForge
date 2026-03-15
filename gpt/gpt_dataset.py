import math

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import tiktoken


def create_dataset(dataset_folder, seq_len, world_size, rank):

    gpt_train_dataset = GPTDataset(f'{dataset_folder}/train.bin', seq_len, world_size, rank)
    gpt_val_dataset = GPTDataset(f'{dataset_folder}/val.bin', seq_len, world_size, rank)

    return gpt_train_dataset, gpt_val_dataset


class GPTDataset(IterableDataset):
    def __init__(self, data_path, seq_len, world_size, rank):
        self.data_path = data_path
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        
        # Open memmap
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        
        # Divide the file into chunks for each GPU
        self.tokens_per_rank = self.total_tokens // world_size
        self.start_idx = self.rank * self.tokens_per_rank
        self.end_idx = self.start_idx + self.tokens_per_rank
        
        enc = tiktoken.get_encoding("gpt2")
        vocab_len = enc.n_vocab
        self.vocab_len = vocab_len


    def __iter__(self):
        # 1. Divide total data across GPUs (Ranks)
        per_rank = int(math.floor(self.total_tokens / self.world_size))
        iter_start = self.rank * per_rank
        iter_end = iter_start + per_rank
    
        # 2. Further divide rank data across DataLoader Workers
        worker_info = torch.utils.data.get_worker_info()

        worker_seed = np.random.get_state()[1][0] + (worker_info.id if worker_info else 0)
        np.random.seed(worker_seed)
    
        if worker_info is not None:
            per_worker = int(math.floor((iter_end - iter_start) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = iter_start + (worker_id * per_worker)
            iter_end = iter_start + per_worker
    
        samples_yielded = 0
        # Use a safer range to avoid index-out-of-bounds at the very end of the file
        max_samples = (iter_end - iter_start - self.seq_len - 1) // self.seq_len 
        
        while True:
            # np.random.randint is now unique per worker thanks to the seed above
            idx = np.random.randint(iter_start, iter_end - self.seq_len - 1)
            x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
            y = torch.from_numpy(self.data[idx+1 : idx + self.seq_len + 1].astype(np.int64))
            yield x, y
            
            samples_yielded += 1
            if samples_yielded >= max_samples:
                break

                