import math
import glob

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist


class GPTDataset(IterableDataset):
    def __init__(self, data_dir, split, batch_size, seq_len, rank, world_size):
        shards = sorted(glob.glob(f"{data_dir}/edu_fineweb_train_*.bin"))
        self.shards = shards[1:] if split == 'train' else shards[:1]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.vocab_len = 50304

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        B, T = self.batch_size, self.seq_len
        tokens_per_batch = B * T

        # unique stream id among all workers on all ranks
        stream_id = self.rank * num_workers + worker_id
        num_streams = self.world_size * num_workers

        shard_idx = worker_id % len(self.shards)

        while True:
            data = np.memmap(self.shards[shard_idx], dtype=np.uint16, mode='r')
            data = data[256:]  # skip header

            pos = stream_id * tokens_per_batch
            step = num_streams * tokens_per_batch

            while pos + tokens_per_batch + 1 <= len(data):
                buf = np.array(data[pos : pos + tokens_per_batch + 1], dtype=np.int64)
                buf = torch.from_numpy(buf)
                x = buf[:-1].view(B, T)
                y = buf[1:].view(B, T)
                yield x, y
                pos += step

            shard_idx = (shard_idx + num_workers) % len(self.shards)