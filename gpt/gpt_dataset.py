import math
import glob

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist


class GPTDataset(IterableDataset):
    def __init__(self, data_dir, split, batch_size, seq_len, rank, world_size, num_workers=1):
        shards = sorted(glob.glob(f"{data_dir}/edu_fineweb_train_*.bin"))
        self.shards = shards[1:] if split == 'train' else shards[:1]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.vocab_len = 50304
        self.shard_idx = 0
        self.shard_offset = 0
        

    def update_dataset_sharding(self, update_step, grad_accum_steps):
        B, T = self.batch_size, self.seq_len
        tokens_per_batch = B * T

        num_streams = self.world_size * self.num_workers
        stream_step = num_streams * tokens_per_batch

        shard_len = len(np.memmap(self.shards[0], dtype=np.uint16, mode='r')) - 256
        rounds_per_shard = (shard_len - 1) // stream_step

        consumed_microsteps = update_step * grad_accum_steps
        consumed_rounds = consumed_microsteps // self.num_workers

        self.shard_idx = (consumed_rounds // rounds_per_shard) % len(self.shards)
        self.shard_offset = (consumed_rounds % rounds_per_shard) * stream_step

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

        stream_id = self.rank * num_workers + worker_id
        num_streams = self.world_size * num_workers

        shard_idx = self.shard_idx
        shard_offset = self.shard_offset

        while True:
            data = np.memmap(self.shards[shard_idx], dtype=np.uint16, mode='r')
            data = data[256:]  # skip header
            
            if self.rank == 0 and worker_id == 0:
                print(f"\nReading shard {shard_idx}: {self.shards[shard_idx]}, len={len(data)}", flush=True)

            pos = shard_offset + stream_id * tokens_per_batch
            step = num_streams * tokens_per_batch

            while pos + tokens_per_batch + 1 <= len(data):
                buf = np.array(data[pos : pos + tokens_per_batch + 1], dtype=np.int64)
                buf = torch.from_numpy(buf)
                x = buf[:-1].view(B, T)
                y = buf[1:].view(B, T)
                yield (x,), y
                pos += step

            shard_offset = 0
            shard_idx = (shard_idx + 1) % len(self.shards)
