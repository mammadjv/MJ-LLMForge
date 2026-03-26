import math
import glob

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import tiktoken


class QADataset(IterableDataset):
    def __init__(self, data_dir, split, batch_size, seq_len, rank, world_size):
        shards = sorted(glob.glob(f"{data_dir}/squad_train*.bin"))
        self.shards = shards[1:] if split == 'train' else shards[:1]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.vocab_len = 50304
        self.word_encoding = tiktoken.get_encoding("gpt2")
        self.ctx_separator = self.word_encoding.encode('Context:')

    def get_valid_tokens_mask(self, x, B, T):
        matches = (x[:, :-1] == self.ctx_separator[0]) & (x[:, 1:] == self.ctx_separator[1])
        exists = matches.any(dim=1)

        first_indices = torch.where(
            exists,
            matches.to(torch.uint8).argmax(dim=1),
            torch.tensor(-1, device=x.device)
        )

        flipped_matches = torch.flip(matches, dims=[1])
        last_indices_flipped = flipped_matches.to(torch.uint8).argmax(dim=1)

        last_indices = torch.where(
            exists,
            (matches.size(1) - 1) - last_indices_flipped,
            torch.tensor(-1, device=x.device)
        )

        arange = torch.arange(T, device=x.device).unsqueeze(0)
        after_first = arange >= first_indices.unsqueeze(1)
        before_last_start = arange < last_indices.unsqueeze(1)
        mask_bool = after_first & before_last_start & exists.unsqueeze(1)
        final_mask = torch.where(mask_bool, 1, -1)
        return final_mask


    def get_block_causal_mask(self, x, B, T):
        matches = (x[:, :-1] == self.ctx_separator[0]) & (x[:, 1:] == self.ctx_separator[1])
        coords = torch.nonzero(matches)

        # 1. Initialize a Block ID map for each position in the batch
        block_ids = torch.zeros((B, T), device=coords.device, dtype=torch.long)

        # 2. Mark each start point with a 1
        block_ids[coords[:, 0], coords[:, 1]] = 1

        # 3. Use cumsum to fill the segments with a unique ID
        block_ids = torch.cumsum(block_ids, dim=1)

        # 4. Generate the 2D Attention Mask (B, T, T)
        q_ids = block_ids.unsqueeze(2) # (B, T, 1)
        k_ids = block_ids.unsqueeze(1) # (B, 1, T)

        # Base mask: same block AND the block ID must be > 0 (to ignore the prefix)
        attn_mask_bool = (q_ids == k_ids) & (q_ids > 0)

        # 5. Apply Causal and Additive Constraints
        causal_mask = torch.tril(torch.ones((T, T), device=coords.device)).bool()
        final_mask = torch.where(attn_mask_bool & causal_mask, 0.0, float('-inf'))
        return final_mask

        
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

            pos = stream_id * tokens_per_batch
            step = num_streams * tokens_per_batch

            while pos + tokens_per_batch + 1 <= len(data):
                buf = np.array(data[pos : pos + tokens_per_batch + 1], dtype=np.int64)
                buf = torch.from_numpy(buf)
                x = buf[:-1].view(B, T)
                y = buf[1:].view(B, T)

                causal_mask = self.get_block_causal_mask(x, B, T)
                mask_x = self.get_valid_tokens_mask(x, B, T)
                mask_y = F.pad(mask_x[:, :-1], (1, 0), value=-1)

                yield x, y, mask_x, mask_y, causal_mask
                pos += step

            shard_idx = (shard_idx + num_workers) % len(self.shards)
