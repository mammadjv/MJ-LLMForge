import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import tiktoken


def get_vocab_len():
    return 50304


class QADataset(IterableDataset):
    def __init__(self, data_dir, split, batch_size, seq_len, rank=0, world_size=1):
        all_shards = sorted(glob.glob(f"{data_dir}/squad_train*.bin"))
        if len(all_shards) == 0:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")

        if split == 'val':
            self.shards = all_shards[:1]

        elif split == 'test':
            self.shards = all_shards[1:3]

        elif split == 'train':
            train_shards = all_shards[3:]  # 21 shards
            self.shards = train_shards[rank::world_size]
            if not self.shards:
                raise ValueError(f"Rank {rank} has 0 shards assigned! Total train shards: {len(train_shards)}, World size: {world_size}")

        else:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        
        self.word_encoding = tiktoken.get_encoding("gpt2")
        self.ans_tokens = torch.tensor(self.word_encoding.encode("Answer:"))

        self.ctx_tokens = torch.tensor(self.word_encoding.encode("Context:"))

        self.eot_token = self.word_encoding.eot_token
        self.empty_answer_tokens = torch.tensor(self.word_encoding.encode_ordinary(" "), dtype=torch.long)
        
    def get_valid_tokens_mask(self, x, B, T):
        ans_seq = self.ans_tokens.to(x.device)
        ctx_seq = self.ctx_tokens.to(x.device)
        empty_seq = self.empty_answer_tokens.to(x.device)
    
        ans_win = ans_seq.numel()
        ctx_win = ctx_seq.numel()
    
        # Find Context: and Answer:
        ctx_match = (x.unfold(1, ctx_win, 1) == ctx_seq).all(dim=2)
        ans_match = (x.unfold(1, ans_win, 1) == ans_seq).all(dim=2)

        ctx_starts = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        ans_starts = torch.zeros((B, T), dtype=torch.bool, device=x.device)

        ctx_starts[:, :ctx_match.size(1)] = ctx_match
        ans_starts[:, :ans_match.size(1)] = ans_match
    
        mask_bool = torch.zeros((B, T), dtype=torch.bool, device=x.device)
    
        for b in range(B):
            ctx_coords = torch.nonzero(ctx_starts[b], as_tuple=False).flatten()
    
            # Only Context -> next Context blocks are complete
            for i in range(ctx_coords.numel() - 1):
                block_start = ctx_coords[i].item()
                block_end = ctx_coords[i + 1].item()
    
                # Find Answer: inside this block
                ans_coords = torch.nonzero(ans_starts[b, block_start:block_end], as_tuple=False).flatten()
    
                if ans_coords.numel() != 1:
                    continue
    
                ans_start = block_start + ans_coords[0].item()
                answer_start = ans_start + ans_win
    
                # Find EOT after Answer: but before next Context:
                eot_coords = torch.nonzero(x[b, answer_start:block_end] == self.eot_token, as_tuple=False).flatten()
    
                if eot_coords.numel() == 0:
                    continue
    
                eot_pos = answer_start + eot_coords[0].item()
    
                # Tokens that make up the actual answer
                answer_tokens = x[b, answer_start:eot_pos]
                mask_bool[b, answer_start:eot_pos + 1] = True
    
        return torch.where(
            mask_bool,
            torch.ones((B, T), dtype=torch.long, device=x.device),
            -torch.ones((B, T), dtype=torch.long, device=x.device),
        )


    def get_block_causal_mask(self, x, B, T):
        ctx_seq = self.ctx_tokens.to(x.device)
        ctx_win = len(ctx_seq)
    
        ctx_match = (x.unfold(1, ctx_win, 1) == ctx_seq).all(dim=2)
    
        ctx_starts = torch.zeros((B, T), device=x.device, dtype=torch.bool)
        ctx_starts[:, :ctx_match.size(1)] = ctx_match
    
        block_ids = torch.cumsum(ctx_starts.long(), dim=1)
    
        # Number of Context: markers in each row
        num_blocks = ctx_starts.sum(dim=1)
    
        # Keep only blocks that have another Context: after them
        # So final block is removed
        valid_pos = block_ids > 0
        valid_pos &= block_ids < num_blocks.unsqueeze(1)
    
        q_ids = block_ids.unsqueeze(2)
        k_ids = block_ids.unsqueeze(1)
    
        same_block = q_ids == k_ids
        q_valid = valid_pos.unsqueeze(2)
        k_valid = valid_pos.unsqueeze(1)

        causal = torch.tril(
            torch.ones((T, T), device=x.device, dtype=torch.bool)
        ).unsqueeze(0)

        allowed = same_block & q_valid & k_valid & causal

        # Invalid query positions attend only to themselves.
        # They are ignored by the loss anyway.
        identity = torch.eye(
            T, device=x.device, dtype=torch.bool
        ).unsqueeze(0)
        
        invalid_queries = ~valid_pos
        
        allowed |= identity & invalid_queries.unsqueeze(-1)
        
        return torch.where(allowed, 0.0, float("-inf"))


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Divide rank shards further among DataLoader workers
        worker_shards = self.shards[worker_id::num_workers]
        if not worker_shards:
            return

        # Split into dedicated helper generators
        if self.split == "test":
            yield from self._iter_test(worker_shards)

        else:
            yield from self._iter_train_val(worker_shards)

    def _iter_test(self, worker_shards):
        ctx_win = self.ctx_tokens.numel()
        ans_win = self.ans_tokens.numel()

        for shard in worker_shards:
            data = np.memmap(shard, dtype=np.uint16, mode="r")
            data_tensor = torch.from_numpy(np.asarray(data, dtype=np.int64))

            ctx_matches = (data_tensor.unfold(0, ctx_win, 1) == self.ctx_tokens).all(dim=1)
            ctx_coords = torch.nonzero(ctx_matches, as_tuple=False).flatten()

            for i in range(ctx_coords.numel() - 1):
                start_idx = ctx_coords[i].item()
                end_idx = ctx_coords[i + 1].item()
                sample_tokens = data_tensor[start_idx:end_idx]

                if sample_tokens.numel() < ans_win:
                    continue

                ans_matches = (sample_tokens.unfold(0, ans_win, 1) == self.ans_tokens).all(dim=1)
                ans_coords = torch.nonzero(ans_matches, as_tuple=False).flatten()

                if ans_coords.numel() != 1:
                    continue

                answer_start = ans_coords[0].item()
                prompt_cutoff = answer_start + ans_win

                # Find the real EOT token
                eot_coords = torch.nonzero(sample_tokens[prompt_cutoff:] == self.eot_token, as_tuple=False).flatten()
                if eot_coords.numel() == 0:
                    continue

                eot_pos = prompt_cutoff + eot_coords[0].item()
                prompt = sample_tokens[:prompt_cutoff]

                # Include answer + EOT, but NOT the newline
                ground_truth = sample_tokens[prompt_cutoff : eot_pos + 1]

                yield prompt, ground_truth

    def _iter_train_val(self, worker_shards):
        B, T = self.batch_size, self.seq_len
        tokens_per_batch = B * (T + 1)
        shard_idx = 0

        while True:
            shard = worker_shards[shard_idx]
            data = np.memmap(shard, dtype=np.uint16, mode='r')
            num_tokens = len(data)

            pos = 0
            while pos + tokens_per_batch <= num_tokens:
                buf = np.array(data[pos : pos + tokens_per_batch], dtype=np.int64)
                buf_tensor = torch.from_numpy(buf).view(B, T + 1)

                x = buf_tensor[:, :-1]
                y = buf_tensor[:, 1:].clone()

                causal_mask = self.get_block_causal_mask(x, B, T).unsqueeze(1)
                mask_x = self.get_valid_tokens_mask(x, B, T)
                mask_y = F.pad(mask_x[:, 1:], (0, 1), value=-1)

                y[mask_y == -1] = -1

                if (y != -1).any():
                    yield (x, causal_mask), y

                pos += tokens_per_batch

            # Move smoothly to the next assigned shard
            shard_idx = (shard_idx + 1) % len(worker_shards)
