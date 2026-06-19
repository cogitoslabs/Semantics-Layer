import torch
import numpy as np
from typing import Dict

class MemmapDataset(torch.utils.data.Dataset):
    """Dataset wrapper for memory-mapped token arrays."""
    def __init__(self, tokens, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.num_blocks = len(tokens) // block_size

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        chunk = self.tokens[start:end]
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone()
        }
