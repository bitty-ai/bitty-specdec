from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask


class DSparkForwardOutput:
    pass


def create_dspark_attention_mask(*,anchor_positions, block_keep_mask, seq_len, block_size, device):
    pass

def create_noise_embed(embed_tokens, input_ids, anchor_positions, block_keep_mask,*, mask_token_id, block_size):
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device
    
    noise_ids = torch.full(
        (bsz, num_blocks * block_size),  
        mask_token_id, 
        dtype = torch.long, 
        device = device 
    )
    
    block_starts = torch.arange(num_blocks, device = device) * block_size
    
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

    
        
    