"""
Target cache generation pipeline.

Generates target model logits against which the drafter is trained.
In the paper, this is a massive pipeline (38 TB for Qwen3-4B).
This module implements a compact version suitable for smoke testing.
"""

from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@torch.no_grad()
def generate_target_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate target model logits for input sequences.

    Args:
        model: Target HuggingFace model.
        tokenizer: Tokenizer.
        input_ids: [B, seq_len] token ids.
        attention_mask: [B, seq_len] attention mask.

    Returns:
        logits: [B, seq_len, V] target logits.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
    )
    return outputs.logits


def prepare_target_cache_entry(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    device: str = "cuda",
) -> dict:
    """Prepare a single target cache entry.

    Args:
        text: Input text.
        model: Target model.
        tokenizer: Tokenizer.
        max_length: Max sequence length.
        device: Target device.

    Returns:
        dict with input_ids and target_logits.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    logits = generate_target_logits(model, tokenizer, input_ids, attention_mask)

    return {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "target_logits": logits.cpu(),
    }
