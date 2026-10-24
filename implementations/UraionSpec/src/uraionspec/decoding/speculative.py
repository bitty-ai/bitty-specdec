"""
Speculative decoding runner.

Orchestrates the full speculative decoding loop:
1. Draft model proposes γ candidate tokens
2. Target model verifies all candidates in parallel
3. Acceptance rule determines accepted prefix + bonus token
4. Repeat until sequence is complete
"""

from typing import Callable, Optional

import torch
from transformers import PreTrainedModel

from .acceptance import verify_block
from .scheduler import (
    ThroughputProfile,
    hardware_aware_prefix_scheduler,
)


@torch.no_grad()
def speculative_decode_step(
    draft_model: Callable,
    target_model: PreTrainedModel,
    input_ids: torch.Tensor,
    anchor_position: int,
    gamma: int,
    schedule_length: Optional[int] = None,
    temperature: float = 0.0,
    use_confidence: bool = False,
) -> dict:
    """Single step of speculative decoding.

    Args:
        draft_model: function that takes (anchor_id, gamma, temp, conf) -> dict
        target_model: HuggingFace model
        input_ids: [batch, seq_len] current sequence
        anchor_position: index of anchor token (last verified token)
        gamma: maximum draft block size
        schedule_length: if set, only verify first schedule_length tokens
        temperature: sampling temperature for draft
        use_confidence: whether draft model returns confidence scores

    Returns:
        dict with keys:
            new_input_ids: [batch, seq_len + new_tokens]
            num_new_tokens: [batch] accepted + bonus
            accepted_tokens: [batch, accepted]
            bonus_token: [batch]
            schedule_length: int, actual verification length used
            acceptance_rate: mean acceptance rate at each position
    """
    batch_size = input_ids.shape[0]

    # Step 1: Get anchor token (last token)
    anchor_ids = input_ids[:, anchor_position]

    # Step 2: Draft model proposes candidates
    if use_confidence:
        draft_output = draft_model(
            anchor_ids=anchor_ids,
            gamma=gamma,
            temperature=temperature,
            return_confidence=True,
        )
    else:
        draft_output = draft_model(
            anchor_ids=anchor_ids,
            gamma=gamma,
            temperature=temperature,
            return_confidence=False,
        )

    draft_tokens = draft_output["draft_tokens"]  # [B, gamma]
    draft_logits = draft_output["draft_logits"]  # [B, gamma, V]

    # Step 3: Scheduler determines verification length
    if schedule_length is None and use_confidence and "confidence" in draft_output:
        confidence = draft_output["confidence"]  # [B, gamma]
        conf_list = [confidence[b] for b in range(batch_size)]
        profile = ThroughputProfile()
        lengths = hardware_aware_prefix_scheduler(conf_list, profile, gamma)
        schedule_length = max(lengths) if lengths else gamma
    elif schedule_length is None:
        schedule_length = gamma

    actual_gamma = min(schedule_length, gamma)

    # Step 4: Target model verifies candidates
    # Build input for target: [prompt_tokens + draft_tokens[:, :actual_gamma]]
    full_input = torch.cat([input_ids, draft_tokens[:, :actual_gamma]], dim=1)

    target_outputs = target_model(
        input_ids=full_input,
        use_cache=True,
        output_hidden_states=False,
    )
    # Target logits for the draft positions
    target_logits = target_outputs.logits[:, -actual_gamma:, :]  # [B, actual_gamma, V]

    # Step 5: Acceptance verification
    accepted_tokens, num_accepted, bonus_tokens = verify_block(
        draft_tokens[:, :actual_gamma],
        draft_logits[:, :actual_gamma, :],
        target_logits,
    )

    # Step 6: Build new input_ids
    new_tokens_list = []
    for b in range(batch_size):
        n_acc = num_accepted[b].item()
        # Accepted tokens
        accepted = draft_tokens[b, :n_acc]
        # Bonus token
        bonus = bonus_tokens[b]
        new_tokens = torch.cat([accepted, bonus.unsqueeze(0)])
        new_tokens_list.append(new_tokens)

    return {
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted_tokens,
        "num_accepted": num_accepted,
        "bonus_tokens": bonus_tokens,
        "schedule_length": schedule_length,
    }
