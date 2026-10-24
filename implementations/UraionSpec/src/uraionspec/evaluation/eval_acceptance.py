"""
Evaluation pipeline for DSpark speculative decoding.

Measures:
- Accepted length per decoding round
- Acceptance rate per position
- Vanilla decoding latency baseline
- Speculative decoding latency (tokens/sec)
"""

import time
from typing import Callable

import torch

from ..decoding import verify_block


@torch.no_grad()
def evaluate_acceptance(
    draft_model_fn: Callable,
    target_model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gamma: int = 7,
    num_steps: int = 10,
    temperature: float = 0.0,
    use_confidence: bool = False,
    device: str = "cuda",
) -> dict:
    """Evaluate speculative decoding acceptance rates.

    Args:
        draft_model_fn: function (anchor_ids, gamma, temp, conf) -> dict
        target_model: target HuggingFace model
        prompt_ids: [batch, seq_len] prompt token ids
        prompt_mask: [batch, seq_len] attention mask
        gamma: maximum block size
        num_steps: number of decoding steps to evaluate
        temperature: sampling temperature
        use_confidence: whether to use confidence-based scheduling
        device: computation device

    Returns:
        dict with metrics:
            - acceptance_rates: [gamma] mean acceptance rate per position
            - accepted_lengths: [num_steps] accepted tokens per step
            - mean_accepted_length: float
            - acceptance_rate: float
            - bonus_token_rate: float
    """
    batch_size = prompt_ids.shape[0]
    input_ids = prompt_ids.to(device)

    position_accepts = torch.zeros(gamma, device=device)
    position_counts = torch.zeros(gamma, device=device)
    accepted_lengths = []

    for step in range(num_steps):
        anchor_position = input_ids.shape[1] - 1
        anchor_ids = input_ids[:, anchor_position]

        # Draft
        draft_output = draft_model_fn(
            anchor_ids=anchor_ids,
            gamma=gamma,
            temperature=temperature,
            return_confidence=use_confidence,
        )
        draft_tokens = draft_output["draft_tokens"]
        draft_logits = draft_output["draft_logits"]

        # Target verification
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        target_out = target_model(full_input)
        target_logits = target_out.logits[:, -gamma:, :]

        # Acceptance check
        accepted, num_acc, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        for b in range(batch_size):
            n = num_acc[b].item()
            accepted_lengths.append(n + 1)  # +1 for bonus token
            for k in range(n):
                position_accepts[k] += 1
            position_counts[:gamma] += 1

        # Prepare next step: add accepted tokens + bonus
        next_input_ids = [input_ids]
        for b in range(batch_size):
            n = num_acc[b].item()
            new_toks = torch.cat([draft_tokens[b, :n], bonus[b].unsqueeze(0)])
            next_input_ids.append(new_toks.unsqueeze(0))
        input_ids = torch.cat(next_input_ids, dim=-1)

    acceptance_rates = position_accepts / position_counts.clamp_min(1)

    return {
        "acceptance_rates": acceptance_rates.cpu().tolist(),
        "accepted_lengths": accepted_lengths,
        "mean_accepted_length": float(torch.tensor(accepted_lengths).float().mean()),
        "acceptance_rate": position_accepts.sum().item() / position_counts.sum().item(),
    }


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    num_tokens: int = 50,
    num_warmup: int = 3,
    device: str = "cuda",
) -> dict:
    """Benchmark vanilla autoregressive decoding latency.

    Args:
        model: HuggingFace model.
        prompt_ids: [batch, seq_len] prompt tokens.
        num_tokens: Number of tokens to generate.
        num_warmup: Warmup iterations.
        device: Device.

    Returns:
        dict with: tokens_per_second, ms_per_token, total_time
    """
    model.eval()
    batch_size, seq_len = prompt_ids.shape
    input_ids = prompt_ids.to(device)

    # Warmup
    for _ in range(num_warmup):
        _ = model(input_ids[:, :1])

    # Benchmark
    current = input_ids[:, :1]
    times = []

    for _ in range(num_tokens):
        start = time.time()
        out = model(current)
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        current = torch.cat([current, next_token], dim=-1)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    total_time = sum(times)
    return {
        "tokens_per_second": num_tokens / total_time,
        "ms_per_token": 1000 * total_time / num_tokens,
        "total_time_s": total_time,
        "num_tokens": num_tokens,
    }
