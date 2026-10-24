"""Sampling utilities for speculative decoding."""

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Convert logits to probabilities with temperature."""
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Sample tokens from logits with temperature.

    Args:
        logits: [B, L, V] or [B, V]
        temperature: 0 = greedy, >0 = multinomial

    Returns:
        token_ids: same shape as logits minus last dim
    """
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)

    bsz = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.size(-1)) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled.reshape(*bsz)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample from residual distribution p_target - p_draft (clamped).

    Used for bonus token sampling in speculative decoding.

    Args:
        target_probs: [B, V] target probabilities
        draft_probs: [B, V] draft probabilities

    Returns:
        token_ids: [B] sampled tokens
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    # If residual is near-zero, fall back to target distribution
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    flat = residual.reshape(-1, residual.size(-1))
    sampled = torch.multinomial(flat, num_samples=1)
    return sampled.reshape(residual.shape[:-1])


def gather_token_probs(
    probs: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather probabilities of specific tokens from a probability tensor.

    Args:
        probs: [*, V] probability tensor
        token_ids: [*] token indices

    Returns:
        token_probs: [*] probability of each token
    """
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
