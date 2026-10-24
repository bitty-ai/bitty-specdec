"""
Standard lossless speculative decoding acceptance rule.

Implements the rejection sampling verification:
    x_k is accepted with probability min(1, p^t_k(x_k) / p^d_k(x_k))
    First rejection discards all subsequent tokens.
    One bonus token is always accepted from the target distribution.

Reference: Leviathan et al., 2023; Chen et al., 2023
"""

import torch


@torch.no_grad()
def verify_block(
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify a block of draft tokens against target model logits.

    Implements lossless speculative decoding verification:
    - Accept each token with probability min(1, p_target(x) / p_draft(x))
    - Stop at first rejection; discard subsequent draft tokens
    - Sample one bonus token from the target distribution at the rejection position

    Args:
        draft_tokens: [B, gamma] proposed draft token ids
        draft_logits: [B, gamma, V] draft model logits
        target_logits: [B, gamma, V] target model logits

    Returns:
        accepted_tokens: [B, max_accepted] variable-length accepted tokens
        num_accepted: [B] number of tokens accepted per batch element
        bonus_tokens: [B] bonus token sampled from target residual distribution
    """
    batch_size, gamma = draft_tokens.shape
    device = draft_tokens.device

    # Compute probabilities
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)

    # Get probability of sampled token under each distribution
    token_idx = draft_tokens.unsqueeze(-1)  # [B, gamma, 1]
    draft_prob_of_token = draft_probs.gather(-1, token_idx).squeeze(-1)  # [B, gamma]
    target_prob_of_token = target_probs.gather(-1, token_idx).squeeze(-1)  # [B, gamma]

    # Acceptance probability: min(1, p_target / p_draft)
    # To avoid division by zero, handle carefully
    ratio = torch.where(
        draft_prob_of_token > 0,
        target_prob_of_token / draft_prob_of_token,
        torch.ones_like(target_prob_of_token),  # unknown token -> always accept
    )
    accept_prob = torch.minimum(torch.ones_like(ratio), ratio)

    # Sample acceptance decisions
    random_values = torch.rand(batch_size, gamma, device=device)
    accepted = random_values < accept_prob  # [B, gamma]

    # Find first rejection per batch element
    # If all accepted, first_rejection = gamma
    first_rejection = gamma * torch.ones(batch_size, dtype=torch.long, device=device)
    for k in range(gamma):
        # Find first position where not accepted
        not_accepted_at_k = ~accepted[:, k]
        # Only update if this is the first rejection
        first_rejection = torch.where(
            first_rejection == gamma,
            torch.where(
                not_accepted_at_k,
                torch.full_like(first_rejection, k),
                first_rejection,
            ),
            first_rejection,
        )

    num_accepted = first_rejection  # [B]

    # Sampled accepted tokens: prefix up to first rejection
    max_accepted = num_accepted.max().item()
    if max_accepted > 0:
        indices = torch.arange(max_accepted, device=device).unsqueeze(0).expand(batch_size, -1)
        accept_mask = indices < num_accepted.unsqueeze(-1)  # [B, max_accepted]
        accepted_tokens = torch.where(
            accept_mask,
            draft_tokens[:, :max_accepted],
            torch.zeros_like(draft_tokens[:, :max_accepted]),
        )
    else:
        accepted_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    # Bonus token: sample from residual distribution at rejection position
    # p_bonus(x) = max(0, p_target(x) - p_draft(x)) / norm
    # For positions beyond gamma, target samples from its own distribution
    bonus_tokens = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    for b in range(batch_size):
        rej_pos = first_rejection[b].item()
        if rej_pos < gamma:
            # Sample from residual distribution at rejection position
            residual = torch.relu(target_probs[b, rej_pos] - draft_probs[b, rej_pos])
            norm = residual.sum()
            if norm > 0:
                residual_probs = residual / norm
                bonus_tokens[b] = torch.multinomial(residual_probs, 1).item()
            else:
                # If residual is zero (rare), sample from target distribution
                bonus_tokens[b] = torch.multinomial(target_probs[b, rej_pos], 1).item()
        else:
            # All tokens accepted: sample normally from the next target position
            # For simplicity, use the last target logits
            bonus_tokens[b] = torch.multinomial(target_probs[b, -1], 1).item()

    return accepted_tokens, num_accepted, bonus_tokens


@torch.no_grad()
def compute_expected_accept_length(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    gamma: int,
) -> torch.Tensor:
    """Compute expected acceptance length (τ) given distributions.

    τ = 1 + sum_{j=1}^{gamma} prod_{i=1}^{j} (1 - 0.5 * ||p^d_i - p^t_i||_1)

    Includes the bonus token in the count.

    Args:
        draft_logits: [B, gamma, V]
        target_logits: [B, gamma, V]
        gamma: block size

    Returns:
        expected_tau: [B] expected accepted length per batch element
    """
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)

    # Per-step acceptance probability: 1 - 0.5 * TV
    tv = 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
    accept_rate = (1.0 - tv).clamp(0.0, 1.0)  # [B, gamma]

    # Cumulative survival probability
    cumprod = torch.cumprod(accept_rate, dim=-1)  # [B, gamma]

    # Expected accepted tokens
    expected_tokens = cumprod.sum(dim=-1)  # [B]

    # τ = 1 (bonus) + expected draft tokens accepted = sum of cumprod
    return 1.0 + expected_tokens
