"""
DSpark confidence head.

Predicts per-position conditional acceptance probability c_k = P(accept_k | accept_<k).
Uses a lightweight linear projection + sigmoid:

    c_k = sigmoid(w^T [h_k; W1[x_{k-1}]])

Supervised by the analytical acceptance rate c*_k = 1 - 0.5 * ||p^d_k - p^t_k||_1.
"""


import torch
from torch import nn


class ConfidenceHead(nn.Module):
    """DSpark confidence head predicting per-position acceptance probability.

    Input features: [backbone hidden state; Markov embedding of previous token]
    Output: scalar c_k ∈ (0, 1) via sigmoid
    """

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int = 256,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence scores.

        Args:
            hidden_states: [*, d] backbone hidden states
            prev_token_embeddings: [*, r] Markov embeddings of previous tokens

        Returns:
            logits: [*, 1] pre-sigmoid confidence logits
        """
        features = torch.cat([hidden_states, prev_token_embeddings], dim=-1)
        return self.proj(features).squeeze(-1)


def compute_accept_rate(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute analytical per-position acceptance rate.

    c*_k = 1 - 0.5 * ||p^d_k - p^t_k||_1

    Args:
        draft_logits: [*, V] draft model logits
        target_logits: [*, V] target model logits

    Returns:
        accept_rate: [*] acceptance probability per position, ∈ [0, 1]
    """
    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    tv_distance = 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
    return (1.0 - tv_distance).clamp(0.0, 1.0)
