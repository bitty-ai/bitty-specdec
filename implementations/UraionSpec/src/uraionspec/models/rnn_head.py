"""
DSpark-style RNN sequential head.

Maintains a recurrent state s_k across positions within a block,
accumulating full prefix history. Uses a GRU-like gated update:

    z_k = [s_{k-1}; W1[x_{k-1}]; h_k]
    s_k = sigmoid(W_g z_k) ⊙ s_{k-1} + (1 - sigmoid(W_g z_k)) ⊙ tanh(W_c z_k)
    B_k = W2^T tanh(W_o z_k)
"""

from typing import Optional

import torch
from torch import nn


class RNNHead(nn.Module):
    """RNN-based sequential head for DSpark.

    Unlike the memoryless Markov heads, position k can access the full
    prefix history x_<k through a GRU-like recurrent state.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int = 256,
        hidden_size: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.hidden_size = hidden_size
        self.markov_head_type = "rnn"

        # Embedding for previous token
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)

        # Joint projection: [s_{k-1}; W1[x_{k-1}]; h_k] -> 3 * r
        # (gate, candidate, output)
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

        # Output projection: tanh(output_state) -> vocab bias
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get W1 embedding for previous token. [*, r]"""
        return self.markov_w1(token_ids.long())

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single RNN step.

        Args:
            state: [*, r] previous recurrent state
            prev_embeddings: [*, r] W1[x_{k-1}]
            hidden_states: [*, d] backbone hidden at step k

        Returns:
            new_state: [*, r]
            bias: [*, V]
        """
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)

        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.markov_w2(torch.tanh(output_raw))

        return new_state, bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V] or [B, block_size, V]
            token_ids: [B, num_blocks, block_size] or [B, block_size]
            hidden_states: [B, num_blocks, block_size, d] or [B, block_size, d]

        Returns:
            Corrected logits with same shape as base_logits.
        """
        assert hidden_states is not None, "RNNHead requires hidden_states"
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        device = base_logits.device
        dtype = hidden_states.dtype

        state = torch.zeros(*leading_shape, self.markov_rank, device=device, dtype=dtype)
        output_logits = []

        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)

        return torch.stack(output_logits, dim=-2)

    def step(
        self,
        state: torch.Tensor,
        prev_token_ids: torch.Tensor,
        h_k: torch.Tensor,
        base_logits_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single inference step with stateful RNN.

        Args:
            state: [batch, r] recurrent state
            prev_token_ids: [batch] previous token ids
            h_k: [batch, d] backbone hidden at step k
            base_logits_k: [batch, V] base logits at step k

        Returns:
            new_state: [batch, r]
            corrected_logits_k: [batch, V]
        """
        prev_emb = self.get_prev_embeddings(prev_token_ids)
        new_state, bias = self._rnn_step(state, prev_emb, h_k)
        return new_state, base_logits_k + bias
