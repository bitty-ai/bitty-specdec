"""
DSpark-style Markov sequential head.

Implements low-rank transition bias B(x_{k-1}, x_k) = W1[x_{k-1}] @ W2
where W1 ∈ R^{V×r} (embedding), W2 ∈ R^{r×V} (logit projection), r=256 by default.
Also includes a gated variant (GatedMarkovHead) that modulates the bias
with backbone hidden states.
"""

from typing import Optional

import torch
from torch import nn


class VanillaMarkov(nn.Module):
    """Vanilla Markov head with low-rank transition bias.

    B(prev_token, next_token) = W1[prev_token] @ W2
    """

    def __init__(self, *, vocab_size: int, markov_rank: int = 256):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"

        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )

        # Low-rank factorization: W1 embeddings + W2 logit projection
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get W1 embedding for previous token. [*, r]"""
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project latent states through W2. [*, V]"""
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Markov bias for a single step. [*, V]"""
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Markov bias to logits for one step."""
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor, # from the draft model's , parallel backbone
        *,
        token_ids: torch.Tensor, # prev token ids
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Markov bias to all positions in a block (teacher-forced).

        Args:
            base_logits: [B, num_blocks, block_size, V] or [B, block_size, V]
            token_ids: [B, num_blocks, block_size] or [B, block_size] — previous tokens
            hidden_states: ignored in vanilla mode

        Returns:
            Logits with Markov bias applied.
        """
        if base_logits.size(-2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias # add bias to these base-logits here  


    # sampling time / inference time
    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling through the Markov head.

        Args:
            base_logits: [batch, proposal_len, vocab]
            first_prev_token_ids: [batch] — anchor/bonus token
            hidden_states: [batch, proposal_len, d] or None
            temperature: 0 = greedy, >0 = multinomial sampling

        Returns:
            sampled_tokens: [batch, proposal_len]
            corrected_logits: [batch, proposal_len, vocab]
        """
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()

        for step_idx in range(proposal_len):
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx]
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],
                token_ids=prev_token_ids,
                hidden_states=step_hidden,
            )
            corrected_logits.append(step_logits.unsqueeze(1))

            if temperature > 0:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token_ids = step_logits.argmax(dim=-1)

            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkov):
    """Gated Markov head with hidden-state modulation.

    Computes gate = sigmoid(W_g[hidden; W1[prev]])
    then bias = W2(gate * W1[prev])
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int = 256,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        self.markov_head_type = "gated"

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gating factor. [*, r]"""
        assert hidden_states is not None, "GatedMarkovHead requires hidden_states"
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(dtype=prev_embeddings.dtype) # which all to allow which all to not allow here 
        return self.project_bias(gate * prev_embeddings)


def build_markov_head(config) -> Optional[nn.Module]:
    """Build a Markov head from a config object.

    Config attributes used:
        markov_rank (int): Rank of low-rank factorization (0 = no markov head)
        markov_head_type (str): "vanilla", "gated", or "rnn"
        vocab_size (int): Vocabulary size
        hidden_size (int): Hidden size (for gated/rnn variants)
    """
    markov_rank = int(getattr(config, "markov_rank", 256))
    if markov_rank <= 0:
        return None

    head_type = getattr(config, "markov_head_type", "vanilla").lower()
    if head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    elif head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    else:
        raise ValueError(f"Unsupported markov_head_type: {head_type}")
