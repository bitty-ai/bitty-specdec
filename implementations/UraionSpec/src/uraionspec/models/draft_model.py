"""
DSpark draft model combining parallel backbone + sequential head + confidence head.

Architecture:
1. Parallel backbone (DFlash-style) produces hidden states and base logits
2. Sequential head (Markov/RNN) refines logits with inter-token dependency
3. Confidence head predicts per-position acceptance probabilities
"""

from typing import Optional

import torch
from torch import nn

from .markov_head import build_markov_head
from .confidence_head import ConfidenceHead


class DSparkDraftModel(nn.Module):
    """DSpark draft model combining parallel backbone and sequential heads.

    This is the core drafter that proposes γ candidate tokens per round.

    The parallel backbone processes all positions in one forward pass.
    The sequential head (Markov/RNN) refines logits with inter-token dependencies.
    The confidence head predicts per-position acceptance probabilities.

    Usage during inference:
        hidden_states, base_logits = model.backbone(anchor_ids)
        draft_logits = model.sequential_head(base_logits, token_ids, hidden_states)
        conf_logits = model.confidence_head(hidden_states, prev_embeddings)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        use_confidence_head: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank
        self.use_confidence_head = use_confidence_head

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # Token embedding (shared with target model in practice)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size) # TODO : get from target model 

        # Parallel backbone: stack of transformer decoder layers
        # Using a simplified config; in production this would be more sophisticated
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Sequential head
        self.markov_head_type = markov_head_type
        config = self._make_config()
        self.sequential_head = build_markov_head(config)

        # Confidence head
        if use_confidence_head:
            self.confidence_head = ConfidenceHead(
                hidden_size=hidden_size,
                markov_rank=markov_rank,
            )

        self._init_weights()

    def _make_config(self):
        """Create a config-like object for build_markov_head."""
        class _Config:
            pass
        cfg = _Config()
        cfg.markov_rank = self.markov_rank
        cfg.markov_head_type = self.markov_head_type
        cfg.vocab_size = self.vocab_size
        cfg.hidden_size = self.hidden_size
        return cfg

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
        return_confidence: bool = True,
    ) -> dict:
        """Forward pass of the draft model.

        Args:
            input_ids: [B, L] input token ids (anchor + mask tokens) , (B*N, Block-Size)
            prev_token_ids: [B, L] previous token ids for sequential head
                If None, uses input_ids shifted right (first token = anchor).
            return_confidence: Whether to compute confidence scores.

        Returns:
            dict with keys:
                - base_logits: [B, L, V] raw backbone logits
                - hidden_states: [B, L, d] backbone hidden states
                - draft_logits: [B, L, V] sequential-refined logits
                - confidence_logits: [B, L] confidence logits (if enabled)
        """
        # Embeddings
        hidden = self.embed_tokens(input_ids)  # [B, L, d]

        # Parallel backbone
        hidden = self.backbone(hidden)
        base_logits = self.lm_head(hidden)

        # Sequential refinement
        if self.sequential_head is not None and prev_token_ids is not None:
            draft_logits = self.sequential_head.apply_block_logits(
                base_logits, # parallel backbone logits
                token_ids=prev_token_ids, # prev token for sequential head
                hidden_states=hidden,
            )
        else:
            draft_logits = base_logits

        result = {
            "base_logits": base_logits,
            "hidden_states": hidden,
            "draft_logits": draft_logits,
        }

        # Confidence prediction
        if return_confidence and self.use_confidence_head and prev_token_ids is not None:
            if self.sequential_head is not None:
                prev_emb = self.sequential_head.get_prev_embeddings(prev_token_ids)
            else:
                prev_emb = self.embed_tokens(prev_token_ids)[:, :, :self.markov_rank]
            
            result["confidence_logits"] = self.confidence_head(hidden, prev_emb)

        return result


    # sampling / inference time 
    def sample_block(
        self,
        anchor_ids: torch.Tensor,
        gamma: int,
        temperature: float = 0.0,
        return_confidence: bool = True,
    ) -> dict:
        """Generate a draft block autoregressively with sequential head.

        Args:
            anchor_ids: [batch] anchor/bonus token from previous round
            gamma: number of draft tokens to generate
            temperature: sampling temperature (0 = greedy)
            return_confidence: whether to compute confidence scores

        Returns:
            dict with keys:
                - draft_tokens: [batch, gamma] sampled draft tokens
                - draft_logits: [batch, gamma, V] corrected logits
                - confidence: [batch, gamma] confidence scores (if enabled)
        """
        batch_size = anchor_ids.shape[0]
        device = anchor_ids.device

        # Create input: [anchor, mask, mask, ..., mask] (1 + gamma - 1 = gamma)
        mask_token_id = 0  # placeholder; will be overridden by embeddings
        input_ids = torch.full(
            (batch_size, gamma),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids[:, 0] = anchor_ids

        # Embed and run backbone in one shot
        hidden = self.embed_tokens(input_ids)
        hidden = self.backbone(hidden)
        base_logits = self.lm_head(hidden)

        # Autoregressive sampling through sequential head
        if self.sequential_head is not None:
            # Use sample_block_tokens for autoregressive generation
            sampled_tokens, corrected_logits = self.sequential_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=anchor_ids,
                hidden_states=hidden,
                temperature=temperature,
            )
        else:
            # No sequential head: sample independently from base_logits
            if temperature > 0:
                probs = torch.softmax(base_logits / temperature, dim=-1)
                sampled_tokens = torch.multinomial(
                    probs.reshape(-1, self.vocab_size),
                    num_samples=1,
                ).reshape(batch_size, gamma)
            else:
                sampled_tokens = base_logits.argmax(dim=-1)
            corrected_logits = base_logits

        result = {
            "draft_tokens": sampled_tokens,
            "draft_logits": corrected_logits,
        }

        # Confidence scores
        if return_confidence and self.use_confidence_head:
            prev_token_ids = torch.cat(
                [anchor_ids.unsqueeze(1), sampled_tokens[:, :-1]],
                dim=1,
            )
            if self.sequential_head is not None:
                prev_emb = self.sequential_head.get_prev_embeddings(prev_token_ids)
            else:
                prev_emb = self.embed_tokens(prev_token_ids)[:, :, :self.markov_rank]
            conf_logits = self.confidence_head(hidden, prev_emb)
            result["confidence"] = torch.sigmoid(conf_logits)

        return result
