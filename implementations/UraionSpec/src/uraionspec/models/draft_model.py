"""
DSpark draft model combining parallel backbone + sequential head + confidence head.

Architecture:
1. Parallel backbone (DFlash-style) attends to projected target hidden states (KV injection)
   and produces hidden states and base logits for all block positions at once
2. Sequential head (Markov/RNN) refines logits with inter-token dependency
3. Confidence head predicts per-position acceptance probabilities
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .markov_head import build_markov_head
from .confidence_head import ConfidenceHead
from .dflash_backbone import DFlashBackbone, DSparkAttentionMask


class DSparkDraftModel(nn.Module):
    """DSpark draft model combining parallel backbone and sequential heads.

    Each block is [anchor, MASK x (block_size - 1)]. The block anchored at position p
    attends to target context features at positions < p and bidirectionally within itself.
    Slot k predicts token p + k + 1.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 5,
        num_attention_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        num_target_layers: int = 5,
        block_size: int = 7,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        use_confidence_head: bool = True,
        rope_theta: float = 1_000_000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.markov_rank = markov_rank
        self.use_confidence_head = use_confidence_head
        self.num_target_layers = num_target_layers

        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.mask_embed = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(num_target_layers * hidden_size, hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden_size, eps=1e-6)

        self.backbone = DFlashBackbone(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            rope_theta=rope_theta,
        )

        self.markov_head_type = markov_head_type
        self.sequential_head = build_markov_head(self._make_config())

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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_embed, mean=0.0, std=0.02)
        if self.sequential_head is not None:
            nn.init.zeros_(self.sequential_head.markov_w2.weight)  # start as identity on base logits

    def share_target_embeddings(self, target_model: nn.Module, freeze: bool = True):
        """Reuse the target's token embedding and lm_head so drafts live in the target's output space."""
        self.embed_tokens.weight = target_model.get_input_embeddings().weight
        self.lm_head.weight = target_model.get_output_embeddings().weight
        # lm_head was trained on the target's final RMSNorm gain (mean ~2.8 for Qwen3-4B), not 1
        with torch.no_grad():
            self.backbone.norm.weight.copy_(target_model.model.norm.weight.float())
        if freeze:
            self.embed_tokens.weight.requires_grad_(False)
            self.lm_head.weight.requires_grad_(False)

    def trainable_state_dict(self) -> dict:
        frozen = {name for name, p in self.named_parameters(remove_duplicate=False) if not p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k not in frozen}

    def _embed_blocks(self, anchor_ids: torch.Tensor) -> torch.Tensor:
        """anchor_ids: [B, N] -> [B, N * block_size, d]"""
        B, N = anchor_ids.shape
        anchor_emb = self.embed_tokens(anchor_ids).to(self.mask_embed.dtype)
        masks = self.mask_embed.expand(B, N, self.block_size - 1, -1)
        return torch.cat([anchor_emb.unsqueeze(2), masks], dim=2).view(B, N * self.block_size, -1)

    def _forward_backbone(
        self,
        anchor_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N = anchor_ids.shape
        L_ctx = target_hidden_states.shape[1]
        # per-layer RMS normalization: residual-stream scales differ by ~100x across target layers
        per_layer = target_hidden_states.view(B, L_ctx, self.num_target_layers, self.hidden_size)
        per_layer = F.rms_norm(per_layer.float(), (self.hidden_size,), eps=1e-6)
        context = self.hidden_norm(self.fc(per_layer.view(B, L_ctx, -1).to(self.fc.weight.dtype)))
        hidden = self._embed_blocks(anchor_ids)

        offsets = torch.arange(self.block_size, device=anchor_ids.device)
        position_ids = (anchor_positions[:, :, None] + offsets).view(B, N * self.block_size)
        context_position_ids = torch.arange(L_ctx, device=anchor_ids.device).expand(B, -1)
        mask = DSparkAttentionMask.create_block_context_mask(
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            context_len=L_ctx,
            context_mask=context_mask,
        )
        hidden = self.backbone(
            hidden,
            context,
            mask,
            position_ids=position_ids,
            context_position_ids=context_position_ids,
        )
        return hidden.view(B, N, self.block_size, -1)

    def _confidence_logits(self, hidden: torch.Tensor, prev_token_ids: torch.Tensor) -> torch.Tensor:
        if self.sequential_head is not None:
            prev_emb = self.sequential_head.get_prev_embeddings(prev_token_ids)
        else:
            prev_emb = self.embed_tokens(prev_token_ids)[..., :self.markov_rank]
        return self.confidence_head(hidden, prev_emb.to(hidden.dtype))

    def forward(
        self,
        anchor_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = True,
    ) -> dict:
        """Teacher-forced forward over N blocks per sequence.

        Args:
            anchor_ids: [B, N] anchor token ids (x_p)
            anchor_positions: [B, N] anchor positions p
            target_hidden_states: [B, L_ctx, num_target_layers * d] concatenated target features
            prev_token_ids: [B, N, block_size] ground-truth previous tokens x_{p+k}
                (slot 0 = anchor) for the sequential and confidence heads
            context_mask: [B, L_ctx] 1 for real (non-pad) context tokens
            return_confidence: Whether to compute confidence scores.

        Returns:
            dict with keys:
                - base_logits: [B, N, block_size, V] raw backbone logits
                - hidden_states: [B, N, block_size, d] backbone hidden states
                - draft_logits: [B, N, block_size, V] sequential-refined logits
                - confidence_logits: [B, N, block_size] confidence logits (if enabled)
        """
        hidden = self._forward_backbone(anchor_ids, anchor_positions, target_hidden_states, context_mask)
        base_logits = self.lm_head(hidden)

        if self.sequential_head is not None and prev_token_ids is not None:
            draft_logits = self.sequential_head.apply_block_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden,
            )
        else:
            draft_logits = base_logits

        result = {
            "base_logits": base_logits,
            "hidden_states": hidden,
            "draft_logits": draft_logits,
        }

        if return_confidence and self.use_confidence_head and prev_token_ids is not None:
            result["confidence_logits"] = self._confidence_logits(hidden, prev_token_ids)

        return result

    @torch.no_grad()
    def sample_block(
        self,
        anchor_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        return_confidence: bool = True,
    ) -> dict:
        """Draft one block per request.

        Args:
            anchor_ids: [batch] anchor/bonus token from previous round
            anchor_positions: [batch] position of the anchor token
            target_hidden_states: [batch, L_ctx, num_target_layers * d] target features for
                positions < anchor position
            context_mask: [batch, L_ctx] 1 for real context tokens
            temperature: sampling temperature (0 = greedy)
            return_confidence: whether to compute confidence scores

        Returns:
            dict with keys:
                - draft_tokens: [batch, block_size] sampled draft tokens
                - draft_logits: [batch, block_size, V] corrected logits
                - confidence: [batch, block_size] confidence scores (if enabled)
        """
        batch_size = anchor_ids.shape[0]
        hidden = self._forward_backbone(
            anchor_ids[:, None], anchor_positions[:, None], target_hidden_states, context_mask
        )[:, 0]
        base_logits = self.lm_head(hidden)

        if self.sequential_head is not None:
            sampled_tokens, corrected_logits = self.sequential_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=anchor_ids,
                hidden_states=hidden,
                temperature=temperature,
            )
        else:
            if temperature > 0:
                probs = torch.softmax(base_logits.float() / temperature, dim=-1)
                sampled_tokens = torch.multinomial(
                    probs.reshape(-1, self.vocab_size),
                    num_samples=1,
                ).reshape(batch_size, self.block_size)
            else:
                sampled_tokens = base_logits.argmax(dim=-1)
            corrected_logits = base_logits

        result = {
            "draft_tokens": sampled_tokens,
            "draft_logits": corrected_logits,
        }

        if return_confidence and self.use_confidence_head:
            prev_token_ids = torch.cat([anchor_ids.unsqueeze(1), sampled_tokens[:, :-1]], dim=1)
            result["confidence"] = torch.sigmoid(self._confidence_logits(hidden, prev_token_ids))

        return result
