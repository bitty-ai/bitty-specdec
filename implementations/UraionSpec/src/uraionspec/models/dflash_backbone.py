"""
DFlash-style parallel backbone with target model KV injection.

This is the core parallel backbone used in DSpark (and DFlash).
Key innovation over standard Transformer decoders:
- Each layer receives target_hidden_states as context
- Attention concatenates target KVs with draft KVs:
    K = [W_K @ target_hidden; W_K @ draft_hidden]
    V = [W_V @ target_hidden; W_V @ draft_hidden]
- Draft tokens attend bidirectionally to context and intra-block draft tokens
- This gives the draft model rich contextual information from the target

Reference: DSpark paper Section 3.1, DFlash (Chen et al., 2026)
"""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [B, L] -> cos, sin: [B, 1, L, head_dim] (fp32)"""
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)


class DFlashAttention(nn.Module):
    """Multi-head attention with target KV injection.

    Concatenates target context KVs with draft KVs so draft tokens
    can attend to target representations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_draft, D] draft token hidden states
            target_hidden_states: [B, L_ctx, D] target model context features
            attention_mask: [B, 1, L_draft, L_ctx + L_draft] bool (True = attend)
                or additive float mask, or None
            position_embeddings: (cos, sin) of shape [B, 1, L_ctx + L_draft, head_dim]
                for the concatenated [context; draft] keys. Queries use the last L_draft.

        Returns:
            output: [B, L_draft, D] attended hidden states
        """
        B, L_draft, _ = hidden_states.shape
        L_ctx = target_hidden_states.shape[1]

        q = self.q_proj(hidden_states).view(B, L_draft, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        kv_input = torch.cat([target_hidden_states.to(hidden_states.dtype), hidden_states], dim=1)
        k = self.k_proj(kv_input).view(B, L_ctx + L_draft, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, L_ctx + L_draft, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = apply_rotary(q, cos[:, :, L_ctx:], sin[:, :, L_ctx:])
            k = apply_rotary(k, cos, sin)

        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(q.dtype)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_kv_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, L_draft, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class DFlashDecoderLayer(nn.Module):
    """Single decoder layer for the DFlash-style backbone.

    Pre-norm architecture: norm → attention → residual → norm → FFN → residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        intermediate_size = intermediate_size or hidden_size * 4

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = DFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            bias=bias,
        )
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        # FFN
        if activation == "gelu":
            act_fn = nn.GELU(approximate="tanh")
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=bias),
            act_fn,
            nn.Linear(intermediate_size, hidden_size, bias=bias),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with context injection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden_states,
            attention_mask,
            position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashBackbone(nn.Module):
    """DFlash-style parallel backbone for DSpark.

    A stack of DFlashDecoderLayers that all receive target_hidden_states
    as context for KV injection.

    This processes all draft positions in a single forward pass
    (parallel, not autoregressive), making drafting latency nearly
    independent of block size γ.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = False,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads or num_attention_heads

        self.layers = nn.ModuleList([
            DFlashDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=self.num_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                activation=activation,
                bias=bias,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = RotaryEmbedding(hidden_size // num_attention_heads, theta=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        context_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_draft, D] draft token embeddings
            target_hidden_states: [B, L_ctx, D] target model context features
            attention_mask: optional mask for causal/non-causal attention
            output_hidden_states: if True, return all hidden layer outputs
            position_ids: [B, L_draft] absolute positions of draft tokens (enables RoPE)
            context_position_ids: [B, L_ctx] absolute positions of context tokens

        Returns:
            hidden_states: [B, L_draft, D] after all backbone layers
        """
        all_hidden = [hidden_states] if output_hidden_states else None

        position_embeddings = None
        if position_ids is not None:
            if context_position_ids is None:
                B, L_ctx = target_hidden_states.shape[:2]
                context_position_ids = torch.arange(L_ctx, device=position_ids.device).expand(B, -1)
            position_embeddings = self.rotary_emb(torch.cat([context_position_ids, position_ids], dim=1))

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden_states,
                attention_mask,
                position_embeddings,
            )
            if output_hidden_states:
                all_hidden.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            return hidden_states, all_hidden
        return hidden_states

# based on blocks we are just making a spec dec attention mask nothing else here !
class DSparkAttentionMask:
    """Build custom attention masks for DSpark training.

    Draft tokens in the same block attend bidirectionally to each other
    and to all context tokens, but NOT to draft tokens in other blocks.
    """

    @staticmethod
    def create_dspark_attention_mask(
        *,
        batch_size: int,
        seq_len: int,
        num_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a block-diagonal attention mask for DSpark.

        Warning: lets every block see all context, which leaks future tokens when the
        context is the full training sequence. Training uses create_block_context_mask.

        Each block's draft tokens can attend to:
        - All context tokens (bidirectional)
        - All other draft tokens in the same block (bidirectional)
        - But NOT to draft tokens in other blocks
        - As in training , this would leak the info as anchors are different for all points here 

        Returns:
            mask: [B, 1, L_draft, L_ctx + L_draft] float mask
        """
        L_ctx = seq_len
        L_draft = num_blocks * block_size
        L_total = L_ctx + L_draft

        # Start with all-to-all
        mask = torch.zeros(batch_size, 1, L_draft, L_total, device=device)

        for b in range(batch_size):
            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = start + block_size

                # Can attend to all context tokens
                mask[b, :, start:end, :L_ctx] = 0.0

                # Can attend to all draft tokens in the same block
                mask[b, :, start:end, L_ctx + start:L_ctx + end] = 0.0

                # Cannot attend to draft tokens in other blocks (already -inf)
                mask[b, :, start:end, L_ctx:L_ctx + start] = float("-inf")
                mask[b, :, start:end, L_ctx + end:L_ctx + L_draft] = float("-inf")

        return mask


    @staticmethod
    def create_block_context_mask(
        *,
        anchor_positions: torch.Tensor,
        block_size: int,
        context_len: int,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Leak-free mask matching inference: the block anchored at p sees target
        context j < p (the anchor itself is not yet consumed by the target) and its
        own block bidirectionally, never other blocks.

        Args:
            anchor_positions: [B, N]
            context_mask: [B, L_ctx] 1 for real tokens

        Returns:
            mask: [B, 1, N * block_size, L_ctx + N * block_size] bool (True = attend)
        """
        B, N = anchor_positions.shape
        device = anchor_positions.device
        q_block = torch.arange(N, device=device).repeat_interleave(block_size)
        q_anchor = anchor_positions[:, q_block]
        ctx_ok = torch.arange(context_len, device=device)[None, None, :] < q_anchor[:, :, None]
        if context_mask is not None:
            ctx_ok = ctx_ok & context_mask[:, None, :].bool()
        blk_ok = (q_block[:, None] == q_block[None, :]).expand(B, -1, -1)
        return torch.cat([ctx_ok, blk_ok], dim=-1)[:, None]
