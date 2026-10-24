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

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_draft, D] draft token hidden states
            target_hidden_states: [B, L_ctx, D] target model context features
            attention_mask: [B, 1, L_draft, L_ctx + L_draft] or None

        Returns:
            output: [B, L_draft, D] attended hidden states
        """
        B, L_draft, _ = hidden_states.shape
        L_ctx = target_hidden_states.shape[1]

        # Project Q from draft hidden states
        q = self.q_proj(hidden_states)
        q = q.view(B, L_draft, self.num_heads, self.head_dim).transpose(1, 2)

        # Project K, V from BOTH target context and draft
        k_ctx = self.k_proj(target_hidden_states)
        k_draft = self.k_proj(hidden_states)
        k = torch.cat([k_ctx, k_draft], dim=1)
        k = k.view(B, L_ctx + L_draft, self.num_kv_heads, self.head_dim).transpose(1, 2)

        v_ctx = self.v_proj(target_hidden_states)
        v_draft = self.v_proj(hidden_states)
        v = torch.cat([v_ctx, v_draft], dim=1)
        v = v.view(B, L_ctx + L_draft, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)

        # Handle empty draft (reshape with 0 elements)
        if L_draft == 0:
            return self.o_proj(attn_output.transpose(1, 2).reshape(B, 0, self.num_heads * self.head_dim))

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, L_draft, -1)

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
    ) -> torch.Tensor:
        # Pre-norm attention with context injection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden_states,
            attention_mask,
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
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_draft, D] draft token embeddings
            target_hidden_states: [B, L_ctx, D] target model context features
            attention_mask: optional mask for causal/non-causal attention
            output_hidden_states: if True, return all hidden layer outputs

        Returns:
            hidden_states: [B, L_draft, D] after all backbone layers
        """
        all_hidden = [hidden_states] if output_hidden_states else None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden_states,
                attention_mask,
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
