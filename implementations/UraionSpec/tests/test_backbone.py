"""Tests for the DFlash-style parallel backbone with KV injection."""

import torch
import pytest

from uraionspec.models.dflash_backbone import (
    DFlashBackbone,
    DFlashDecoderLayer,
    DFlashAttention,
    DSparkAttentionMask,
)


class TestDFlashAttention:
    """Test the target KV injection attention."""

    @pytest.fixture
    def attn(self):
        return DFlashAttention(hidden_size=64, num_heads=4, dropout=0.0)

    def test_forward_shape(self, attn):
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        out = attn(hidden, target)
        assert out.shape == (B, L_draft, D)

    def test_gqa_forward(self):
        """Test grouped query attention."""
        attn = DFlashAttention(hidden_size=64, num_heads=4, num_kv_heads=2)
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        out = attn(hidden, target)
        assert out.shape == (B, L_draft, D)

    def test_with_mask(self, attn):
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        # Causal mask
        mask = torch.zeros(B, 1, L_draft, L_ctx + L_draft)
        mask[:, :, :, L_ctx:] = torch.triu(
            torch.full((L_draft, L_draft), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        out = attn(hidden, target, attention_mask=mask)
        assert out.shape == (B, L_draft, D)

    def test_gradient_flow(self, attn):
        B, L_draft, L_ctx, D = 1, 2, 4, 64
        hidden = torch.randn(B, L_draft, D, requires_grad=True)
        target = torch.randn(B, L_ctx, D)

        out = attn(hidden, target)
        loss = out.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == (B, L_draft, D)


class TestDFlashDecoderLayer:
    """Test a single DFlash decoder layer."""

    @pytest.fixture
    def layer(self):
        return DFlashDecoderLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            dropout=0.0,
        )

    def test_forward_shape(self, layer):
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        out = layer(hidden, target)
        assert out.shape == (B, L_draft, D)

    def test_residual_connection(self, layer):
        """Output should differ from input (non-identity transformation)."""
        B, L_draft, L_ctx, D = 1, 2, 4, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        with torch.no_grad():
            out = layer(hidden, target)
        assert not torch.allclose(out, hidden, atol=1e-4)

    def test_all_activations(self):
        for act in ["gelu", "relu", "silu"]:
            layer = DFlashDecoderLayer(
                hidden_size=32, num_heads=2, intermediate_size=64,
                dropout=0.0, activation=act,
            )
            B, L_draft, L_ctx = 1, 2, 4
            hidden = torch.randn(B, L_draft, 32)
            target = torch.randn(B, L_ctx, 32)
            out = layer(hidden, target)
            assert out.shape == (B, L_draft, 32)


class TestDFlashBackbone:
    """Test the full DFlash backbone stack."""

    @pytest.fixture
    def backbone(self):
        return DFlashBackbone(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            dropout=0.0,
        )

    def test_forward_shape(self, backbone):
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        out = backbone(hidden, target)
        assert out.shape == (B, L_draft, D)

    def test_output_hidden_states(self, backbone):
        B, L_draft, L_ctx, D = 2, 4, 8, 64
        hidden = torch.randn(B, L_draft, D)
        target = torch.randn(B, L_ctx, D)

        out, all_hidden = backbone(hidden, target, output_hidden_states=True)
        assert len(all_hidden) == 3  # input + 2 layers
        for h in all_hidden:
            assert h.shape == (B, L_draft, D)

    def test_gradient_flow(self, backbone):
        B, L_draft, L_ctx, D = 1, 3, 6, 64
        hidden = torch.randn(B, L_draft, D, requires_grad=True)
        target = torch.randn(B, L_ctx, D)

        out = backbone(hidden, target)
        loss = out.sum()
        loss.backward()
        assert hidden.grad is not None

    def test_empty_draft(self, backbone):
        """Edge case: no draft tokens."""
        B, L_ctx, D = 2, 8, 64
        hidden = torch.randn(B, 0, D)
        target = torch.randn(B, L_ctx, D)

        out = backbone(hidden, target)
        assert out.shape == (B, 0, D)

    def test_single_draft_token(self, backbone):
        """Edge case: single draft token."""
        B, L_ctx, D = 2, 8, 64
        hidden = torch.randn(B, 1, D)
        target = torch.randn(B, L_ctx, D)

        out = backbone(hidden, target)
        assert out.shape == (B, 1, D)

    def test_many_layers(self):
        """Test with more layers."""
        backbone = DFlashBackbone(
            hidden_size=32, num_layers=6, num_attention_heads=4,
        )
        B, L_draft, L_ctx = 2, 4, 8
        hidden = torch.randn(B, L_draft, 32)
        target = torch.randn(B, L_ctx, 32)

        out = backbone(hidden, target)
        assert out.shape == (B, L_draft, 32)


class TestDSparkAttentionMask:
    """Test the custom DSpark attention mask builder."""

    def test_mask_shape(self):
        B, seq_len = 2, 10
        num_blocks, block_size = 3, 4
        device = "cpu"

        mask = DSparkAttentionMask.create_dspark_attention_mask(
            batch_size=B,
            seq_len=seq_len,
            num_blocks=num_blocks,
            block_size=block_size,
            device=torch.device(device),
        )
        L_draft = num_blocks * block_size
        assert mask.shape == (B, 1, L_draft, seq_len + L_draft)

    def test_context_attention(self):
        """Draft tokens should be able to attend to all context tokens."""
        B, seq_len = 1, 5
        num_blocks, block_size = 2, 3
        device = "cpu"

        mask = DSparkAttentionMask.create_dspark_attention_mask(
            batch_size=B, seq_len=seq_len,
            num_blocks=num_blocks, block_size=block_size,
            device=torch.device(device),
        )
        # All draft positions should have 0.0 for all context positions
        context_slice = mask[0, 0, :, :seq_len]
        assert (context_slice == 0.0).all()

    def test_intra_block_attention(self):
        """Draft tokens in the same block should attend to each other."""
        B, seq_len = 1, 5
        num_blocks, block_size = 2, 3
        device = "cpu"

        mask = DSparkAttentionMask.create_dspark_attention_mask(
            batch_size=B, seq_len=seq_len,
            num_blocks=num_blocks, block_size=block_size,
            device=torch.device(device),
        )
        L_ctx = seq_len

        # Block 0: positions 0,1,2 should attend to each other
        intra_block_0 = mask[0, 0, 0:3, L_ctx:L_ctx+3]
        assert (intra_block_0 == 0.0).all(), "Block 0 intra-attention should be 0"

        # Block 1: positions 3,4,5 should attend to each other
        intra_block_1 = mask[0, 0, 3:6, L_ctx+3:L_ctx+6]
        assert (intra_block_1 == 0.0).all(), "Block 1 intra-attention should be 0"

    def test_cross_block_no_attention(self):
        """Draft tokens should NOT attend to draft tokens in other blocks."""
        B, seq_len = 1, 5
        num_blocks, block_size = 2, 3
        device = "cpu"

        mask = DSparkAttentionMask.create_dspark_attention_mask(
            batch_size=B, seq_len=seq_len,
            num_blocks=num_blocks, block_size=block_size,
            device=torch.device(device),
        )
        L_ctx = seq_len

        # Block 0 should NOT attend to Block 1's draft tokens
        cross_block = mask[0, 0, 0:3, L_ctx+3:L_ctx+6]
        assert (cross_block == float("-inf")).all(), \
            "Cross-block attention should be -inf"

        # Block 1 should NOT attend to Block 0's draft tokens
        cross_block_2 = mask[0, 0, 3:6, L_ctx:L_ctx+3]
        assert (cross_block_2 == float("-inf")).all(), \
            "Cross-block attention should be -inf"
