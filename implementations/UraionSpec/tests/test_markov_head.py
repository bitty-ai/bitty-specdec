"""Tests for DSpark Markov head implementations."""

import torch
import pytest

from uraionspec.models.markov_head import VanillaMarkov, GatedMarkovHead, build_markov_head
from uraionspec.models.rnn_head import RNNHead


class TestVanillaMarkov:
    """Test the vanilla Markov head."""

    @pytest.fixture
    def head(self):
        return VanillaMarkov(vocab_size=100, markov_rank=32)

    def test_init(self, head):
        assert head.vocab_size == 100
        assert head.markov_rank == 32
        assert head.markov_w1.weight.shape == (100, 32)
        assert head.markov_w2.weight.shape == (100, 32)  # Linear: (out_features, in_features)

    def test_get_prev_embeddings(self, head):
        token_ids = torch.tensor([5, 10, 15])
        emb = head.get_prev_embeddings(token_ids)
        assert emb.shape == (3, 32)

    def test_compute_step_bias(self, head):
        token_ids = torch.tensor([5])
        bias = head.compute_step_bias(token_ids)
        assert bias.shape == (1, 100)

    def test_apply_step_logits(self, head):
        logits = torch.randn(1, 100)
        token_ids = torch.tensor([5])
        result = head.apply_step_logits(logits, token_ids=token_ids)
        assert result.shape == (1, 100)
        assert not torch.allclose(result, logits)

    def test_apply_block_logits(self, head):
        batch, num_blocks, block_size, V = 2, 3, 4, 100
        base_logits = torch.randn(batch, num_blocks, block_size, V)
        token_ids = torch.randint(0, 100, (batch, num_blocks, block_size))
        result = head.apply_block_logits(base_logits, token_ids=token_ids)
        assert result.shape == (batch, num_blocks, block_size, V)

    def test_sample_block_tokens_greedy(self, head):
        batch_size, gamma = 2, 4
        base_logits = torch.randn(batch_size, gamma, 100)
        first_prev = torch.tensor([10, 20])

        tokens, logits = head.sample_block_tokens(
            base_logits, first_prev_token_ids=first_prev, temperature=0.0
        )
        assert tokens.shape == (batch_size, gamma)
        assert logits.shape == (batch_size, gamma, 100)

    def test_sample_block_tokens_empty(self, head):
        tokens, logits = head.sample_block_tokens(
            torch.randn(2, 0, 100),
            first_prev_token_ids=torch.tensor([1, 2]),
        )
        assert tokens.shape == (2, 0)

    def test_build_vanilla(self):
        class Config:
            markov_rank = 64
            markov_head_type = "vanilla"
            vocab_size = 100
            hidden_size = 256

        head = build_markov_head(Config())
        assert isinstance(head, VanillaMarkov)
        assert head.markov_rank == 64

    def test_build_gated(self):
        class Config:
            markov_rank = 64
            markov_head_type = "gated"
            vocab_size = 100
            hidden_size = 256

        head = build_markov_head(Config())
        assert isinstance(head, GatedMarkovHead)
        assert head.markov_rank == 64


class TestGatedMarkovHead:
    """Test the gated Markov head variant."""

    @pytest.fixture
    def head(self):
        return GatedMarkovHead(vocab_size=100, markov_rank=32, hidden_size=256)

    def test_init(self, head):
        assert head.markov_head_type == "gated"
        assert head.gate_proj.weight.shape == (32, 256 + 32)

    def test_forward_shape(self, head):
        batch, gamma = 2, 4
        base_logits = torch.randn(batch, gamma, 100)
        token_ids = torch.randint(0, 100, (batch, gamma))
        hidden = torch.randn(batch, gamma, 256)

        result = head.apply_block_logits(base_logits, token_ids=token_ids, hidden_states=hidden)
        assert result.shape == (batch, gamma, 100)


class TestRNNHead:
    """Test the RNN sequential head."""

    @pytest.fixture
    def head(self):
        return RNNHead(vocab_size=100, markov_rank=32, hidden_size=256)

    def test_init(self, head):
        assert head.markov_head_type == "rnn"
        # joint_proj: 2*r + d -> 3*r = 64+256 -> 96
        assert head.joint_proj.weight.shape[0] == 3 * 32

    def test_apply_block_logits(self, head):
        batch, num_blocks, block_size, V = 2, 3, 4, 100
        base_logits = torch.randn(batch, num_blocks, block_size, V)
        token_ids = torch.randint(0, 100, (batch, num_blocks, block_size))
        hidden = torch.randn(batch, num_blocks, block_size, 256)

        result = head.apply_block_logits(base_logits, token_ids=token_ids, hidden_states=hidden)
        assert result.shape == (batch, num_blocks, block_size, V)

    def test_step(self, head):
        batch = 2
        state = torch.zeros(batch, 32)
        prev = torch.tensor([10, 20])
        h_k = torch.randn(batch, 256)
        base = torch.randn(batch, 100)

        new_state, corrected = head.step(state, prev, h_k, base)
        assert new_state.shape == (batch, 32)
        assert corrected.shape == (batch, 100)

    def test_empty_block(self, head):
        result = head.apply_block_logits(
            torch.randn(2, 0, 100),
            token_ids=torch.zeros(2, 0, dtype=torch.long),
            hidden_states=torch.randn(2, 0, 256),
        )
        assert result.shape == (2, 0, 100)
