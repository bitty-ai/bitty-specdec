"""Tests for sampling utilities."""

import torch
import pytest

from uraionspec.utils.sampling import (
    logits_to_probs,
    sample_tokens,
    sample_residual,
    gather_token_probs,
)


class TestSampling:
    """Test sampling utilities."""

    def test_logits_to_probs_greedy(self):
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        probs = logits_to_probs(logits, temperature=0.0)
        assert probs.shape == (1, 3)
        assert probs[0, 1] == 1.0  # argmax at index 1

    def test_logits_to_probs_temperature(self):
        logits = torch.randn(2, 100)
        probs = logits_to_probs(logits, temperature=1.0)
        assert probs.shape == (2, 100)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

    def test_sample_tokens_greedy(self):
        logits = torch.randn(2, 5, 50)
        tokens = sample_tokens(logits, temperature=0.0)
        assert tokens.shape == (2, 5)
        assert (tokens >= 0).all() and (tokens < 50).all()

    def test_sample_tokens_temperature(self):
        logits = torch.randn(2, 50)
        tokens = sample_tokens(logits, temperature=1.0)
        assert tokens.shape == (2,)
        assert (tokens >= 0).all() and (tokens < 50).all()

    def test_sample_tokens_2d(self):
        logits = torch.randn(3, 100)
        tokens = sample_tokens(logits, temperature=0.5)
        assert tokens.shape == (3,)

    def test_sample_residual(self):
        target = torch.softmax(torch.randn(2, 50) + 2, dim=-1)
        draft = torch.softmax(torch.randn(2, 50), dim=-1)
        tokens = sample_residual(target, draft)
        assert tokens.shape == (2,)
        assert (tokens >= 0).all() and (tokens < 50).all()

    def test_sample_residual_identical(self):
        """When target == draft, residual should fall back to target."""
        probs = torch.softmax(torch.randn(2, 50), dim=-1)
        tokens = sample_residual(probs, probs)
        assert tokens.shape == (2,)

    def test_gather_token_probs(self):
        probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        token_ids = torch.tensor([1, 2])
        gathered = gather_token_probs(probs, token_ids)
        assert gathered.shape == (2,)
        assert gathered[0].item() == pytest.approx(0.7)
        assert gathered[1].item() == pytest.approx(0.4)
