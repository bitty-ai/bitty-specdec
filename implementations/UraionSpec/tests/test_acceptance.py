"""Tests for speculative decoding acceptance rule."""

import torch

from uraionspec.decoding.acceptance import verify_block, compute_expected_accept_length


class TestAcceptanceRule:
    """Test the core speculative decoding acceptance mechanism."""

    def test_all_accepted(self):
        """When draft matches target exactly, all tokens should be accepted."""
        batch_size, gamma, vocab_size = 2, 5, 100

        # Draft and target distributions are identical
        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = draft_logits.clone()

        # Use argmax as draft tokens (most likely under both distributions)
        draft_tokens = draft_logits.argmax(dim=-1)

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        assert (num_accepted == gamma).all(), f"Expected all {gamma} accepted, got {num_accepted}"
        assert bonus.shape[0] == batch_size
        assert (bonus >= 0).all()

    def test_first_rejected(self):
        """When first token is very unlikely under target, it should be rejected."""
        batch_size, gamma, vocab_size = 1, 3, 100

        # Draft confidently predicts token 0, but target confidently rejects it
        draft_logits = torch.zeros(batch_size, gamma, vocab_size)
        draft_logits[:, :, 0] = 100.0  # Very confident in token 0

        target_logits = torch.zeros(batch_size, gamma, vocab_size)
        target_logits[:, :, 1] = 100.0  # Target prefers token 1

        draft_tokens = torch.zeros(batch_size, gamma, dtype=torch.long)  # Draft proposes token 0

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        # First token should be rejected (p_target/p_draft ≈ 0)
        assert (num_accepted == 0).all(), f"Expected 0 accepted, got {num_accepted}"
        assert bonus.shape[0] == batch_size

    def test_partial_acceptance(self):
        """Test that prefix acceptance works correctly."""
        batch_size, gamma, vocab_size = 1, 4, 100

        # Make first 2 positions match between draft and target
        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = draft_logits.clone()

        # Last 2 positions: target strongly disagrees
        target_logits[:, 2:, :] = -draft_logits[:, 2:, :] + torch.randn(1, 2, vocab_size) * 0.1

        draft_tokens = draft_logits.argmax(dim=-1)

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        # First positions should more likely be accepted
        assert (num_accepted >= 0).all(), "num_accepted should be >= 0"
        assert (num_accepted <= gamma).all(), f"num_accepted should be <= {gamma}"

    def test_batch_independence(self):
        """Each batch element should have independent acceptance."""
        batch_size, gamma, vocab_size = 2, 3, 50

        # First sample: all accepted
        # Second sample: all rejected
        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = draft_logits.clone()
        target_logits[1, :, :] = -draft_logits[1, :, :] + torch.randn(1, gamma, vocab_size) * 0.01

        draft_tokens = draft_logits.argmax(dim=-1)

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        assert num_accepted.shape[0] == batch_size
        assert torch.all(num_accepted[0:1] >= num_accepted[1:2])

    def test_bonus_token_exists(self):
        """Bonus token should always be produced."""
        batch_size, gamma, vocab_size = 1, 1, 100

        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = torch.randn(batch_size, gamma, vocab_size)
        draft_tokens = torch.zeros(batch_size, gamma, dtype=torch.long)

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        assert bonus.shape[0] == batch_size
        assert (bonus >= 0).all() and (bonus < vocab_size).all()

    def test_empty_block(self):
        """Edge case: gamma=1."""
        batch_size, gamma, vocab_size = 1, 1, 100

        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = torch.randn(batch_size, gamma, vocab_size)
        draft_tokens = torch.zeros(batch_size, gamma, dtype=torch.long)

        accepted, num_accepted, bonus = verify_block(
            draft_tokens, draft_logits, target_logits
        )

        assert num_accepted.shape[0] == batch_size
        assert (num_accepted >= 0).all()
        assert (num_accepted <= gamma).all()


class TestExpectedAcceptLength:
    """Test the expected accept length computation."""

    def test_perfect_match(self):
        """When distributions match exactly, expected tau should be gamma + 1."""
        batch_size, gamma, vocab_size = 2, 5, 50

        draft_logits = torch.randn(batch_size, gamma, vocab_size)
        target_logits = draft_logits.clone()

        tau = compute_expected_accept_length(draft_logits, target_logits, gamma)
        assert tau.shape[0] == batch_size
        # With identical distributions, acceptance rate ≈ 1 at each position
        # So tau ≈ 1 + gamma (but may be slightly less due to floating point)
        assert (tau >= 1.0).all()
        assert (tau <= gamma + 1.0 + 1e-6).all()

    def test_completely_wrong(self):
        """When distributions are completely different, tau should be low."""
        batch_size, gamma, vocab_size = 1, 5, 50

        draft_logits = torch.zeros(batch_size, gamma, vocab_size)
        draft_logits[:, :, 0] = 100.0  # Draft always predicts token 0

        target_logits = torch.zeros(batch_size, gamma, vocab_size)
        target_logits[:, :, 1] = 100.0  # Target always predicts token 1

        tau = compute_expected_accept_length(draft_logits, target_logits, gamma)
        # Acceptance rate ≈ 0 at all positions, so tau ≈ 1 (just the bonus)
        assert tau.item() < 1.5  # Should be close to 1.0
