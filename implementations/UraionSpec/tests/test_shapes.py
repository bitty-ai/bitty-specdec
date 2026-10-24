"""Shape and integration tests for all modules."""

import torch
import pytest

from uraionspec.models import (
    DSparkDraftModel, ConfidenceHead, compute_accept_rate,
)
from uraionspec.decoding import (
    verify_block, ThroughputProfile, hardware_aware_prefix_scheduler,
)
from uraionspec.training import compute_dspark_loss, build_position_weights


class TestModelShapes:
    """Test tensor shapes through all model components."""

    @pytest.fixture
    def model(self):
        return DSparkDraftModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            markov_rank=16,
            markov_head_type="vanilla",
            use_confidence_head=True,
        )

    def test_forward_shapes(self, model):
        batch, block_size = 2, 4
        input_ids = torch.randint(0, 100, (batch, block_size))
        prev_ids = torch.randint(0, 100, (batch, block_size))

        output = model(input_ids, prev_token_ids=prev_ids, return_confidence=True)

        assert output["base_logits"].shape == (batch, block_size, 100)
        assert output["hidden_states"].shape == (batch, block_size, 64)
        assert output["draft_logits"].shape == (batch, block_size, 100)
        assert output["confidence_logits"].shape == (batch, block_size)

    def test_forward_no_confidence(self, model):
        model.use_confidence_head = False
        batch, block_size = 2, 4
        input_ids = torch.randint(0, 100, (batch, block_size))

        output = model(input_ids, return_confidence=False)
        assert "confidence_logits" not in output

    def test_sample_block_shapes(self, model):
        batch, gamma = 2, 4
        anchor_ids = torch.tensor([10, 20])

        output = model.sample_block(anchor_ids, gamma=gamma, temperature=0.0)

        assert output["draft_tokens"].shape == (batch, gamma)
        assert output["draft_logits"].shape == (batch, gamma, 100)
        assert output["confidence"].shape == (batch, gamma)


class TestLossShapes:
    """Test loss computation shapes."""

    def test_loss_shapes(self):
        B, N, block_size, V = 2, 3, 4, 100
        draft_logits = torch.randn(B, N, block_size, V, requires_grad=True)
        target_ids = torch.randint(0, V, (B, N, block_size))
        eval_mask = torch.ones(B, N, block_size, dtype=torch.bool)

        result = compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            ce_alpha=0.1,
            tv_alpha=0.0,
            conf_alpha=0.0,
        )

        loss = result["loss"]
        loss.backward()
        assert loss.item() > 0
        assert draft_logits.grad is not None
        assert draft_logits.grad.shape == (B, N, block_size, V)

    def test_loss_with_all_terms(self):
        B, N, block_size, V = 2, 3, 4, 100
        draft_logits = torch.randn(B, N, block_size, V, requires_grad=True)
        target_ids = torch.randint(0, V, (B, N, block_size))
        target_logits = torch.randn(B, N, block_size, V)
        eval_mask = torch.ones(B, N, block_size, dtype=torch.bool)
        confidence_pred = torch.randn(B, N, block_size, requires_grad=True)
        confidence_targets = torch.sigmoid(torch.randn(B, N, block_size))

        result = compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            target_logits=target_logits,
            confidence_pred=confidence_pred,
            confidence_targets=confidence_targets,
            ce_alpha=0.1,
            tv_alpha=0.9,
            conf_alpha=1.0,
        )

        result["loss"].backward()
        assert result["loss"].item() > 0

    def test_loss_decay_weights(self):
        weights = build_position_weights(7, 7.0, "cpu")
        assert weights.shape == (7,)
        assert weights[0] > weights[-1]  # Earlier positions have higher weight
        assert (weights > 0).all()


class TestConfidenceHeadShapes:
    """Test confidence head shapes."""

    def test_confidence_head(self):
        batch, gamma, d, r = 2, 4, 64, 16
        head = ConfidenceHead(hidden_size=d, markov_rank=r)

        hidden = torch.randn(batch, gamma, d)
        prev_emb = torch.randn(batch, gamma, r)

        logits = head(hidden, prev_emb)
        assert logits.shape == (batch, gamma)

        probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_accept_rate_shape(self):
        batch, gamma, V = 2, 4, 100
        draft_logits = torch.randn(batch, gamma, V)
        target_logits = torch.randn(batch, gamma, V)

        rate = compute_accept_rate(draft_logits, target_logits)
        assert rate.shape == (batch, gamma)
        assert (rate >= 0).all() and (rate <= 1).all()


class TestIntegration:
    """End-to-end integration tests."""

    def test_draft_verify_cycle(self):
        """Test draft -> verify -> accept cycle end-to-end."""
        V = 50
        model = DSparkDraftModel(
            vocab_size=V, hidden_size=32, num_layers=1,
            num_attention_heads=2, intermediate_size=64,
            markov_rank=8, markov_head_type="vanilla",
            use_confidence_head=True,
        )

        batch, gamma = 2, 4
        anchor_ids = torch.tensor([10, 20])

        # Draft
        draft_out = model.sample_block(anchor_ids, gamma=gamma, temperature=0.0)
        assert draft_out["draft_tokens"].shape == (batch, gamma)
        assert draft_out["draft_logits"].shape == (batch, gamma, V)
        assert draft_out["confidence"].shape == (batch, gamma)

        # Mock target logits (just noise for testing)
        target_logits = torch.randn(batch, gamma, V)

        # Accept
        accepted, num_accepted, bonus = verify_block(
            draft_out["draft_tokens"],
            draft_out["draft_logits"],
            target_logits,
        )
        assert num_accepted.shape == (batch,)
        assert bonus.shape == (batch,)

        # Scheduler
        profile = ThroughputProfile(None)
        conf_list = [draft_out["confidence"][b] for b in range(batch)]
        lengths = hardware_aware_prefix_scheduler(conf_list, profile, gamma)
        assert len(lengths) == batch
        assert all(0 <= length <= gamma for length in lengths)

    def test_train_step_shape(self):
        """Test that all components in a training step have correct shapes."""
        V = 50
        B, N, bs = 2, 2, 4

        draft_logits = torch.randn(B, N, bs, V, requires_grad=True)
        target_ids = torch.randint(0, V, (B, N, bs))
        eval_mask = torch.ones(B, N, bs, dtype=torch.bool)
        target_logits = torch.randn(B, N, bs, V)

        # Confidence targets
        confidence_pred = torch.randn(B, N, bs, requires_grad=True)
        confidence_targets = torch.sigmoid(
            compute_accept_rate(draft_logits, target_logits).detach()
        )

        loss_dict = compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            target_logits=target_logits,
            confidence_pred=confidence_pred,
            confidence_targets=confidence_targets,
            ce_alpha=0.1,
            tv_alpha=0.9,
            conf_alpha=1.0,
        )

        loss_dict["loss"].backward()
        assert draft_logits.grad is not None
        assert confidence_pred.grad is not None
