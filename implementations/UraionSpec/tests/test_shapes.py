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

    V, D, G, F_LAYERS = 100, 64, 4, 2

    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return DSparkDraftModel(
            vocab_size=self.V,
            hidden_size=self.D,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_target_layers=self.F_LAYERS,
            block_size=self.G,
            markov_rank=16,
            markov_head_type="vanilla",
            use_confidence_head=True,
        ).eval()

    def _inputs(self, batch=2, n=3, ctx_len=12):
        anchors = torch.randint(1, ctx_len - self.G, (batch, n))
        target_hidden = torch.randn(batch, ctx_len, self.F_LAYERS * self.D)
        prev_ids = torch.randint(0, self.V, (batch, n, self.G))
        return anchors, target_hidden, prev_ids

    def test_forward_shapes(self, model):
        anchors, target_hidden, prev_ids = self._inputs()
        output = model(prev_ids[:, :, 0], anchors, target_hidden, prev_token_ids=prev_ids)

        assert output["base_logits"].shape == (2, 3, self.G, self.V)
        assert output["hidden_states"].shape == (2, 3, self.G, self.D)
        assert output["draft_logits"].shape == (2, 3, self.G, self.V)
        assert output["confidence_logits"].shape == (2, 3, self.G)

    def test_forward_no_confidence(self, model):
        anchors, target_hidden, prev_ids = self._inputs()
        output = model(prev_ids[:, :, 0], anchors, target_hidden, prev_token_ids=prev_ids,
                       return_confidence=False)
        assert "confidence_logits" not in output

    def test_no_future_context_leak(self, model):
        """A block anchored at p must not see target features at positions >= p or other blocks."""
        anchors = torch.tensor([[3, 7]])
        _, target_hidden, prev_ids = self._inputs(batch=1, n=2)
        out = model(prev_ids[:, :, 0], anchors, target_hidden, prev_token_ids=prev_ids)

        perturbed = target_hidden.clone()
        perturbed[:, 3:] += torch.randn_like(perturbed[:, 3:])
        out2 = model(prev_ids[:, :, 0], anchors, perturbed, prev_token_ids=prev_ids)
        assert torch.allclose(out["draft_logits"][:, 0], out2["draft_logits"][:, 0], atol=1e-5)
        assert not torch.allclose(out["draft_logits"][:, 1], out2["draft_logits"][:, 1], atol=1e-5)

    def test_mask_slots_differ(self, model):
        """Positional encoding must make identical mask slots produce different outputs."""
        anchors, target_hidden, prev_ids = self._inputs(batch=1, n=1)
        hidden = model(prev_ids[:, :, 0], anchors, target_hidden)["hidden_states"][0, 0]
        assert not torch.allclose(hidden[1], hidden[2], atol=1e-5)

    def test_sample_block_shapes(self, model):
        _, target_hidden, _ = self._inputs()
        anchor_ids = torch.tensor([10, 20])
        anchor_positions = torch.tensor([5, 8])

        output = model.sample_block(anchor_ids, anchor_positions, target_hidden, temperature=0.0)

        assert output["draft_tokens"].shape == (2, self.G)
        assert output["draft_logits"].shape == (2, self.G, self.V)
        assert output["confidence"].shape == (2, self.G)


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
        V, D, gamma = 50, 32, 4
        model = DSparkDraftModel(
            vocab_size=V, hidden_size=D, num_layers=1,
            num_attention_heads=2, intermediate_size=64,
            num_target_layers=1, block_size=gamma,
            markov_rank=8, markov_head_type="vanilla",
            use_confidence_head=True,
        )

        batch = 2
        anchor_ids = torch.tensor([10, 20])

        # Draft
        draft_out = model.sample_block(
            anchor_ids, torch.tensor([3, 5]), torch.randn(batch, 6, D), temperature=0.0
        )
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
        confidence_targets = compute_accept_rate(draft_logits, target_logits).detach()

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
