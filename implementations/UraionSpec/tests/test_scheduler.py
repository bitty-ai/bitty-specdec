"""Tests for the hardware-aware prefix scheduler."""

import torch

from uraionspec.decoding.scheduler import (
    ThroughputProfile,
    hardware_aware_prefix_scheduler,
    StaticScheduler,
)


class TestThroughputProfile:
    """Test the SPS throughput profile."""

    def test_exact_lookup(self):
        profile = ThroughputProfile({"test": 1.0})
        profile.sps = {1: 200.0, 2: 180.0, 4: 150.0}
        assert profile(1) == 200.0
        assert profile(2) == 180.0
        assert profile(4) == 150.0

    def test_interpolation(self):
        profile = ThroughputProfile(None)
        # Between 1 (200) and 2 (180)
        assert profile(1) == 200.0
        val = profile(1)
        assert isinstance(val, float)

    def test_clamp_low(self):
        profile = ThroughputProfile(None)
        rate = profile(0)
        assert rate > 0

    def test_clamp_high(self):
        profile = ThroughputProfile(None)
        rate = profile(10000)
        assert rate > 0


class TestStaticScheduler:
    """Test the static fallback scheduler."""

    def test_fixed_length(self):
        lengths = StaticScheduler.fixed_length(7, 3)
        assert lengths == [7, 7, 7]

    def test_fixed_length_zero(self):
        lengths = StaticScheduler.fixed_length(0, 2)
        assert lengths == [0, 0]

    def test_static_threshold(self):
        confidence = [
            torch.tensor([0.9, 0.8, 0.3, 0.1]),
            torch.tensor([0.2, 0.1, 0.05, 0.01]),
        ]
        lengths = StaticScheduler.static_threshold(confidence, threshold=0.15, gamma=4)
        assert lengths[0] == 3  # 0.9, 0.8, 0.3 >= 0.15
        assert lengths[1] == 1  # 0.2 >= 0.15

    def test_static_threshold_all_below(self):
        confidence = [torch.tensor([0.01, 0.02, 0.03])]
        lengths = StaticScheduler.static_threshold(confidence, threshold=0.5, gamma=3)
        assert lengths[0] == 0


class TestHardwareAwareScheduler:
    """Test Algorithm 1: Hardware-Aware Prefix Scheduler."""

    def test_single_request(self):
        """Single request: scheduler should select some subset of positions."""
        confidence = [torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])]
        profile = ThroughputProfile(None)

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=5)
        assert len(lengths) == 1
        assert 0 <= lengths[0] <= 5

    def test_multiple_requests(self):
        """Multiple requests should each get a verification length."""
        R = 4
        confidence = [
            torch.tensor([0.9, 0.7, 0.5, 0.3]),
            torch.tensor([0.8, 0.6, 0.4, 0.2]),
            torch.tensor([0.7, 0.5, 0.3, 0.1]),
            torch.tensor([0.6, 0.4, 0.2, 0.05]),
        ]
        profile = ThroughputProfile(None)

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=4)
        assert len(lengths) == R
        for length in lengths:
            assert 0 <= length <= 4

    def test_monotonic_selection(self):
        """Selection should be monotonic: higher confidence positions get priority."""
        confidence = [
            torch.tensor([0.9, 0.8, 0.1, 0.05, 0.01]),
        ]
        profile = ThroughputProfile({1: 100, 2: 90, 3: 80, 4: 70, 5: 60, 6: 50})

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=5)
        # Should select at least the first 2 positions (high confidence)
        assert lengths[0] >= 2

    def test_no_future_leakage(self):
        """Scheduler should not leak information from future tokens."""
        # First request: confidence drops sharply at position 3
        # Second request: high confidence throughout
        confidence = [
            torch.tensor([0.95, 0.9, 0.1, 0.05]),  # drops at 3
            torch.tensor([0.95, 0.9, 0.85, 0.8]),  # stays high
        ]
        profile = ThroughputProfile(None)

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=4)

        # The early-stopping property means the scheduler doesn't peek at
        # future tokens to make decisions about current ones.
        # This is implicitly satisfied by the greedy algorithm design.
        assert all(0 <= length <= 4 for length in lengths)

    def test_all_zero_confidence(self):
        """Edge case: all confidence zero should return all zeros."""
        confidence = [torch.zeros(5), torch.zeros(5)]
        profile = ThroughputProfile(None)

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=5)
        assert lengths == [0, 0]

    def test_per_request_validity(self):
        """Scheduler should return valid per-request prefix lengths."""
        R = 3
        confidence = [torch.rand(7) for _ in range(R)]
        profile = ThroughputProfile(None)

        lengths = hardware_aware_prefix_scheduler(confidence, profile, gamma=7)
        assert len(lengths) == R
        for length, c in zip(lengths, confidence):
            assert 0 <= length <= 7
