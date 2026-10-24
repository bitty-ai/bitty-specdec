"""Tests for Sequential Temperature Scaling (STS) calibration."""

import torch
import pytest

from uraionspec.calibration.sts import (
    expected_calibration_error,
    sequential_temperature_scaling,
    STSCalibrator,
)


class TestExpectedCalibrationError:
    """Test ECE computation."""

    def test_perfectly_calibrated(self):
        probs = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
        ece = expected_calibration_error(probs, labels, n_bins=5)
        assert ece >= 0

    def test_miscalibrated(self):
        probs = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])
        labels = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        ece = expected_calibration_error(probs, labels, n_bins=3)
        assert ece > 0

    def test_shape(self):
        probs = torch.rand(100)
        labels = (torch.rand(100) > 0.7).float()
        ece = expected_calibration_error(probs, labels)
        assert ece >= 0
        assert ece <= 1.0


class TestSequentialTemperatureScaling:
    """Test STS algorithm."""

    def test_sts_returns_temperatures(self):
        N, gamma = 100, 4
        scores = torch.sigmoid(torch.randn(N, gamma))
        labels = (torch.rand(N, gamma) < scores).float()

        temps = sequential_temperature_scaling(
            scores, labels, block_size=gamma, n_grid=20, device="cpu"
        )
        assert len(temps) == gamma
        for t in temps:
            assert t > 0

    def test_sts_monotonic(self):
        """Temperatures should generally be meaningful."""
        N, gamma = 198, 3  # Multiple of 3
        scores = torch.tensor([
            [0.95, 0.90, 0.85],
            [0.92, 0.88, 0.82],
            [0.98, 0.95, 0.90],
        ]).repeat(N // 3, 1)
        labels = (torch.rand(N, gamma) < scores).float()

        temps = sequential_temperature_scaling(
            scores, labels, block_size=gamma, n_grid=50, device="cpu"
        )
        assert all(t > 0 for t in temps)


class TestSTSCalibrator:
    """Test the STSCalibrator class."""

    @pytest.fixture
    def calibrator(self):
        return STSCalibrator(n_grid=20, n_bins=5)

    def test_fit_transform(self, calibrator):
        N, gamma = 100, 4
        scores = torch.sigmoid(torch.randn(N, gamma))
        labels = (torch.rand(N, gamma) < scores).float()

        calibrated = calibrator.fit_transform(scores, labels, device="cpu")
        assert calibrated.shape == (N, gamma)
        assert calibrator.fitted
        assert len(calibrator.temperatures) == gamma

    def test_transform_unfitted(self):
        calibrator = STSCalibrator()
        with pytest.raises(AssertionError):
            calibrator.transform(torch.randn(10, 4))

    def test_shape_preserved(self, calibrator):
        N, gamma = 100, 4
        scores = torch.sigmoid(torch.randn(N, gamma))
        labels = (torch.rand(N, gamma) < scores).float()

        calibrated = calibrator.fit_transform(scores, labels, device="cpu")
        assert calibrated.shape == scores.shape
        # STS is order-preserving
        assert (calibrated.argsort(dim=0) == scores.argsort(dim=0)).all()
