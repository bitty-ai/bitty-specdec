"""
Sequential Temperature Scaling (STS) for DSpark confidence calibration.

Calibrates cumulative prefix survival probabilities left-to-right.
At each position k:
    Find optimal temperature T_k that minimizes ECE of cumprod(c_1..c_k)
    while keeping already-calibrated scores at positions < k fixed.

Key insight: Temperature scaling is order-preserving, so it fixes
probability magnitude without disrupting relative rankings.

Reference: DSpark paper Section 3.2.1 (Post-hoc Calibration).
"""

from typing import Optional

import torch


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Compute Expected Calibration Error (ECE).

    Bins probabilities into n_bins equal-width bins and computes
    |accuracy - confidence| per bin, weighted by bin size.

    Args:
        probs: [N] predicted probabilities in [0, 1]
        labels: [N] binary targets (0 or 1)
        n_bins: Number of equal-width bins

    Returns:
        ece: Scalar ECE value
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=probs.device, dtype=torch.float32)

    for i in range(n_bins):
        in_bin = (probs > bin_lowers[i]) & (probs <= bin_uppers[i])
        bin_count = in_bin.sum().float()

        if bin_count > 0:
            bin_accuracy = labels[in_bin].mean()
            bin_confidence = probs[in_bin].mean()
            ece += (bin_confidence - bin_accuracy).abs() * (bin_count / probs.shape[0])

    return ece


def sequential_temperature_scaling(
    confidence_scores: torch.Tensor,
    acceptance_labels: torch.Tensor,
    block_size: int,
    temp_range: tuple[float, float] = (0.1, 10.0),
    n_grid: int = 100,
    n_bins: int = 15,
    device: str = "cuda",
) -> list[float]:
    """Sequential Temperature Scaling (STS) calibration.

    Calibrates cumulative prefix survival probabilities left-to-right.
    For each position k:
        1. Compute cumulative product of calibrated scores up to k
        2. Find optimal temperature T_k via grid search minimizing ECE
        3. Apply T_k to the k-th position score

    Args:
        confidence_scores: [N, block_size] raw confidence scores (pre-calibration)
        acceptance_labels: [N, block_size] binary acceptance labels (0/1)
        block_size: Number of draft positions
        temp_range: (min_temp, max_temp) for grid search
        n_grid: Number of grid points
        n_bins: Number of bins for ECE computation
        device: Computation device

    Returns:
        temperatures: [block_size] optimal temperatures per position
    """
    N = confidence_scores.shape[0]
    temperatures = []

    # Process positions left to right
    for k in range(block_size):
        # Compute cumulative survival probabilities up to position k
        # using previously calibrated temperatures for positions < k
        cumprod = torch.ones(N, device=device, dtype=torch.float32)

        for j in range(k + 1):
            if j < len(temperatures):
                # Apply previously calibrated temperature
                temp = temperatures[j]
                calibrated = confidence_scores[:, j] ** (1.0 / temp)
            else:
                # Use raw score for current position
                calibrated = confidence_scores[:, k]

            cumprod = cumprod * calibrated

        # Labels: cumulative acceptance at position k
        cum_labels = acceptance_labels[:, :k + 1].all(dim=-1).float()

        # Grid search for optimal temperature
        temps = torch.linspace(temp_range[0], temp_range[1], n_grid, device=device)
        best_ece = float("inf")
        best_temp = 1.0

        for temp in temps:
            # Apply temperature: calibrated = raw^(1/T)
            # For cumulative product, we only need to calibrate position k
            if k < len(temperatures):
                # Position already calibrated; cumprod is already fixed
                # But the position-specific calibration only applies to position k
                # which is too tightly coupled to the cumprod... In practice,
                # we apply temp to position k cumprod contribution
                pass

            # Simplified: calibrate by scaling the cumprod directly
            # This matches the per-position scaling described in the paper
            # where T_k is found by ECE of cumprod after applying T_k to position k
            if len(temperatures) == k:
                calibrated_k = confidence_scores[:, k] ** (1.0 / temp)
                new_cumprod = cumprod * calibrated_k
            else:
                new_cumprod = cumprod  # already calibrated

            ece = expected_calibration_error(new_cumprod, cum_labels, n_bins).item()
            if ece < best_ece:
                best_ece = ece
                best_temp = temp.item()

        temperatures.append(best_temp)

    return temperatures


class STSCalibrator:
    """Sequential Temperature Scaling calibrator.

    Fits temperatures on a calibration set and applies them to new data.
    """

    def __init__(
        self,
        temp_range: tuple[float, float] = (0.1, 10.0),
        n_grid: int = 100,
        n_bins: int = 15,
    ):
        self.temp_range = temp_range
        self.n_grid = n_grid
        self.n_bins = n_bins
        self.temperatures: list[float] = []
        self.block_size: int = 0
        self.fitted = False

    def fit(
        self,
        confidence_scores: torch.Tensor,
        acceptance_labels: torch.Tensor,
        block_size: Optional[int] = None,
        device: str = "cuda",
    ) -> "STSCalibrator":
        """Fit STS temperatures on calibration data.

        Args:
            confidence_scores: [N, block_size] raw confidence scores
            acceptance_labels: [N, block_size] binary acceptance labels
            block_size: Block size (inferred from data if None)
            device: Computation device

        Returns:
            self for chaining.
        """
        N, bs = confidence_scores.shape
        self.block_size = block_size or bs

        self.temperatures = sequential_temperature_scaling(
            confidence_scores=confidence_scores.to(device),
            acceptance_labels=acceptance_labels.to(device),
            block_size=self.block_size,
            temp_range=self.temp_range,
            n_grid=self.n_grid,
            n_bins=self.n_bins,
            device=device,
        )
        self.fitted = True
        return self

    def transform(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        """Apply STS calibration to new confidence scores.

        Args:
            confidence_scores: [N, block_size] raw scores

        Returns:
            calibrated_scores: [N, block_size] calibrated scores
        """
        assert self.fitted, "Calibrator not fitted yet"
        N, bs = confidence_scores.shape
        assert bs == self.block_size, (
            f"Expected block_size={self.block_size}, got {bs}"
        )

        calibrated = confidence_scores.clone()
        for k in range(self.block_size):
            temp = self.temperatures[k]
            calibrated[:, k] = confidence_scores[:, k] ** (1.0 / temp)

        return calibrated

    def fit_transform(
        self,
        confidence_scores: torch.Tensor,
        acceptance_labels: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Fit and apply in one call."""
        self.fit(confidence_scores, acceptance_labels, device=device)
        return self.transform(confidence_scores)
