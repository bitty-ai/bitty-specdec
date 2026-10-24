"""
Hardware-Aware Prefix Scheduler (Algorithm 1 from DSpark paper).

Dynamically selects per-request verification lengths to maximize
expected system throughput: Θ = τ * SPS(B)

where τ is expected accepted tokens, SPS(B) is steps-per-second at batch size B.

Key properties:
- Monotonic candidate selection via global survival probability sorting
- Early stopping to preserve non-anticipating property
- Pre-profiled throughput curve (SPS table)
- Fallback to static threshold or fixed length
"""

from typing import Optional

import torch


class ThroughputProfile:
    """Pre-profiled engine throughput curve SPS(B).

    Maps batch size B (tokens per forward pass) to steps-per-second.
    Profiled once during engine initialization.
    """

    def __init__(self, sps_table: Optional[dict[int, float]] = None):
        """Initialize with an SPS lookup table.

        Args:
            sps_table: dict mapping batch_size -> steps_per_second.
                If None, uses a default synthetic profile.
        """
        if sps_table is not None:
            self.sps = sps_table
        else:
            # Default synthetic SPS curve: realistic-ish for an A100
            # Starts high for small batches, decreases as batch grows
            self.sps = {
                1: 200.0,
                2: 180.0,
                4: 150.0,
                8: 120.0,
                16: 90.0,
                32: 60.0,
                64: 35.0,
                128: 20.0,
                256: 10.0,
                512: 5.0,
            }

    def __call__(self, batch_size: int) -> float:
        """Lookup SPS for a given batch size, with interpolation.

        Args:
            batch_size: Number of tokens to verify.

        Returns:
            steps_per_second: Estimated throughput.
        """
        batch_sizes = sorted(self.sps.keys())

        # Exact match
        if batch_size in self.sps:
            return self.sps[batch_size]

        # Clamp to range
        if batch_size <= batch_sizes[0]:
            return self.sps[batch_sizes[0]]
        if batch_size >= batch_sizes[-1]:
            return self.sps[batch_sizes[-1]]

        # Linear interpolation
        for i in range(len(batch_sizes) - 1):
            b_low, b_high = batch_sizes[i], batch_sizes[i + 1]
            if b_low <= batch_size <= b_high:
                frac = (batch_size - b_low) / (b_high - b_low)
                return self.sps[b_low] + frac * (self.sps[b_high] - self.sps[b_low])

        return self.sps[batch_sizes[0]]


def hardware_aware_prefix_scheduler(
    confidence_scores: list[torch.Tensor],
    throughput_profile: ThroughputProfile,
    gamma: int,
    min_accept_prob: float = 1e-6,
) -> list[int]:
    """Hardware-Aware Prefix Scheduler (Algorithm 1 from DSpark paper).

    For each request r with confidence scores c_{r,1..gamma}:
    1. Compute prefix survival probabilities a_{r,j} = prod_{i<=j} c_{r,i}
    2. Globally sort all valid prefix extensions by survival probability
    3. Greedily add tokens, tracking throughput Θ = τ * SPS(B)
    4. Early stop when throughput stops improving
    5. Return per-request verification lengths

    Args:
        confidence_scores: list of [gamma] tensors, one per request
        throughput_profile: SPS(B) curve
        gamma: maximum block size
        min_accept_prob: minimum survival probability for valid candidates

    Returns:
        verification_lengths: list of ints, selected ℓ_r per request
    """
    R = len(confidence_scores)

    # Step 1: Compute prefix survival probabilities
    survival_probs = []
    for r in range(R):
        c = confidence_scores[r]
        a = torch.cumprod(c, dim=-1)  # cumulative product
        survival_probs.append(a)

    # Step 2: Construct candidate space E = {(r, j) | a_{r,j} > min_accept_prob}
    candidates = []  # list of (survival_prob, request_idx, position)
    for r in range(R):
        for j in range(gamma):
            prob = survival_probs[r][j].item()
            if prob > min_accept_prob:
                candidates.append((prob, r, j))

    # Step 3-5: Greedy selection
    if not candidates:
        return [0] * R

    # Sort descending by survival probability
    candidates.sort(key=lambda x: -x[0])

    # Initialize states
    lengths = [0] * R  # ℓ_r per request
    B = R  # Current batch size (each request has 1 anchor token)
    tau = float(R)  # Expected accepts (1 per request for anchor)
    best_throughput = tau * throughput_profile(B)
    best_lengths = list(lengths)
    # Track which requests have already been processed at each position
    {r: set() for r in range(R)}

    # Greedy admission
    for prob, r, j in candidates:
        # Check if this candidate extends a contiguous prefix
        # (i.e., length[r] == j, meaning positions 1..j are already scheduled)
        if lengths[r] != j:
            continue

        # Update: extend request r's verification length to j+1
        lengths[r] = j + 1
        B += 1  # One extra token
        tau += prob

        # Current throughput
        current_throughput = tau * throughput_profile(B)

        if current_throughput > best_throughput + 1e-8:
            best_throughput = current_throughput
            best_lengths = list(lengths)
        else:
            # Early stopping: throughput stopped improving
            # Restore state and exit
            lengths[r] = j
            break

    return best_lengths


class StaticScheduler:
    """Simple fallback schedulers for verification length."""

    @staticmethod
    def fixed_length(gamma: int, R: int) -> list[int]:
        """Always verify the full block."""
        return [gamma] * R

    @staticmethod
    def static_threshold(
        confidence_scores: list[torch.Tensor],
        threshold: float = 0.1,
        gamma: int = 7,
    ) -> list[int]:
        """Verify prefix until confidence drops below threshold."""
        lengths = []
        for c in confidence_scores:
            # Count how many consecutive positions exceed threshold
            count = 0
            for k in range(gamma):
                if k < len(c) and c[k].item() >= threshold:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths
