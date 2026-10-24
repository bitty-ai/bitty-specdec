# DSpark Implementation Notes

## Architecture Summary

DSpark introduces two key innovations over existing speculative decoding methods:

### 1. Semi-Autoregressive Generation
- **Parallel backbone** (DFlash-style): Processes all γ positions in a single forward pass, making drafting latency nearly independent of block size.
- **Sequential head**: Two variants:
  - *Markov head* (`VanillaMarkov`/`GatedMarkovHead`): Low-rank transition bias `B = W1[x_{k-1}] @ W2` (r=256).
  - *RNN head* (`RNNHead`): GRU-like recurrent state accumulating full prefix history.
- The sequential stage fixes the "multi-modal collision" problem where parallel drafters produce incoherent combinations like "of problem" instead of "of course".

### 2. Confidence-Scheduled Verification
- **Confidence head**: Predicts per-position conditional acceptance probability `c_k = σ(w^T[h_k; W1[x_{k-1}]])`.
- **Sequential Temperature Scaling (STS)**: Calibrates cumulative prefix survival probabilities left-to-right via 1D grid search minimizing ECE.
- **Hardware-Aware Prefix Scheduler (Algorithm 1)**: Maximizes `Θ = τ · SPS(B)` by greedily admitting highest-survival-probability tokens with early stopping.

## Algorithm Details

### Acceptance Rule (Lossless)
```
P(accept token x_k) = min(1, p^t_k(x_k) / p^d_k(x_k))
```
- First rejection at position k discards tokens k+1..γ
- One bonus token sampled from residual distribution at rejection position
- Preserves exact target distribution

### Draft Distribution
```
p_k(v | x_0, x_<k) = exp(U_k(v) + B_k(x_0, x_<k, v)) / Σ exp(U_k(u) + B_k(...))
```
where `U_k` are base logits from parallel backbone and `B_k` is the sequential transition bias.

### Training Objective (Eq. 12)
```
L = 0.1 · L_ce + 0.9 · L_tv + 1.0 · L_conf
```
- `L_ce`: Cross-entropy for next-token prediction
- `L_tv`: Total variation distance `||p_d - p_t||_1` (proxy for acceptance rate)
- `L_conf`: Binary cross-entropy on confidence predictions
- All position-weighted by `w_k = exp(-(k-1)/γ)`

### STS Calibration
For each position k = 1..γ:
- Compute cumulative product of calibrated scores up to k
- Find temperature T_k minimizing ECE of cumulative product
- Apply T_k to k-th position score
- Order-preserving: doesn't disrupt relative rankings

### Algorithm 1: Hardware-Aware Prefix Scheduler
```
1. Compute a_{r,j} = ∏_{i≤j} c_{r,i} for each request r, position j
2. Create candidate set E = {(r,j) | a_{r,j} > 0}
3. Sort E descending by a_{r,j}
4. Greedily add candidates:
   - Update ℓ_r = j, B += 1, τ += a_{r,j}
   - Compute Θ = τ * SPS(B)
   - If Θ > best, update best lengths; else break
5. Return per-request verification lengths
```

## What's Implemented Now

| Component | Status | Location |
|---|---|---|
| Markov head (vanilla, gated) | ✅ | `models/markov_head.py` |
| RNN head | ✅ | `models/rnn_head.py` |
| Confidence head | ✅ | `models/confidence_head.py` |
| Acceptance rule | ✅ | `decoding/acceptance.py` |
| Hardware-aware scheduler | ✅ | `decoding/scheduler.py` |
| Static scheduler (fallback) | ✅ | `decoding/scheduler.py` |
| Speculative decoding loop | ✅ | `decoding/speculative.py` |
| Training dataset | ✅ | `training/dataset.py` |
| Loss functions (CE + TV + Conf) | ✅ | `training/losses.py` |
| Training loop | ✅ | `training/train_drafter.py` |
| Target cache generation | ✅ | `training/cache_targets.py` |
| STS calibration | ✅ | `calibration/sts.py` |
| Acceptance evaluation | ✅ | `evaluation/eval_acceptance.py` |
| Latency benchmarking | ✅ | `evaluation/benchmark_latency.py` |
| Unit tests (55) | ✅ | `tests/` |
| Smoke train script | ✅ | `scripts/smoke_train.py` |
| Smoke eval script | ✅ | `scripts/smoke_eval.py` |
| Benchmark script | ✅ | `scripts/run_benchmark.py` |

## Deviations from Paper

1. **Parallel backbone**: Uses `nn.TransformerEncoder` instead of the full DFlash-style backbone with target model KV injection.
2. **Training efficiency**: No multi-GPU training, no 38TB target cache. Target logits computed on-the-fly.
3. **SPS profiling**: Uses a synthetic throughput profile instead of real engine profiling.
4. **Mask token**: Uses token 0 as placeholder; the paper uses learned mask embeddings.
5. **Anchor modification**: Paper treats anchor as first prediction position (γ inputs → γ outputs). We follow the original DFlash style (γ inputs → γ outputs) where anchor is part of the input.

## Future Work

1. Implement full DFlash backbone with target model KV injection
2. Multi-GPU distributed training (DeepSpeed/FSDP)
3. Real engine throughput profiling
4. Tree-based verification for autoregressive drafters
5. Integration with vLLM or similar serving frameworks
6. Qwen3-specific model classes with proper configuration
