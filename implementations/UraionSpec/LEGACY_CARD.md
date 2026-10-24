---
library_name: uraionspec
license: mit
language:
- en
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- deepseek
- llm-inference
- model-optimization
- transformer
- pytorch
- efficient-llm
- inference-acceleration
- draft-model
- torch
- uraion-labs
- uraion
- systems-research
- icml-2026
- acceptance-scheduling
- semi-autoregressive
- confidence-prediction
- calibration
sdk: docker
sdk_version: "1.0"
---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://uraionlabs.com/public/icons/icon-192.png">
    <img src="https://uraionlabs.com/public/icons/icon-192.png" alt="Uraion Labs" width="80" height="80">
  </picture>
</p>

<p align="center">
  <strong style="font-family: 'Instrument Serif', Georgia, serif; font-size: 2rem; color: #F7F4ED; letter-spacing: -0.02em;">
    Uraion Labs
  </strong>
  <br>
  <span style="font-family: 'Inter', sans-serif; font-size: 0.875rem; color: #8A8478;">Foundational systems research.</span>
</p>

<p align="center">
  <strong style="font-family: 'Inter', sans-serif; font-size: 1.15rem; color: #E45A1A;">
    UraionSpec
  </strong>
  <br>
  <span style="font-family: 'Inter', sans-serif; font-size: 0.875rem; color: #8A8478;">
    Faithful DSpark-style Speculative Decoding — modular, runnable, verified.
  </span>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/pytorch-2.1+-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build"/>
  <img src="https://img.shields.io/badge/tests-80%20passing-brightgreen.svg" alt="Tests"/>
</p>

---

**UraionSpec** is a clean, modular, and runnable implementation of [**DSpark**](https://www.alphaxiv.org/abs/2026.dspark) — Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation — accepted at **ICML 2026**. It faithfully reproduces the core DSpark algorithm while being practical for small-scale experimentation, training, and evaluation.

This is **research infrastructure** — not a model checkpoint. It provides the training, evaluation, calibration, and decoding pipeline so you can train and evaluate draft models for speculative decoding on your own target models and data.

**Intelligence is a systems problem.** This codebase is one piece of that system.

## What is DSpark?

DSpark is a state-of-the-art speculative decoding framework from DeepSeek-AI that introduces two key innovations:

1. **Semi-Autoregressive Generation** — A parallel backbone handles bulk compute while a lightweight sequential head (Markov or RNN) injects inter-token dependency, combining the speed of parallel drafters with the quality of autoregressive ones.
2. **Confidence-Scheduled Verification** — A confidence head predicts per-position acceptance probabilities, and a hardware-aware scheduler dynamically tailors the verification length based on prefix survival probabilities and engine throughput profiles. This prevents wasted compute on high-rejection tokens under heavy load.

### Architecture

```
UraionSpec/
├── src/uraionspec/
│   ├── models/              # DSpark draft model
│   │   ├── markov_head.py       # Low-rank transition bias (r=256)
│   │   ├── rnn_head.py          # GRU-like recurrent sequential head
│   │   ├── confidence_head.py   # Per-position acceptance predictor
│   │   ├── dflash_backbone.py   # DFlash-style backbone with KV injection ⭐
│   │   └── draft_model.py       # Combined parallel backbone + heads
│   ├── decoding/            # Speculative decoding core
│   │   ├── acceptance.py        # Lossless rejection sampling (min ratio)
│   │   ├── scheduler.py         # Algorithm 1: Hardware-aware prefix scheduler
│   │   └── speculative.py       # Orchestration: draft → verify → accept
│   ├── training/            # Training pipeline
│   │   ├── dataset.py           # Anchor-block dataset preparation
│   │   ├── losses.py            # CE + TV + Confidence (position-weighted)
│   │   ├── train_drafter.py     # Training loop (frozen target)
│   │   └── cache_targets.py     # Target logit cache generation
│   ├── calibration/         # Sequential Temperature Scaling
│   │   └── sts.py               # Left-to-right ECE minimization
│   ├── evaluation/          # Evaluation & benchmarking
│   │   ├── eval_acceptance.py   # Acceptance rate / length metrics
│   │   └── benchmark_latency.py # Vanilla vs speculative latency
│   └── utils/               # HF helpers, logging, seeding
├── scripts/                 # Runnable entry points
│   ├── smoke_train.py
│   ├── smoke_eval.py
│   └── run_benchmark.py
├── tests/                   # 80 unit & integration tests
└── docs/                    # Implementation notes, reports
```

## Installation

```bash
# Install directly from HuggingFace
pip install git+https://huggingface.co/UraionLabs/UraionSpec

# Or clone from HuggingFace
git clone https://huggingface.co/UraionLabs/UraionSpec
cd UraionSpec
pip install -e .

# With development dependencies (tests, linting)
pip install -e ".[dev]"
```

## Quick Start

### Smoke Training
Train a DSpark draft model on a tiny dataset to verify end-to-end gradient flow:

```bash
python scripts/smoke_train.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --samples 32 \
    --steps 5 \
    --batch-size 2 \
    --block-size 4
```

### Smoke Evaluation
Evaluate a trained draft model's acceptance characteristics:

```bash
python scripts/smoke_eval.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --checkpoint /path/to/checkpoint.pt \
    --gamma 7 \
    --steps 5
```

### Benchmark
Compare speculative decoding against vanilla autoregressive generation:

```bash
python scripts/run_benchmark.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --prompts examples/prompts.jsonl \
    --gamma 7 \
    --steps 10
```

### Run Tests
```bash
pytest tests/ -v
```

## Key Components

### Markov Sequential Head
Implements low-rank transition bias `B(x_{k-1}, x_k) = W1[x_{k-1}] @ W2` where `W1 ∈ R^{V×r}`, `W2 ∈ R^{r×V}` (r=256 default). Available as `VanillaMarkov` or `GatedMarkovHead` (modulated by backbone hidden state).

### RNN Sequential Head
GRU-like gated recurrent state across positions: 
```
s_k = sigmoid(W_g z_k) ⊙ s_{k-1} + (1 - sigmoid(W_g z_k)) ⊙ tanh(W_c z_k)
```
where `z_k = [s_{k-1}; W1[x_{k-1}]; h_k]`. Captures full prefix history.

### Confidence Head
Predicts per-position conditional acceptance probability:
```
c_k = sigmoid(w^T [h_k; W1[x_{k-1}])
```
Supervised by analytical acceptance rate `c*_k = 1 - 0.5 × ||p_d - p_t||_1`.

### Hardware-Aware Prefix Scheduler (Algorithm 1)
Maximizes expected throughput `Θ = τ × SPS(B)` by:
1. Computing prefix survival probabilities `a_{r,j} = ∏_{i≤j} c_{r,i}`
2. Globally sorting candidates by `a_{r,j}`
3. Greedily admitting tokens with early stopping to preserve non-anticipating property

### Sequential Temperature Scaling
Calibrates cumulative confidence products left-to-right via 1D grid search minimizing Expected Calibration Error (ECE) at each position.

### Loss Functions (DSpark Eq. 12)
```
L = 0.1 × L_ce + 0.9 × L_tv + 1.0 × L_conf
```
- `L_ce`: Cross-entropy for next-token prediction
- `L_tv`: Total variation distance `||p_d - p_t||_1`
- `L_conf`: Binary cross-entropy on confidence predictions

All position-weighted by `w_k = exp(-(k-1)/γ)` emphasizing earlier positions.

## Verification

| Component | Status |
|---|---|
| 80 unit & integration tests | ✅ All passing |
| DFlash backbone with KV injection | ✅ 17 tests, all shapes & gradients verified |
| Sampling utilities (residual, GQA) | ✅ 8 tests |
| Package import | ✅ Clean |
| Linting (ruff) | ✅ All checks passed |
| Smoke training (CPU) | ✅ 3 steps, all losses decreasing |
| Confidence head training | ✅ Supervised by analytical acceptance rate |

## Reproducing Paper Results

The DSpark paper trains on the full [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend) dataset (1.3M samples) across multiple GPUs. For production-scale reproduction, see the official [DeepSpec](https://github.com/deepseek-ai/DeepSpec) repository.

For small-scale experimentation:
```bash
# Train on Colab A100
colab run -s uraionspec-train --gpu A100 --keep --timeout 28800 \
  python scripts/smoke_train.py --target Qwen/Qwen3-4B --samples 10000 --steps 1000
```

## Relation to DeepSpec

UraionSpec is an independent, faithful implementation of the DSpark algorithm described in the [paper](https://www.alphaxiv.org/abs/2026.dspark) and the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) repository (MIT license). While DeepSpec is a production-grade codebase with multi-GPU training, 38 TB target caches, and vLLM integration, UraionSpec focuses on:

- **Clarity** — Modular, documented Python with clean separations
- **Runability** — Smoke tests that work on a single GPU or CPU
- **Completeness** — Every algorithm component from the paper is implemented

## Current Limitations

- **Parallel backbone**: Uses `nn.TransformerEncoder` — not the full DFlash-style backbone with target model KV injection described in Section 3.1 of the paper.
- **No multi-GPU**: Single-device only.
- **Synthetic SPS profile**: Uses a default throughput curve — for real systems, profile your engine and pass the table.
- **No vLLM integration**: For production serving, see DeepSpec's integration.

## License

MIT License. Built with reference to [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT) and the DSpark paper. Copyright © 2026 Uraion Labs.

## Citation

```bibtex
@software{uraionspec2026,
  author = {Uraion Labs},
  title = {UraionSpec: Faithful DSpark-style Speculative Decoding},
  year = {2026},
  url = {https://huggingface.co/UraionLabs/UraionSpec}
}

@article{cheng2026dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and Li, Jiashi and Xiong, Yunfan and others},
  journal={ICML},
  year={2026}
}
```
