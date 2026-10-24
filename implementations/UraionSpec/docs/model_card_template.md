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
    <img src="https://uraionlabs.com/public/icons/icon-192.png" alt="Uraion Labs" width="64" height="64">
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

---

**UraionSpec** is a clean, modular, and runnable implementation of [**DSpark**](https://www.alphaxiv.org/abs/2026.dspark) — Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation — accepted at **ICML 2026**.

It faithfully reproduces the core DSpark algorithm: a semi-autoregressive draft model that combines a parallel backbone with a lightweight sequential Markov/RNN head, a confidence head for per-position acceptance prediction, and a hardware-aware prefix scheduler that dynamically tailors verification length based on engine throughput profiles.

This is **research infrastructure** — not a model checkpoint. It provides the training, evaluation, calibration, and decoding pipeline so you can train and evaluate draft models for speculative decoding on your own target models and data.

**Intelligence is a systems problem.** This codebase is one piece of that system.

---

## What is DSpark?

DSpark is a speculative decoding framework from DeepSeek-AI that introduces two key innovations over existing methods:

1. **Semi-Autoregressive Generation** — A parallel backbone handles bulk computation while a lightweight sequential head injects inter-token dependency, combining the speed of parallel drafters with the quality of autoregressive ones. This fixes the "multi-modal collision" problem where parallel drafters produce incoherent combinations like "of problem" instead of "of course".

2. **Confidence-Scheduled Verification** — A confidence head predicts per-position acceptance probabilities, and a hardware-aware scheduler (Algorithm 1) dynamically tailors verification length based on estimated prefix survival probabilities and engine-specific throughput profiles `SPS(B)`. This prevents wasting compute on tokens with high rejection risk under heavy load.

### Architecture

```
UraionSpec/
├── src/uraionspec/
│   ├── models/              # Draft model architecture
│   │   ├── markov_head.py       # Low-rank transition bias (r=256)
│   │   ├── rnn_head.py          # GRU-like recurrent sequential head
│   │   ├── confidence_head.py   # Per-position acceptance predictor
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
│   └── utils/               # HF helpers, logging, seeding
├── scripts/                 # Runnable entry points
├── tests/                   # 55 unit & integration tests
└── docs/                    # Implementation notes, reports
```

---

## Quick Start

```bash
# Install
pip install git+https://huggingface.co/UraionLabs/UraionSpec

# Run tests
pytest tests/ -v

# Smoke train a draft model
python scripts/smoke_train.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --samples 32 --steps 5

# Evaluate acceptance
python scripts/smoke_eval.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --gamma 7 --steps 5

# Benchmark vs vanilla decoding
python scripts/run_benchmark.py \
    --target Qwen/Qwen2.5-0.5B-Instruct \
    --prompts examples/prompts.jsonl
```

---

## Key Design

| Component | Paper Reference | Implementation |
|---|---|---|
| Draft distribution | Eq. 4: `p_k(v) = softmax(U_k(v) + B_k(v))` | `draft_model.py` + `markov_head.py` |
| Markov transition bias | Eq. 5: `B = W1[x_{k-1}] W2` | `VanillaMarkov`, `GatedMarkovHead` |
| RNN sequential head | Eq. 6: GRU-like state `s_k` | `RNNHead` |
| Confidence head | Eq. 7: `c_k = σ(w^T[h_k; W1[x_{k-1}]])` | `ConfidenceHead` |
| Acceptance rate target | Eq. 8: `c* = 1 - ½‖p_d - p_t‖₁` | `compute_accept_rate()` |
| Prefix scheduler | Algorithm 1: `argmax τ · SPS(B)` | `hardware_aware_prefix_scheduler()` |
| STS calibration | §3.2.1: per-position ECE minimization | `STSCalibrator` |
| Loss function | Eq. 12: `L = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf` | `compute_dspark_loss()` |

---

## Verification

| Check | Status |
|---|---|
| 55 unit & integration tests | ✅ All passing |
| Package imports | ✅ Clean |
| Linting (ruff) | ✅ All checks passed |
| Smoke training (CPU, Qwen2.5-0.5B) | ✅ 3 steps, all losses decreasing |
| Confidence head gradient flow | ✅ Supervised by analytical acceptance rate |
| Acceptance rule correctness | ✅ Verified: all-accepted, first-rejected, partial, bonus token |

---

## Relation to DeepSpec

UraionSpec is an independent, faithful implementation of the DSpark algorithm described in the [paper](https://www.alphaxiv.org/abs/2026.dspark) and the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) repository (MIT license). While DeepSpec is a production-grade codebase with multi-GPU training, 38 TB target caches, and vLLM integration, UraionSpec focuses on:

- **Clarity** — Modular, documented Python with clean separations
- **Runability** — Smoke tests that work on a single GPU or CPU
- **Completeness** — Every algorithm component from the paper is implemented

---

## License

MIT License. Built with reference to [DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT) and the DSpark paper. Copyright © 2026 Uraion Labs.

---

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
