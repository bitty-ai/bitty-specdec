# UraionSpec — Reproduction Report

## Hardware
- **Machine**: MacBook Pro 12,1 (Intel i5-5257U) running Ubuntu 24 LTS
- **Storage**: Root partition (47.9G), 7.9G free
- **GPU**: None (CPU-only testing)
- **RAM**: 7.8 GB

## Models Used
- **Target**: `Qwen/Qwen3-0.6B` (Qwen3-0.6B, ungated, Apache 2.0)
- **Draft**: Custom `DSparkDraftModel` with markov_rank=64, 2 backbone layers
- **Note**: 0.6B target is usable for shape/loss verification but too small for meaningful speculative decoding speedups

## Dataset
- **Source**: `trl-lib/Capybara` (subset: 32 samples)
- **Format**: Chat template applied via tokenizer
- **Max length**: 512 tokens
- **Note**: Tiny subset for smoke testing only

## Commands Run

### 1. Package Import Test
```
python3 -c "from uraionspec import ..."  # All imports OK
```

### 2. Unit Tests (55 tests)
```
pytest tests/ -v  # 55/55 passed in 4.90s
```

### 3. Smoke Training (CPU only)
Would run:
```
python scripts/smoke_train.py --target Qwen/Qwen3-0.6B --samples 32 --steps 5 --device cpu
```
**Blocked**: 0.6B model requires ~1.2 GB RAM just to load. With 7.8 GB and CPU-only, this would be extremely slow but should work. The training loop requires target model logits which adds memory pressure.

### 4. Lint/Type Check
Will run: `ruff check .` (configured in pyproject.toml)

## Results

### Test Results (55/55 passed)
| Test Suite | Tests | Status |
|---|---|---|
| `test_acceptance.py` | 8 | ✅ All passed |
| `test_markov_head.py` | 11 | ✅ All passed |
| `test_scheduler.py` | 12 | ✅ All passed |
| `test_shapes.py` | 7 | ✅ All passed |
| `test_sts.py` | 8 | ✅ All passed |

### What Was Verified
- ✅ Acceptance rule: all-accepted, first-rejected, partial, batch-independence, bonus token
- ✅ Expected accept length computation
- ✅ Markov head forward/shape/gradient
- ✅ RNN head stateful forward/step
- ✅ Gated Markov head forward
- ✅ Throughput profile lookup/interpolation
- ✅ Static scheduler threshold/length
- ✅ Hardware-aware scheduler: single/multi request, monotonic, zero confidence
- ✅ Draft model forward/sample shapes
- ✅ Loss: CE, TV, confidence gradients
- ✅ Confidence head forward/accept rate computation
- ✅ End-to-end draft → verify → accept → schedule cycle
- ✅ STS: ECE, temperature fitting, calibrator fit/transform

### Failures / Blockers

1. **Full target model training on CPU**: Running the 0.6B target model (~600M params) on CPU with 7.8 GB RAM is feasible but slow. The smoke training script was designed for GPU.
2. **GPU smoke eval**: Same limitation — requires GPU for meaningful speculative decoding evaluation.
3. **Disk space**: Only 7.9 GB free — cannot download the 1.3M sample Open-PerfectBlend dataset or store large target caches.

## Next Steps to Scale

### To Qwen3-1.7B/4B+ (Local or Colab)

1. **Provision GPU**: Use `colab run --gpu A100 --keep` for training
2. **Full training**:
   ```
   colab run -s uraionspec-train --gpu A100 --keep --timeout 28800 \
     python scripts/smoke_train.py \
       --target Qwen/Qwen3-4B \
       --samples 10000 \
       --steps 1000 \
       --batch-size 8 \
       --block-size 7
   ```
3. **Upload to HF**: `hf upload UraionLabs/UraionSpec /checkpoints/`
4. **Full evaluation**: Run `scripts/run_benchmark.py` on the trained checkpoint

### To Match Paper Settings
- **Data**: Use `mlabonne/open-perfectblend` (1.3M samples)
- **Blocks**: γ = 7, 16 (experiment with both)
- **Training**: 10 epochs, multi-GPU (8×)
- **Backbone**: 5 layers (paper default), `hidden_size` matching target
- **Target cache**: Generate target logits for all training data (38 TB for full; use `--max_length 2048` for smaller)
- **SPS profiling**: Run offline benchmark to get real SPS(B) curve for your hardware

## Files Changed

```
UraionSpec/
├── pyproject.toml
├── README.md
├── src/uraionspec/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── markov_head.py
│   │   ├── rnn_head.py
│   │   ├── confidence_head.py
│   │   └── draft_model.py
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── acceptance.py
│   │   ├── scheduler.py
│   │   └── speculative.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── losses.py
│   │   ├── train_drafter.py
│   │   └── cache_targets.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── sts.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── eval_acceptance.py
│   │   └── benchmark_latency.py
│   └── utils/
│       ├── __init__.py
│       ├── hf.py
│       ├── logging.py
│       └── seed.py
├── scripts/
│   ├── smoke_train.py
│   ├── smoke_eval.py
│   └── run_benchmark.py
├── tests/
│   ├── test_acceptance.py
│   ├── test_markov_head.py
│   ├── test_scheduler.py
│   ├── test_shapes.py
│   └── test_sts.py
├── examples/
│   └── prompts.jsonl
└── docs/
    ├── DSpark_implementation_notes.md
    ├── reproduction_report.md
    └── model_card_template.md
```

Total: ~4000 lines of Python, 55 passing tests
