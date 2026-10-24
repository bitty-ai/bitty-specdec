#!/usr/bin/env python3
"""
Smoke training script for UraionSpec.

Trains a DSpark draft model on a tiny dataset subset to verify
the training pipeline end-to-end (forward/backward pass, loss computation).

Usage:
    uv run python scripts/smoke_train.py --target Qwen/Qwen3-0.6B --samples 32 --steps 5
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uraionspec.models import DSparkDraftModel
from uraionspec.training import DSparkDataset, get_dataloader, DSparkTrainer
from uraionspec.utils import seed_everything, get_hf_token


def parse_args():
    parser = argparse.ArgumentParser(description="UraionSpec smoke training")
    parser.add_argument("--target", type=str, default="Qwen/Qwen3-0.6B",
                        help="Target model name")
    parser.add_argument("--samples", type=int, default=32,
                        help="Number of training samples")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--block-size", type=int, default=4,
                        help="Draft block size (gamma)")
    parser.add_argument("--num-anchors", type=int, default=2,
                        help="Number of anchor blocks per sample")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Checkpoint save directory")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    hf_token = get_hf_token()

    print("=== UraionSpec Smoke Training ===")
    print(f"Target model: {args.target}")
    print(f"Samples: {args.samples}, Steps: {args.steps}")
    print(f"Device: {args.device}")

    # 1. Load target model
    print(f"\n[1/5] Loading target model {args.target}...")
    token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        token=token,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    target.eval()
    for p in target.parameters():
        p.requires_grad = False
    print(f"  Target model loaded: {target.config.model_type}, {sum(p.numel() for p in target.parameters())/1e6:.1f}M params")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.target,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load tiny dataset
    print("\n[2/5] Loading dataset...")
    dataset = load_dataset("trl-lib/Capybara", split="train")
    texts = [tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    ) for sample in dataset.select(range(args.samples))]
    print(f"  Loaded {len(texts)} samples")

    train_dataset = DSparkDataset(
        tokenizer(texts, truncation=True, max_length=512, padding=False)["input_ids"]
    )
    dataloader = get_dataloader(
        train_dataset, batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    # 4. Create draft model
    print("\n[3/5] Creating DSpark draft model...")
    vocab_size = target.config.vocab_size
    hidden_size = target.config.hidden_size

    target_layer_ids = [1, target.config.num_hidden_layers // 2, target.config.num_hidden_layers - 3]
    draft = DSparkDraftModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,  # tiny backbone for smoke test
        num_attention_heads=target.config.num_attention_heads,
        num_kv_heads=target.config.num_key_value_heads,
        intermediate_size=hidden_size * 2,
        num_target_layers=len(target_layer_ids),
        block_size=args.block_size,
        markov_rank=64,  # smaller rank for smoke test
        markov_head_type="vanilla",
        use_confidence_head=True,
    )
    draft.share_target_embeddings(target)
    num_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  Draft model: {num_params/1e6:.2f}M params ({num_params:,})")

    # 5. Train
    print(f"\n[4/5] Training for {args.steps} steps...")
    trainer = DSparkTrainer(
        draft_model=draft,
        target_model=target,
        target_layer_ids=target_layer_ids,
        learning_rate=args.lr,
        warmup_steps=0,
        total_steps=args.steps,
        num_anchors=args.num_anchors,
        ce_alpha=0.1,
        tv_alpha=0.9,
        conf_alpha=1.0,
        device=args.device,
    )

    history = trainer.train(
        dataloader=dataloader,
        num_steps=args.steps,
        log_interval=1,
        save_dir=args.save_dir,
    )

    # 6. Summary
    print("\n[5/5] Training complete!")
    final = history
    print(f"  Final loss: {final['loss'][-1]:.4f}")
    print(f"  Final CE: {final['ce_loss'][-1]:.4f}")
    print(f"  Final TV: {final['tv_loss'][-1]:.4f}")
    print(f"  Final Conf: {final['conf_loss'][-1]:.4f}")
    print("\n=== Smoke training PASSED ===")


if __name__ == "__main__":
    main()
