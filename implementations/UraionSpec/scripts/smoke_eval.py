#!/usr/bin/env python3
"""
Smoke evaluation script for UraionSpec.

Evaluates a trained DSpark draft model on a few prompts, measuring
accepted length, acceptance rate, and latency.

Usage:
    uv run python scripts/smoke_eval.py --target Qwen/Qwen3-0.6B --checkpoint <path>
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uraionspec.models import DSparkDraftModel
from uraionspec.evaluation import evaluate_acceptance, benchmark_latency
from uraionspec.utils import seed_everything, get_hf_token


def parse_args():
    parser = argparse.ArgumentParser(description="UraionSpec smoke evaluation")
    parser.add_argument("--target", type=str, default="Qwen/Qwen3-0.6B",
                        help="Target model name")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Draft model checkpoint path (optional)")
    parser.add_argument("--gamma", type=int, default=7,
                        help="Draft block size")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of decoding steps")
    parser.add_argument("--num-prompts", type=int, default=2,
                        help="Number of test prompts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    hf_token = get_hf_token()

    print("=== UraionSpec Smoke Evaluation ===")
    print(f"Target: {args.target}, Device: {args.device}")
    token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)

    # 1. Load target model
    print("\n[1/4] Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, token=token,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, token=token, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Create draft model
    print("\n[2/4] Creating draft model...")
    vocab_size = target.config.vocab_size
    hidden_size = target.config.hidden_size

    draft = DSparkDraftModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden_size * 2,
        markov_rank=64,
        markov_head_type="vanilla",
        use_confidence_head=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        draft.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint} (step={ckpt.get('step', '?')})")

    draft.to(args.device)
    draft.eval()

    # 3. Create draft model function
    def draft_fn(anchor_ids, gamma, temperature=0.0, return_confidence=True):
        return draft.sample_block(
            anchor_ids=anchor_ids,
            gamma=gamma,
            temperature=temperature,
            return_confidence=return_confidence,
        )

    # 4. Run acceptance evaluation
    print("\n[3/4] Running acceptance evaluation...")
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In this paper, we propose a novel approach to speculative decoding.",
        "Machine learning is a field of artificial intelligence that uses",
        "Once upon a time, in a kingdom far, far away,",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ][:args.num_prompts]

    all_results = []
    for prompt in test_prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        print(f"  Evaluating prompt ({len(prompt)} chars)...")

        # Benchmark vanilla
        benchmark_latency(
            target, encoded["input_ids"][:, :10],
            num_tokens=10, num_warmup=2, device=args.device,
        )

        # Benchmark speculative
        spec_results = evaluate_acceptance(
            draft_model_fn=draft_fn,
            target_model=target,
            prompt_ids=encoded["input_ids"],
            prompt_mask=encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            gamma=args.gamma,
            num_steps=args.steps,
            temperature=0.0,
            device=args.device,
        )
        all_results.append(spec_results)

    # Summary
    print("\n[4/4] Results:")
    avg_accepted = torch.tensor([r["mean_accepted_length"] for r in all_results]).mean().item()
    avg_accept_rate = torch.tensor([r["acceptance_rate"] for r in all_results]).mean().item()

    print(f"  Mean accepted length (τ): {avg_accepted:.2f}")
    print(f"  Mean acceptance rate: {avg_accept_rate:.3f}")
    print(f"  γ = {args.gamma}")
    print(f"  Speedup potential: τ / 1 ≈ {avg_accepted:.1f}x (vs 1 token/step)")
    print("\n=== Smoke evaluation complete ===")


if __name__ == "__main__":
    main()
