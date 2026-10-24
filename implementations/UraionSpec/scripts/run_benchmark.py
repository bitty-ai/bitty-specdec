#!/usr/bin/env python3
"""
Benchmark script for UraionSpec.

Compares speculative decoding against vanilla autoregressive decoding
on a set of prompts.

Usage:
    uv run python scripts/run_benchmark.py --target Qwen/Qwen3-0.6B --prompts examples/prompts.jsonl
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uraionspec.models import DSparkDraftModel
from uraionspec.evaluation import evaluate_acceptance
from uraionspec.utils import seed_everything, get_hf_token


def parse_args():
    parser = argparse.ArgumentParser(description="UraionSpec benchmark")
    parser.add_argument("--target", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompts", type=str, default=None,
                        help="JSONL file with prompts")
    parser.add_argument("--gamma", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    hf_token = get_hf_token()
    token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)

    print("=== UraionSpec Benchmark ===")
    print(f"Target: {args.target}, Gamma: {args.gamma}, Device: {args.device}")

    # Load prompts
    test_prompts = [
        "Explain the concept of recursive functions in programming.",
        "Write a Python function to sort a list of integers.",
        "What is the difference between a tensor and a matrix?",
        "Describe how speculative decoding works in LLM inference.",
        "Write a haiku about machine learning.",
    ]
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as f:
            test_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

    print(f"  Using {len(test_prompts)} prompts")

    # Load models
    print(f"\nLoading target model: {args.target}")
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

    print("\nCreating draft model...")
    draft = DSparkDraftModel(
        vocab_size=target.config.vocab_size,
        hidden_size=target.config.hidden_size,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=target.config.hidden_size * 2,
        markov_rank=64,
        markov_head_type="vanilla",
        use_confidence_head=True,
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        draft.load_state_dict(ckpt["model_state_dict"])
    draft.to(args.device).eval()

    def draft_fn(anchor_ids, gamma, temperature=0.0, return_confidence=True):
        return draft.sample_block(anchor_ids, gamma, temperature, return_confidence)

    # Run benchmarks
    results = {"vanilla": [], "speculative": []}

    print("\nRunning benchmarks...")
    for prompt in tqdm(test_prompts, desc="Benchmarking"):
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)

        # Vanilla
        v_start = time.time()
        _ = target.generate(
            encoded["input_ids"],
            max_new_tokens=args.steps * args.gamma,
            do_sample=False,
            temperature=1.0,
        )
        v_time = time.time() - v_start

        # Speculative
        spec = evaluate_acceptance(
            draft_model_fn=draft_fn,
            target_model=target,
            prompt_ids=encoded["input_ids"],
            prompt_mask=encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            gamma=args.gamma,
            num_steps=args.steps,
            temperature=0.0,
            device=args.device,
        )

        results["vanilla"].append({"time_s": v_time})
        results["speculative"].append(spec)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    if results["vanilla"]:
        avg_v_time = sum(r["time_s"] for r in results["vanilla"]) / len(results["vanilla"])
        print(f"  Vanilla:       {avg_v_time:.2f}s avg")
    if results["speculative"]:
        avg_tau = sum(r["mean_accepted_length"] for r in results["speculative"]) / len(results["speculative"])
        avg_rate = sum(r["acceptance_rate"] for r in results["speculative"]) / len(results["speculative"])
        print(f"  Speculative τ: {avg_tau:.2f} tokens/round")
        print(f"  Acceptance:    {avg_rate:.3f}")
    print(f"{'='*60}")
    print(f"  γ = {args.gamma}")
    print(f"  Steps = {args.steps}")
    print(f"  Target = {args.target}")
    if avg_tau > 1:
        speedup = avg_tau / (1 + 1/args.gamma)
        print(f"  Est. speedup:  {speedup:.2f}x (vs 1 token/step)")


if __name__ == "__main__":
    main()
