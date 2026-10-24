#!/usr/bin/env python3
"""Tokenize a chat dataset with the target's chat template and mark assistant tokens for loss."""

import argparse
import os
import random

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Path saved with datasets.save_to_disk")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assistant_mask(ids: list[int], im_start: int, im_end: int, assistant_ids: list[int]) -> list[int]:
    mask = [0] * len(ids)
    n_header = len(assistant_ids)
    i = 0
    while i < len(ids):
        if ids[i] == im_start and ids[i + 1:i + 1 + n_header] == assistant_ids:
            j = i + 1 + n_header
            while j < len(ids) and ids[j] != im_end:
                mask[j] = 1
                j += 1
            if j < len(ids):
                mask[j] = 1  # model must learn to emit <|im_end|>
            i = j
        i += 1
    return mask


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_from_disk(args.dataset).to_list()
    random.Random(args.seed).shuffle(ds)

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_ids = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]

    samples = []
    for row in ds:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]
        mask = assistant_mask(ids, im_start, im_end, assistant_ids)
        if sum(mask) < 16:
            continue
        samples.append({"input_ids": ids, "loss_mask": mask})

    os.makedirs(args.out, exist_ok=True)
    splits = {"eval": samples[:args.eval_samples], "train": samples[args.eval_samples:]}
    for name, split in splits.items():
        torch.save(split, os.path.join(args.out, f"{name}.pt"))
        n_tok = sum(len(s["input_ids"]) for s in split)
        n_loss = sum(sum(s["loss_mask"]) for s in split)
        print(f"{name}: {len(split)} samples, {n_tok} tokens, {n_loss} assistant tokens")


if __name__ == "__main__":
    main()
