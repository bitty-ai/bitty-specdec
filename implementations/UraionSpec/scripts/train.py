#!/usr/bin/env python3
"""
Train a DSpark drafter against a frozen Qwen3 target with WandB logging.

Usage:
    python scripts/train.py --target /path/to/Qwen3-4B --data /path/to/prepared --out runs/dspark
"""

import argparse
import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uraionspec.models import DSparkDraftModel
from uraionspec.training import DSparkDataset, DSparkTrainer, collate_fn
from uraionspec.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="DSpark drafter training")
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--data", type=str, required=True, help="Dir with train.pt / eval.pt from prepare_data.py")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-anchors", type=int, default=64)
    p.add_argument("--block-size", type=int, default=7)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    p.add_argument("--markov-rank", type=int, default=256)
    p.add_argument("--markov-head-type", type=str, default="vanilla")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--ce-alpha", type=float, default=0.1)
    p.add_argument("--tv-alpha", type=float, default=0.9)
    p.add_argument("--conf-alpha", type=float, default=1.0)
    p.add_argument("--kl-alpha", type=float, default=0.0)
    p.add_argument("--tv-ramp", type=int, nargs=2, default=None, metavar=("START", "END"),
                   help="Linearly ramp tv_alpha from 0 between these steps")
    p.add_argument("--loss-decay-gamma", type=float, default=4.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", type=str, default="uraionspec-dspark")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def length_grouped_batches(lengths: list[int], batch_size: int, seed: int, group_factor: int = 50) -> list[list[int]]:
    """Shuffle, sort within large groups by length, then shuffle the batches (less padding)."""
    rng = random.Random(seed)
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    group = batch_size * group_factor
    batches = []
    for start in range(0, len(indices), group):
        chunk = sorted(indices[start:start + group], key=lambda i: lengths[i], reverse=True)
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


class EpochBatchSampler:
    def __init__(self, lengths: list[int], batch_size: int, seed: int):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        batches = length_grouped_batches(self.lengths, self.batch_size, self.seed + self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_dataset(path: str, max_length: int) -> DSparkDataset:
    samples = torch.load(path)
    return DSparkDataset(
        input_ids=[s["input_ids"] for s in samples],
        loss_mask=[s["loss_mask"] for s in samples],
        max_length=max_length,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16, device_map="cuda")
    target.eval()
    cfg = target.config

    train_ds = make_dataset(os.path.join(args.data, "train.pt"), args.max_length)
    eval_ds = make_dataset(os.path.join(args.data, "eval.pt"), args.max_length)
    pad_id = cfg.eos_token_id if isinstance(cfg.eos_token_id, int) else 0
    collate = lambda b: collate_fn(b, pad_token_id=pad_id)
    lengths = [min(len(x), args.max_length) for x in train_ds.input_ids]
    train_dl = DataLoader(
        train_ds,
        batch_sampler=EpochBatchSampler(lengths, args.batch_size, args.seed),
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model_config = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.hidden_size * 4,
        num_target_layers=len(args.target_layer_ids),
        block_size=args.block_size,
        markov_rank=args.markov_rank,
        markov_head_type=args.markov_head_type,
        use_confidence_head=args.conf_alpha > 0,
        rope_theta=getattr(cfg, "rope_theta", 1_000_000.0),
    )
    draft = DSparkDraftModel(**model_config)
    draft.share_target_embeddings(target, freeze=True)
    n_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)

    total_steps = args.max_steps or int(len(train_dl) * args.epochs)
    run_config = {**vars(args), **{f"model/{k}": v for k, v in model_config.items()},
                  "trainable_params": n_trainable, "total_steps": total_steps,
                  "train_samples": len(train_ds), "eval_samples": len(eval_ds)}
    print(json.dumps(run_config, indent=2))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log_fn = None
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=run_config, dir=args.out)
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        log_fn = lambda metrics, step: wandb.log({**metrics, "step": step})

    trainer = DSparkTrainer(
        draft_model=draft,
        target_model=target,
        target_layer_ids=args.target_layer_ids,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        ce_alpha=args.ce_alpha,
        tv_alpha=args.tv_alpha,
        conf_alpha=args.conf_alpha,
        kl_alpha=args.kl_alpha,
        tv_ramp=tuple(args.tv_ramp) if args.tv_ramp else None,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        log_fn=log_fn,
    )

    trainer.train(
        dataloader=train_dl,
        num_steps=total_steps,
        log_interval=args.log_interval,
        eval_dataloader=eval_dl,
        eval_interval=args.eval_interval,
        save_dir=os.path.join(args.out, "checkpoints"),
        save_interval=args.save_interval,
        checkpoint_extra={"model_config": model_config, "target_layer_ids": args.target_layer_ids},
    )
    print(f"peak memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    if log_fn is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
