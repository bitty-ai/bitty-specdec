#!/usr/bin/env python3
"""
Greedy speculative decoding with a trained DSpark drafter against the target model.

Reports the real acceptance length (tokens produced per target forward) and checks that
the output matches plain greedy decoding of the target.

Usage:
    python scripts/eval_spec.py --target /path/to/Qwen3-4B --checkpoint runs/x/checkpoints/last.pt \
        --data /path/to/prepared --num-prompts 20
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uraionspec.models import DSparkDraftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True, help="Dir with eval.pt from prepare_data.py")
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--check-greedy", action="store_true", help="Compare against target-only greedy decoding")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def target_forward(target, ids: torch.Tensor, layer_ids: list[int]):
    out = target.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    features = torch.cat([out.hidden_states[i + 1] for i in layer_ids], dim=-1)
    logits = target.lm_head(out.last_hidden_state)
    return features, logits


@torch.no_grad()
def speculative_generate(target, draft, prompt: torch.Tensor, layer_ids: list[int], max_new_tokens: int, stop_ids: set[int]):
    seq = prompt
    features, logits = target_forward(target, seq, layer_ids)
    anchor = logits[:, -1].argmax(-1)
    generated, accept_lengths = [], []

    while len(generated) < max_new_tokens:
        p = seq.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            block = draft.sample_block(anchor, torch.tensor([p], device=seq.device), features[:, :p], return_confidence=False)
        draft_tokens = block["draft_tokens"]

        verify_ids = torch.cat([seq, anchor[:, None], draft_tokens], dim=1)
        features, logits = target_forward(target, verify_ids, layer_ids)
        target_tokens = logits[:, p:].argmax(-1)  # target's choice for positions p+1 .. p+γ+1

        matches = (draft_tokens[0] == target_tokens[0, :-1]).int()
        n_acc = int(matches.cumprod(0).sum())
        new_tokens = [int(anchor)] + draft_tokens[0, :n_acc].tolist()
        accept_lengths.append(n_acc + 1)

        stop = False
        for tok in new_tokens:
            generated.append(tok)
            if tok in stop_ids or len(generated) >= max_new_tokens:
                stop = True
                break
        if stop:
            break
        seq = verify_ids[:, :p + 1 + n_acc]
        anchor = target_tokens[:, n_acc]

    return generated, accept_lengths


@torch.no_grad()
def greedy_generate(target, prompt: torch.Tensor, max_new_tokens: int, stop_ids: set[int]):
    seq, generated = prompt, []
    while len(generated) < max_new_tokens:
        tok = int(target(input_ids=seq, use_cache=False).logits[0, -1].argmax())
        generated.append(tok)
        if tok in stop_ids:
            break
        seq = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
    return generated


def build_prompts(eval_path: str, tokenizer, n: int) -> list[list[int]]:
    think = tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    prompts = []
    for s in torch.load(eval_path):
        first = s["loss_mask"].index(1)
        prompt = s["input_ids"][:first]
        if prompt[-len(think):] != think:
            prompt = prompt + think
        prompts.append(prompt)
        if len(prompts) == n:
            break
    return prompts


def main():
    args = parse_args()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16, device_map=device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    layer_ids = ckpt["target_layer_ids"]
    draft = DSparkDraftModel(**ckpt["model_config"])
    draft.share_target_embeddings(target)
    missing, unexpected = draft.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not unexpected and set(missing) <= {"embed_tokens.weight", "lm_head.weight"}, (missing, unexpected)
    draft.to(device).eval()

    stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
    prompts = build_prompts(os.path.join(args.data, "eval.pt"), tokenizer, args.num_prompts)

    all_lengths, results, exact = [], [], 0
    for i, prompt in enumerate(prompts):
        prompt_t = torch.tensor([prompt], device=device)
        generated, lengths = speculative_generate(target, draft, prompt_t, layer_ids, args.max_new_tokens, stop_ids)
        all_lengths += lengths
        row = {"prompt_tokens": len(prompt), "new_tokens": len(generated), "rounds": len(lengths),
               "tau": sum(lengths) / len(lengths)}
        if args.check_greedy:
            reference = greedy_generate(target, prompt_t, args.max_new_tokens, stop_ids)
            row["matches_greedy"] = reference == generated
            exact += row["matches_greedy"]
        results.append(row)
        print(f"[{i}] {row}", flush=True)

    hist = [sum(1 for x in all_lengths if x == k) / len(all_lengths) for k in range(1, ckpt["model_config"]["block_size"] + 2)]
    summary = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "num_prompts": len(prompts),
        "tau_mean": sum(all_lengths) / len(all_lengths),
        "accept_length_hist": [round(h, 4) for h in hist],
        "greedy_exact_match": f"{exact}/{len(prompts)}" if args.check_greedy else None,
        "sample_output": tokenizer.decode(generated),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_prompt": results}, f, indent=2)


if __name__ == "__main__":
    main()
