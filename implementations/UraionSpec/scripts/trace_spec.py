#!/usr/bin/env python3
"""
Record a per-round trace of greedy speculative decoding: what the drafter proposed,
what the target chose at each slot, what was accepted, and the bonus token.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_spec import build_prompts, target_forward
from uraionspec.models import DSparkDraftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--prompt-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def traced_generate(target, draft, prompt, layer_ids, max_new_tokens, stop_ids, tok):
    seq = prompt
    features, logits = target_forward(target, seq, layer_ids)
    anchor = logits[:, -1].argmax(-1)
    generated, rounds = [], []

    while len(generated) < max_new_tokens:
        p = seq.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            block = draft.sample_block(anchor, torch.tensor([p], device=seq.device), features[:, :p])
        draft_tokens = block["draft_tokens"]
        verify_ids = torch.cat([seq, anchor[:, None], draft_tokens], dim=1)
        features, logits = target_forward(target, verify_ids, layer_ids)
        target_tokens = logits[:, p:].argmax(-1)
        matches = (draft_tokens[0] == target_tokens[0, :-1]).int()
        n_acc = int(matches.cumprod(0).sum())

        d, t = draft_tokens[0].tolist(), target_tokens[0].tolist()
        rounds.append({
            "anchor": tok.decode([int(anchor)]),
            "draft": [tok.decode([x]) for x in d],
            "target": [tok.decode([x]) for x in t[:-1]],
            "confidence": [round(float(c), 3) for c in block["confidence"][0]],
            "accepted": n_acc,
            "bonus": tok.decode([t[n_acc]]),
        })

        new_tokens = [int(anchor)] + d[:n_acc]
        stop = False
        for x in new_tokens:
            generated.append(x)
            if x in stop_ids or len(generated) >= max_new_tokens:
                stop = True
                break
        if stop:
            break
        seq = verify_ids[:, :p + 1 + n_acc]
        anchor = target_tokens[:, n_acc]
    return generated, rounds


@torch.no_grad()
def greedy_with_margins(target, prompt, max_new_tokens, stop_ids):
    seq, generated, margins = prompt, [], []
    while len(generated) < max_new_tokens:
        logits = target(input_ids=seq, use_cache=False).logits[0, -1].float()
        top2 = logits.topk(2)
        tok_id = int(top2.indices[0])
        generated.append(tok_id)
        margins.append(float(top2.values[0] - top2.values[1]))
        if tok_id in stop_ids:
            break
        seq = torch.cat([seq, torch.tensor([[tok_id]], device=seq.device)], dim=1)
    return generated, margins


def main():
    args = parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16, device_map=device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    layer_ids = ckpt["target_layer_ids"]
    draft = DSparkDraftModel(**ckpt["model_config"])
    draft.share_target_embeddings(target)
    draft.load_state_dict(ckpt["model_state_dict"], strict=False)
    draft.to(device).eval()

    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    prompts = build_prompts(os.path.join(args.data, "eval.pt"), tok, max(args.prompt_ids) + 1)

    traces = []
    for i in args.prompt_ids:
        prompt_t = torch.tensor([prompts[i]], device=device)
        generated, rounds = traced_generate(target, draft, prompt_t, layer_ids, args.max_new_tokens, stop_ids, tok)
        reference, margins = greedy_with_margins(target, prompt_t, args.max_new_tokens, stop_ids)
        divergence = None
        for j, (a, b) in enumerate(zip(generated, reference)):
            if a != b:
                divergence = {"index": j, "spec_token": tok.decode([a]), "greedy_token": tok.decode([b]),
                              "greedy_top2_logit_margin": round(margins[j], 4)}
                break
        if divergence is None and len(generated) != len(reference):
            divergence = {"index": min(len(generated), len(reference)), "note": "length differs"}

        text = tok.decode(prompts[i])
        user = text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        lengths = [r["accepted"] + 1 for r in rounds]
        traces.append({
            "prompt_id": i,
            "prompt": user.strip(),
            "rounds": rounds,
            "new_tokens": len(generated),
            "tau": sum(lengths) / len(lengths),
            "matches_greedy": divergence is None,
            "divergence": divergence,
        })
        print(i, len(rounds), round(traces[-1]["tau"], 3), divergence, flush=True)

    with open(args.out, "w") as f:
        json.dump({"step": ckpt.get("step"), "block_size": ckpt["model_config"]["block_size"], "traces": traces}, f, indent=1)


if __name__ == "__main__":
    main()
