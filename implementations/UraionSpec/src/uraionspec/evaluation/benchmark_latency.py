"""Latency benchmarking utilities."""
import time

import torch
from transformers import PreTrainedModel


@torch.no_grad()
def benchmark_latency(
    model: PreTrainedModel,
    prompt_length: int = 128,
    num_new_tokens: int = 100,
    num_warmup: int = 5,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """Benchmark model latency for autoregressive generation.

    Args:
        model: HuggingFace model.
        prompt_length: Length of synthetic prompt.
        num_new_tokens: How many tokens to generate.
        num_warmup: Warmup iterations.
        batch_size: Batch size.
        device: Device.

    Returns:
        dict with latency metrics.
    """
    vocab_size = model.config.vocab_size
    prompt = torch.randint(
        0, vocab_size, (batch_size, prompt_length), device=device
    )

    model.eval()
    model.to(device)

    # Warmup
    for _ in range(num_warmup):
        _ = model(prompt[:, :10])

    # Benchmark
    input_ids = prompt[:, :1]
    times = []

    for _ in range(num_new_tokens):
        start = time.time()
        outputs = model(input_ids)
        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    total = sum(times)
    return {
        "tokens_per_second": num_new_tokens / total,
        "ms_per_token": 1000 * total / num_new_tokens,
        "total_time_s": total,
        "prompt_length": prompt_length,
        "num_new_tokens": num_new_tokens,
        "batch_size": batch_size,
    }
