"""HuggingFace model loading utilities."""

import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_hf_token() -> str:
    """Get HuggingFace token from env file or environment variable."""
    # Try loading from known .env paths
    env_candidates = [
        os.path.expanduser("~/.env"),
        "/mnt/bup/Projects/uraion-agent-small/model/.env",
        os.path.expanduser("~/.huggingface/token"),
    ]
    for env_path in env_candidates:
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except (OSError, PermissionError):
                pass
    return os.environ.get("HF_TOKEN", "")


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_cache: bool = True,
    token: Optional[str] = None,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model ID or local path.
        device_map: Device map strategy.
        torch_dtype: Torch dtype for model weights.
        use_cache: Enable KV cache.
        token: HF token (auto-detected if None).

    Returns:
        Tuple of (model, tokenizer).
    """
    if token is None:
        token = get_hf_token()
    if not token:
        token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_cache=use_cache,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return model, tokenizer


def get_model_config(model: PreTrainedModel) -> dict:
    """Extract relevant configuration from a model."""
    cfg = model.config
    return {
        "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "d_model", None)),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", None)),
        "vocab_size": cfg.vocab_size,
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "model_type": getattr(cfg, "model_type", "unknown"),
    }
