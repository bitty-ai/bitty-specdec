"""
DSpark loss functions.

Three-term objective (from DSpark paper, Eq. 12):

    L = α_ce * L_ce + α_tv * L_tv + α_conf * L_conf

with position weighting w_k = exp(-(k-1)/γ) emphasizing earlier positions.

L_ce: Cross-entropy loss for next-token prediction
L_tv: Total Variation distance between draft and target distributions
L_conf: Binary cross-entropy for confidence head, supervised by c*_k
"""

from typing import Optional

import torch
import torch.nn.functional as F


def build_position_weights(
    block_size: int,
    gamma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build position decay weights w_k = exp(-(k-1)/γ).

    Args:
        block_size: Number of draft positions.
        gamma: Decay gamma parameter (typically equals block_size).
        device: Target device.
        dtype: Target dtype.

    Returns:
        weights: [block_size] position weights.
        
    Use: 
        Mistakes early cost more than mistakes later , as one wrong fit and we discrad rest of this thing  
    """
    positions = torch.arange(block_size, device=device, dtype=dtype)
    return torch.exp(-positions / gamma)


def compute_dspark_loss(
    draft_logits: torch.Tensor,
    target_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    target_logits: Optional[torch.Tensor] = None,
    confidence_pred: Optional[torch.Tensor] = None,
    confidence_targets: Optional[torch.Tensor] = None,
    ce_alpha: float = 0.1,
    tv_alpha: float = 0.9,
    conf_alpha: float = 1.0,
    loss_decay_gamma: Optional[float] = None,
) -> dict:
    """Compute DSpark loss with all three terms.

    Args:
        draft_logits: [B, N, block_size, V] draft model logits
        target_ids: [B, N, block_size] ground truth token ids
        eval_mask: [B, N, block_size] boolean mask for valid positions
        target_logits: [B, N, block_size, V] target model logits (for TV loss)
        confidence_pred: [B, N, block_size] confidence logits (pre-sigmoid)
        confidence_targets: [B, N, block_size] target acceptance rates
        ce_alpha: Cross-entropy loss weight
        tv_alpha: Total variation loss weight
        conf_alpha: Confidence loss weight
        loss_decay_gamma: Position decay gamma (default: block_size)

    Returns:
        dict with keys: loss, ce_loss, tv_loss, conf_loss, metrics
    """
    B, N, block_size, V = draft_logits.shape
    device = draft_logits.device

    # Build position weights
    gamma = loss_decay_gamma if loss_decay_gamma is not None else float(block_size)
    pos_weights = build_position_weights(block_size, gamma, device) # position weights

    # Expand weights: [block_size] -> [B, N, block_size]
    weight_mask = pos_weights.view(1, 1, block_size).expand(B, N, block_size)
    weight_mask = weight_mask * eval_mask.to(dtype=torch.float32)

    # ========== 1. Cross-Entropy Loss ==========
    flat_logits = draft_logits.reshape(-1, V)
    flat_targets = target_ids.reshape(-1)
    flat_weights = weight_mask.reshape(-1)

    ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    ce_loss_num = (ce_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum().clamp_min(1.0)
    ce_loss = ce_loss_num / ce_loss_den

    # ========== 2. Total Variation Loss ==========
    tv_loss = ce_loss.new_zeros(())
    if target_logits is not None and tv_alpha > 0:
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        target_probs = torch.softmax(target_logits.float(), dim=-1)
        tv_distance = (draft_probs - target_probs).abs().sum(dim=-1)  # [B, N, block_size]
        tv_num = (tv_distance * weight_mask).sum()
        tv_den = weight_mask.sum().clamp_min(1.0)
        tv_loss = tv_num / tv_den

    # ========== 3. Confidence Loss ==========
    conf_loss = ce_loss.new_zeros(())
    if confidence_pred is not None and confidence_targets is not None and conf_alpha > 0:
        conf_errors = F.binary_cross_entropy_with_logits(
            confidence_pred.float(),
            confidence_targets.float(),
            reduction="none",
        )
        conf_num = (conf_errors * weight_mask).sum()
        conf_den = weight_mask.sum().clamp_min(1.0)
        conf_loss = conf_num / conf_den

    # Total loss
    total_loss = ce_alpha * ce_loss + tv_alpha * tv_loss + conf_alpha * conf_loss

    # Metrics
    metrics = {}
    with torch.no_grad():
        metrics["ce_loss"] = ce_loss.item()
        metrics["tv_loss"] = tv_loss.item()
        metrics["conf_loss"] = conf_loss.item()
        metrics["loss"] = total_loss.item()
        if eval_mask.any():
            metrics["active_tokens"] = eval_mask.sum().item()

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "tv_loss": tv_loss,
        "conf_loss": conf_loss,
        "metrics": metrics,
    }
