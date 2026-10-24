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
    kl_alpha: float = 0.0,
) -> dict:
    """Compute DSpark loss with all three terms.

    Args:
        draft_logits: [B, N, block_size, V] draft model logits
        target_ids: [B, N, block_size] ground truth token ids
        eval_mask: [B, N, block_size] boolean mask for valid positions
        target_logits: [B, N, block_size, V] target model logits (for TV loss)
        confidence_pred: [B, N, block_size] confidence logits (pre-sigmoid)
        confidence_targets: [B, N, block_size] target acceptance rates
            (derived from the TV distance when None and target_logits is given)
        ce_alpha: Cross-entropy loss weight
        tv_alpha: Total variation loss weight
        conf_alpha: Confidence loss weight
        kl_alpha: Forward KL(p_t || p_d) weight; unlike L1 it pushes up under-predicted tokens
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

    weight_den = weight_mask.sum().clamp_min(1.0)

    # ========== 1. Cross-Entropy Loss ==========
    draft_logp = torch.log_softmax(draft_logits.float(), dim=-1)
    ce_per_token = -draft_logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, N, block_size]
    ce_loss = (ce_per_token * weight_mask).sum() / weight_den

    # ========== 2. KL(p_t || p_d) and Total Variation Loss ==========
    # Note: "tv" is the full L1 distance (= 2 * TV), as in the reference implementation.
    tv_loss = ce_loss.new_zeros(())
    kl_loss = ce_loss.new_zeros(())
    tv_distance = None
    if target_logits is not None:
        target_logp = torch.log_softmax(target_logits.float(), dim=-1)
        target_probs = target_logp.exp()
        if kl_alpha > 0:
            kl_per_token = (target_probs * (target_logp - draft_logp)).sum(dim=-1)
            kl_loss = (kl_per_token * weight_mask).sum() / weight_den
        del target_logp
        with torch.set_grad_enabled(torch.is_grad_enabled() and tv_alpha > 0):
            tv_distance = (draft_logp.exp() - target_probs).abs().sum(dim=-1)  # [B, N, block_size]
        del target_probs
        tv_loss = (tv_distance * weight_mask).sum() / weight_den

    # c*_k = 1 - 0.5 * ||p_d - p_t||_1
    if confidence_targets is None and tv_distance is not None:
        confidence_targets = (1.0 - 0.5 * tv_distance.detach()).clamp(0.0, 1.0)

    # ========== 3. Confidence Loss ==========
    conf_loss = ce_loss.new_zeros(())
    if confidence_pred is not None and confidence_targets is not None and conf_alpha > 0:
        conf_errors = F.binary_cross_entropy_with_logits(
            confidence_pred.float(),
            confidence_targets.float(),
            reduction="none",
        )
        conf_loss = (conf_errors * weight_mask).sum() / weight_den

    # Total loss
    total_loss = ce_alpha * ce_loss + tv_alpha * tv_loss + kl_alpha * kl_loss + conf_alpha * conf_loss

    # Metrics
    metrics = {}
    with torch.no_grad():
        metrics["ce_loss"] = ce_loss.item()
        metrics["tv_loss"] = tv_loss.item()
        metrics["kl_loss"] = kl_loss.item()
        metrics["conf_loss"] = conf_loss.item()
        metrics["loss"] = total_loss.item()
        metrics["active_tokens"] = eval_mask.sum().item()
        valid = eval_mask.float()
        per_pos_count = valid.sum(dim=(0, 1)).clamp_min(1.0)
        metrics["per_pos_ce"] = ((ce_per_token.detach() * valid).sum(dim=(0, 1)) / per_pos_count).tolist()
        draft_top1 = draft_logits.argmax(dim=-1)
        metrics["per_pos_acc_gt"] = (((draft_top1 == target_ids).float() * valid).sum(dim=(0, 1)) / per_pos_count).tolist()
        if target_logits is not None:
            accept = confidence_targets.float() * valid
            metrics["per_pos_accept"] = (accept.sum(dim=(0, 1)) / per_pos_count).tolist()
            top1_match = (draft_top1 == target_logits.argmax(dim=-1)).float() * valid
            metrics["per_pos_top1_agree"] = (top1_match.sum(dim=(0, 1)) / per_pos_count).tolist()
            # Teacher-forced expected accepted length per block (excludes the bonus token)
            block_valid = eval_mask[..., 0]
            n_blocks = block_valid.sum().clamp_min(1)
            metrics["tau_expected"] = (torch.cumprod(accept, dim=-1).sum(-1)[block_valid].sum() / n_blocks).item()
            metrics["tau_greedy"] = (torch.cumprod(top1_match, dim=-1).sum(-1)[block_valid].sum() / n_blocks).item()
        if confidence_pred is not None and confidence_targets is not None:
            metrics["conf_pred_mean"] = ((torch.sigmoid(confidence_pred.float()) * valid).sum() / valid.sum().clamp_min(1.0)).item()

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "tv_loss": tv_loss,
        "conf_loss": conf_loss,
        "kl_loss": kl_loss,
        "metrics": metrics,
    }
