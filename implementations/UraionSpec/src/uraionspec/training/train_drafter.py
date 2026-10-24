"""
DSpark drafter training loop.

Freezes the target model and trains only the drafter backbone,
sequential head, and confidence head.
"""

import math
import os
import time
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from ..models import DSparkDraftModel
from .losses import compute_dspark_loss
from ..utils import setup_logger

logger = setup_logger(__name__)


class DSparkTrainer:
    """Trainer for DSpark draft model.

    Freezes target model and trains drafter backbone + heads.
    """

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        target_model: PreTrainedModel,
        target_layer_ids: list[int],
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 200,
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        ce_alpha: float = 0.1,
        tv_alpha: float = 0.9,
        conf_alpha: float = 1.0,
        kl_alpha: float = 0.0,
        tv_ramp: Optional[tuple[int, int]] = None,
        num_anchors: int = 64,
        loss_decay_gamma: Optional[float] = None,
        device: str = "cuda",
        log_fn: Optional[Callable[[dict, int], None]] = None,
    ):
        '''
        TV loss : Total variation distance to the target model's distribution
        CE alpha : Cross entropy against the true next tokens
        Conf alpha : binary cross entropy for confidence head
        '''
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_layer_ids = list(target_layer_ids)
        self.device = device
        self.block_size = draft_model.block_size
        self.num_anchors = num_anchors
        self.ce_alpha = ce_alpha
        self.tv_alpha = tv_alpha
        self.conf_alpha = conf_alpha
        self.kl_alpha = kl_alpha
        self.tv_ramp = tv_ramp
        self.global_step = 0
        self.loss_decay_gamma = loss_decay_gamma
        self.max_grad_norm = max_grad_norm
        self.log_fn = log_fn

        self.trainable_params = [p for p in draft_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def current_tv_alpha(self) -> float:
        """tv_alpha, linearly ramped from 0 over [start, end) steps when tv_ramp is set."""
        if self.tv_ramp is None:
            return self.tv_alpha
        start, end = self.tv_ramp
        return self.tv_alpha * min(1.0, max(0.0, (self.global_step - start) / max(1, end - start)))

    def _sample_anchor_blocks(
        self,
        loss_mask: torch.Tensor,
        lengths: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions.

        Anchor p is valid if p >= 1 (non-empty context), the whole block p+1..p+block_size
        lies inside the sequence, and the first target p+1 is a loss token.

        Args:
            loss_mask: [B, seq_len] 1 for trainable target tokens
            lengths: [B] true (unpadded) sequence lengths

        Returns:
            anchor_positions: [B, num_anchors]
            block_keep_mask: [B, num_anchors] bool
        """
        B, seq_len = loss_mask.shape
        device = loss_mask.device
        positions = torch.arange(seq_len, device=device)[None, :]
        next_is_loss = torch.nn.functional.pad(loss_mask[:, 1:], (0, 1)) > 0.5
        valid = (positions >= 1) & (positions + self.block_size < lengths[:, None]) & next_is_loss

        scores = torch.rand(B, seq_len, device=device, generator=generator)
        scores = scores.masked_fill(~valid, -1.0)
        k = min(self.num_anchors, seq_len)
        top = scores.topk(k, dim=1)
        anchors = torch.zeros(B, self.num_anchors, dtype=torch.long, device=device)
        keep = torch.zeros(B, self.num_anchors, dtype=torch.bool, device=device)
        keep[:, :k] = top.values >= 0
        anchors[:, :k] = torch.where(keep[:, :k], top.indices, 0)
        return anchors, keep

    @torch.no_grad()
    def _target_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single full-context target pass.

        Returns:
            target_hidden: [B, L, num_target_layers * d] features for KV injection
            target_logits: [B, N, block_size, V] target logits at positions p+k (predicting p+k+1)
        """
        out = self.target_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states[0] is the embedding output, so layer i's output is at i + 1
        target_hidden = torch.cat([out.hidden_states[i + 1] for i in self.target_layer_ids], dim=-1)
        B, N, G = block_positions.shape
        last = out.last_hidden_state
        gathered = last.gather(1, block_positions.view(B, -1, 1).expand(-1, -1, last.shape[-1]))
        target_logits = self.target_model.lm_head(gathered).view(B, N, G, -1)
        return target_hidden, target_logits

    def _forward_loss(self, batch: dict, generator: Optional[torch.Generator] = None) -> dict:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        B, seq_len = input_ids.shape

        anchors, keep = self._sample_anchor_blocks(loss_mask, attention_mask.sum(dim=1), generator)
        offsets = torch.arange(self.block_size, device=self.device)
        block_positions = anchors[:, :, None] + offsets  # p .. p+γ-1
        target_positions = (block_positions + 1).clamp(max=seq_len - 1)  # p+1 .. p+γ

        def gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return t.gather(1, idx.view(B, -1)).view(B, self.num_anchors, self.block_size)

        block_token_ids = gather(input_ids, block_positions)
        block_target_ids = gather(input_ids, target_positions)
        eval_mask = (gather(loss_mask, target_positions) > 0.5) & keep[:, :, None]

        target_hidden, target_logits = self._target_forward(input_ids, attention_mask, block_positions)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            draft_output = self.draft_model(
                anchor_ids=block_token_ids[:, :, 0],
                anchor_positions=anchors,
                target_hidden_states=target_hidden,
                prev_token_ids=block_token_ids,  # teacher-forced x_{p+k}, never the label
                context_mask=attention_mask,
                return_confidence=self.conf_alpha > 0,
            )

        return compute_dspark_loss(
            draft_logits=draft_output["draft_logits"],
            target_ids=block_target_ids,
            eval_mask=eval_mask,
            target_logits=target_logits,
            confidence_pred=draft_output.get("confidence_logits"),
            ce_alpha=self.ce_alpha,
            tv_alpha=self.current_tv_alpha(),
            conf_alpha=self.conf_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            kl_alpha=self.kl_alpha,
        )

    def train_step(self, batch: dict) -> dict:
        """Single training step.

        Args:
            batch: dict with 'input_ids', 'attention_mask', 'loss_mask' [B, seq_len]

        Returns:
            dict with loss and metrics
        """
        loss_dict = self._forward_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        loss_dict["metrics"]["grad_norm"] = grad_norm.item()
        loss_dict["metrics"]["lr"] = self.scheduler.get_last_lr()[0]
        loss_dict["metrics"]["tv_alpha"] = self.current_tv_alpha()
        return loss_dict

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, seed: int = 1234) -> dict:
        self.draft_model.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        totals: dict = {}
        weight_sum = 0
        for batch in dataloader:
            metrics = self._forward_loss(batch, generator)["metrics"]
            w = metrics["active_tokens"]
            weight_sum += w
            for key, value in metrics.items():
                if isinstance(value, list):
                    prev = totals.get(key, [0.0] * len(value))
                    totals[key] = [a + b * w for a, b in zip(prev, value)]
                else:
                    totals[key] = totals.get(key, 0.0) + value * w
        self.draft_model.train()
        weight_sum = max(weight_sum, 1)
        return {
            k: [x / weight_sum for x in v] if isinstance(v, list) else v / weight_sum
            for k, v in totals.items()
        }

    def save_checkpoint(self, path: str, step: int, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step": step,
            "model_state_dict": self.draft_model.trainable_state_dict(),
            **(extra or {}),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _log(self, prefix: str, metrics: dict, step: int):
        if self.log_fn is None:
            return
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    flat[f"{prefix}/{key}/pos{i}"] = v
            else:
                flat[f"{prefix}/{key}"] = value
        self.log_fn(flat, step)

    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 100,
        log_interval: int = 10,
        eval_dataloader: Optional[DataLoader] = None,
        eval_interval: int = 250,
        save_dir: Optional[str] = None,
        save_interval: int = 500,
        checkpoint_extra: Optional[dict] = None,
    ) -> dict:
        """Run training loop.

        Args:
            dataloader: DataLoader with 'input_ids', 'attention_mask' and 'loss_mask'.
            num_steps: Total training steps.
            log_interval: Steps between console logging (log_fn receives every step).
            eval_dataloader: Optional held-out DataLoader.
            eval_interval: Steps between evaluations.
            save_dir: Optional directory to save checkpoints.
            save_interval: Steps between checkpoints.

        Returns:
            dict with training history.
        """
        self.draft_model.train()
        step = 0
        epoch = 0
        history = {"loss": [], "ce_loss": [], "tv_loss": [], "conf_loss": []}

        while step < num_steps:
            epoch += 1
            for batch in dataloader:
                if step >= num_steps:
                    break

                train_start = time.time()
                loss_dict = self.train_step(batch)
                step_time = time.time() - train_start

                step += 1
                metrics = loss_dict["metrics"]
                metrics["step_time"] = step_time
                metrics["epoch"] = epoch
                metrics["tokens_in_batch"] = int(batch["attention_mask"].sum())
                for key in history:
                    history[key].append(metrics[key])
                self._log("train", metrics, step)

                if step % log_interval == 0 or step == 1:
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"CE: {metrics['ce_loss']:.4f} | "
                        f"TV: {metrics['tv_loss']:.4f} | "
                        f"KL: {metrics['kl_loss']:.4f} | "
                        f"Conf: {metrics['conf_loss']:.4f} | "
                        f"tau: {metrics.get('tau_expected', 0):.3f} | "
                        f"lr: {metrics['lr']:.2e} | "
                        f"{step_time*1000:.0f}ms/step"
                    )

                if eval_dataloader is not None and (step % eval_interval == 0 or step == num_steps):
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log("eval", eval_metrics, step)
                    logger.info(
                        f"[eval] step {step} | Loss: {eval_metrics['loss']:.4f} | "
                        f"CE: {eval_metrics['ce_loss']:.4f} | TV: {eval_metrics['tv_loss']:.4f} | "
                        f"KL: {eval_metrics['kl_loss']:.4f} | "
                        f"CE/pos: {[round(x, 2) for x in eval_metrics.get('per_pos_ce', [])]} | "
                        f"tau: {eval_metrics.get('tau_expected', 0):.3f} | "
                        f"accept/pos: {[round(x, 3) for x in eval_metrics.get('per_pos_accept', [])]}"
                    )

                if save_dir and (step % save_interval == 0 or step == num_steps):
                    self.save_checkpoint(os.path.join(save_dir, "last.pt"), step, checkpoint_extra)

        self.draft_model.eval()
        return history
