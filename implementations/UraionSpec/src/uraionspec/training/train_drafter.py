"""
DSpark drafter training loop.

Freezes the target model and trains only the drafter backbone,
sequential head, and confidence head.
"""

import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

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
        target_model: PreTrainedModel, # pretrained qwen3 4b model 
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 1e-4,
        ce_alpha: float = 0.1,
        tv_alpha: float = 0.9,
        conf_alpha: float = 1.0,
        block_size: int = 7, # default is 7 speculative positions 
        num_anchors: int = 4, # 4 positions from need to get this from a sequence 
        loss_decay_gamma: Optional[float] = None,
        device: str = "cuda",
    ):
        '''
        TV loss : Total variation distance to the target model's distribution
        CE alpha : Cross entropy against the true next tokens
        Conf alpha : binary cross entropy for confidence head
         
        '''
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.eval() # freeze
        for param in self.target_model.parameters():
            param.requires_grad = False 

        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.ce_alpha = ce_alpha
        self.tv_alpha = tv_alpha
        self.conf_alpha = conf_alpha
        self.loss_decay_gamma = loss_decay_gamma

        self.optimizer = torch.optim.AdamW(
            draft_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    def _sample_anchor_blocks(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions and build training blocks.

        Args:
            input_ids: [B, seq_len]
            loss_mask: [B, seq_len] float mask 
            loss_mask : Per token flag of ones and zeros, says which positions in a sequence are trainable 
            one is for real token 
            zero is for padding, first token 

        Returns:
            anchor_positions: [B, num_anchors]
            block_keep_mask: [B, num_anchors] bool
        """
        B, seq_len = input_ids.shape
        N = self.num_anchors
        device = input_ids.device

        # Valid anchors: positions where we can predict at least one token
        valid_last = seq_len - self.block_size
        if valid_last <= 0: # if none valid we return 0,0 tensors 
            anchors = torch.zeros(B, N, dtype=torch.long, device=device)
            keep = torch.zeros(B, N, dtype=torch.bool, device=device)
            return anchors, keep

        # Sample random anchor positions
        # Each position p is valid if loss_mask[p] > 0.5 and loss_mask[p+1] > 0.5
        # TODO : why not checking rest ? like p+2, p+3 ..
        anchor_valid = loss_mask[:, :valid_last] > 0.5 
        next_valid = loss_mask[:, 1:valid_last + 1] > 0.5 # Assumption here is self.block_size > 1 
        valid = anchor_valid & next_valid

        # For each sample, pick random valid positions
        anchors = torch.zeros(B, N, dtype=torch.long, device=device)
        keep = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            valid_positions = torch.where(valid[b])[0] # length of valid ones
            n_valid = valid_positions.shape[0] # length
            if n_valid > 0:
                n_take = min(n_valid, N) # 10 valid , need 7 so -> 7 
                perm = torch.randperm(n_valid, device=device)# random 7 values upto 7-1
                chosen = valid_positions[perm[:n_take]]
                anchors[b, :n_take] = chosen
                keep[b, :n_take] = True

        return anchors, keep

    def train_step(self, batch: dict) -> dict:
        """Single training step.

        Args:
            batch: dict with 'input_ids' [B, seq_len], 'loss_mask' [B, seq_len]

        Returns:
            dict with loss and metrics
        """
        input_ids = batch["input_ids"].to(self.device)
        loss_mask = batch.get("loss_mask",
            torch.ones_like(input_ids, dtype=torch.float32)
        ).to(self.device) # TODO : fallback is take all, which makes padding valid and all positions becomes a possible anchor here  
        B, seq_len = input_ids.shape
        device = input_ids.device

        # Sample anchor blocks
        anchor_positions, block_keep_mask = self._sample_anchor_blocks(
            input_ids, loss_mask
        )
        N = self.num_anchors

        # Build training blocks
        # For each anchor, create block of tokens [anchor, anchor+1, ..., anchor+block_size]
        block_token_ids = []
        block_target_ids = []
        block_eval_mask = []

        for b in range(B):
            for n in range(N):
                pos = anchor_positions[b, n].item()
                if pos > 0 and pos + self.block_size <= seq_len:
                    # Token ids: anchor + block_size mask tokens for input
                    # Target ids: ground truth tokens at positions pos+1 .. pos+block_size
                    token_ids = input_ids[b, pos: pos + self.block_size]  # anchor + block_size-1 future tokens
                    target_ids = input_ids[b, pos + 1: pos + self.block_size + 1]  # next tokens

                    block_token_ids.append(token_ids)
                    block_target_ids.append(target_ids)
                    block_eval_mask.append(True) # tells if this block is real and should count in loss
                else:
                    block_token_ids.append(torch.zeros(self.block_size, dtype=torch.long, device=device))
                    block_target_ids.append(torch.zeros(self.block_size, dtype=torch.long, device=device))
                    block_eval_mask.append(False)

        block_token_ids = torch.stack(block_token_ids).view(B, N, self.block_size)
        block_target_ids = torch.stack(block_target_ids).view(B, N, self.block_size)
        eval_mask = torch.tensor(block_eval_mask, device=device).view(B, N)
        # For eval_mask, we need [B, N, block_size] for losses
        eval_mask_3d = eval_mask.unsqueeze(-1).expand(-1, -1, self.block_size).bool()

        # Forward through draft model
        draft_output = self.draft_model(
            block_token_ids.view(B * N, self.block_size),
            prev_token_ids=block_target_ids.view(B * N, self.block_size),  # previous tokens for seq head
            return_confidence=self.conf_alpha > 0,
        )

        draft_logits = draft_output["draft_logits"].view(B, N, self.block_size, -1) # logits from the main model here 

        # Forward through target model for target logits (for TV loss)
        target_logits = None
        if self.tv_alpha > 0 and self.target_model is not None:
            with torch.no_grad():
                flat_input = block_token_ids.view(B * N, self.block_size)
                target_out = self.target_model(flat_input)
                target_logits = target_out.logits.view(B, N, self.block_size, -1)

        # Confidence targets
        confidence_pred = None
        confidence_targets = None
        if self.conf_alpha > 0 and "confidence_logits" in draft_output:
            confidence_pred = draft_output["confidence_logits"].view(B, N, self.block_size)
            if target_logits is not None:
                from ..models.confidence_head import compute_accept_rate
                confidence_targets = compute_accept_rate(
                    draft_logits.float(), target_logits.float()
                ).detach() # compare logits for these 2 models here , draft and target model and see how they are performing here 

        # Compute loss
        loss_dict = compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=block_target_ids,
            eval_mask=eval_mask_3d,
            target_logits=target_logits,
            confidence_pred=confidence_pred,
            confidence_targets=confidence_targets,
            ce_alpha=self.ce_alpha,
            tv_alpha=self.tv_alpha,
            conf_alpha=self.conf_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_dict

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        num_steps: int = 100,
        log_interval: int = 10,
        save_dir: Optional[str] = None,
    ) -> dict:
        """Run training loop.

        Args:
            dataloader: DataLoader with 'input_ids' and optional 'loss_mask'.
            num_epochs: Number of epochs (ignored if dataloader is very small).
            num_steps: Total training steps.
            log_interval: Steps between logging.
            save_dir: Optional directory to save checkpoints.

        Returns:
            dict with training history.
        """
        self.draft_model.train()
        step = 0
        epoch = 0
        history = {"loss": [], "ce_loss": [], "tv_loss": [], "conf_loss": []}
        best_loss = float("inf")

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
                history["loss"].append(metrics["loss"])
                history["ce_loss"].append(metrics["ce_loss"])
                history["tv_loss"].append(metrics["tv_loss"])
                history["conf_loss"].append(metrics["conf_loss"])

                if step % log_interval == 0:
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"CE: {metrics['ce_loss']:.4f} | "
                        f"TV: {metrics['tv_loss']:.4f} | "
                        f"Conf: {metrics['conf_loss']:.4f} | "
                        f"Active: {metrics.get('active_tokens', 0)} | "
                        f"{step_time*1000:.1f}ms/step"
                    )

                # Save checkpoint
                if save_dir and metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    os.makedirs(save_dir, exist_ok=True)
                    ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": self.draft_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": metrics["loss"],
                    }, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

        self.draft_model.eval()
        return history
