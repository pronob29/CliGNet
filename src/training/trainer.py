"""
trainer.py
----------
Training loop for CLiGNet and BERT baselines (B3, B4, B7).

Features
--------
- AdamW with separate learning rates for BERT (2e-5) and other params (1e-3)
- Cosine annealing with linear warmup (10% of total steps)
- Early stopping on validation macro-F1 (patience=5)
- Gradient accumulation (effective_batch = batch_size * grad_accum_steps)
- Checkpoint save / resume
- Comprehensive per-epoch logging
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Cosine annealing schedule with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Unified trainer for CLiGNet and BERT baselines.

    Parameters
    ----------
    model              : CLiGNet or BERTClassifier
    train_loader       : DataLoader for training split
    val_loader         : DataLoader for validation split
    loss_fn            : FocalBCELoss or StandardBCELoss
    num_labels         : number of specialty classes
    output_dir         : directory for checkpoints and logs
    bert_lr            : learning rate for BERT encoder params (default 2e-5)
    head_lr            : learning rate for GCN + classification head (default 1e-3)
    weight_decay       : AdamW weight decay (default 0.01)
    max_epochs         : maximum training epochs (default 30)
    grad_accum_steps   : gradient accumulation steps (default 2)
    patience           : early stopping patience on val macro-F1 (default 5)
    device             : torch device
    seed               : random seed
    is_bert_baseline   : if True, use a single LR for all params (bert_lr)
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        num_labels: int,
        output_dir: str,
        bert_lr: float = 2e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 30,
        grad_accum_steps: int = 2,
        patience: int = 5,
        device: torch.device = None,
        seed: int = 42,
        is_bert_baseline: bool = False,
    ):
        self.model            = model
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.loss_fn          = loss_fn
        self.num_labels       = num_labels
        self.output_dir       = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bert_lr          = bert_lr
        self.head_lr          = head_lr
        self.weight_decay     = weight_decay
        self.max_epochs       = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.patience         = patience
        self.device           = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed             = seed
        self.is_bert_baseline = is_bert_baseline

        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.num_training_steps = (
            len(self.train_loader) // self.grad_accum_steps * self.max_epochs
        )
        num_warmup_steps = int(0.10 * self.num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps, self.num_training_steps
        )

        self.best_val_macro_f1 = -1.0
        self.best_epoch        = 0
        self.no_improve_count  = 0
        self.history: List[Dict] = []

    def _build_optimizer(self):
        """Build AdamW with separate LRs for BERT vs. head parameters."""
        if self.is_bert_baseline:
            return AdamW(self.model.parameters(), lr=self.bert_lr,
                         weight_decay=self.weight_decay)

        # Separate BERT params from GCN + head params
        bert_params = []
        head_params = []
        bert_param_names = []
        head_param_names = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bert" in name or "encoder" in name:
                bert_params.append(param)
                bert_param_names.append(name)
            else:
                head_params.append(param)
                head_param_names.append(name)

        logger.info("Optimizer: %d BERT params (lr=%.2e), %d head params (lr=%.2e)",
                    len(bert_params), self.bert_lr, len(head_params), self.head_lr)

        return AdamW(
            [
                {"params": bert_params, "lr": self.bert_lr},
                {"params": head_params, "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean training loss."""
        self.model.train()
        total_loss   = 0.0
        step_count   = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            num_chunks     = batch["num_chunks"].to(self.device)
            labels         = batch["label"].to(self.device)   # (B,) integer class

            # Convert to one-hot for focal BCE
            one_hot = torch.zeros(labels.size(0), self.num_labels, device=self.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)     # (B, K)

            logits = self.model(input_ids, attention_mask, num_chunks)
            loss   = self.loss_fn(logits, one_hot) / self.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step_count += 1

        mean_loss = total_loss / len(self.train_loader)
        logger.info("Epoch %d | train_loss=%.4f", epoch, mean_loss)
        return mean_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Run evaluation on a DataLoader.

        Returns
        -------
        metrics     : dict with macro_f1, micro_f1, etc.
        all_preds   : (N, K) binary predictions (argmax one-hot)
        all_labels  : (N, K) one-hot ground truth
        """
        self.model.eval()
        all_logits = []
        all_labels = []

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            num_chunks     = batch["num_chunks"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask, num_chunks)  # (B, K)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)   # (N, K)
        all_labels = np.concatenate(all_labels, axis=0)   # (N,) integer

        # Convert labels to one-hot for metric computation
        N, K = all_logits.shape
        one_hot = np.zeros((N, K), dtype=int)
        one_hot[np.arange(N), all_labels] = 1

        # Convert logits to predictions via argmax (single-label, threshold=0.5)
        probs      = 1.0 / (1.0 + np.exp(-all_logits))    # sigmoid
        preds_hard = np.zeros_like(probs, dtype=int)
        preds_hard[np.arange(N), probs.argmax(axis=1)] = 1

        metrics = compute_all_metrics(one_hot, preds_hard)
        return metrics, preds_hard, one_hot

    def fit(self) -> Dict:
        """Full training loop with early stopping.

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        logger.info("Starting training: max_epochs=%d, patience=%d, device=%s",
                    self.max_epochs, self.patience, self.device)
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)

            val_metrics, _, _ = self.evaluate(self.val_loader)
            val_macro_f1 = val_metrics["macro_f1"]

            record = {
                "epoch":         epoch,
                "train_loss":    round(train_loss, 5),
                "val_macro_f1":  round(val_macro_f1, 5),
                "val_micro_f1":  round(val_metrics["micro_f1"], 5),
                "val_accuracy":  round(val_metrics["accuracy"], 5),
            }
            self.history.append(record)
            logger.info(
                "Epoch %d | val_macro_f1=%.4f | val_micro_f1=%.4f",
                epoch, val_macro_f1, val_metrics["micro_f1"]
            )

            # Checkpoint best model
            if val_macro_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_macro_f1
                self.best_epoch        = epoch
                self.no_improve_count  = 0
                self._save_checkpoint("best_model.pt")
                logger.info("New best val_macro_f1=%.4f at epoch %d", val_macro_f1, epoch)
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs). "
                        "Best epoch: %d, best val_macro_f1: %.4f",
                        epoch, self.patience, self.best_epoch, self.best_val_macro_f1
                    )
                    break

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds.", elapsed)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

        return self.history

    def load_best_model(self) -> None:
        """Load the best checkpoint."""
        self._load_checkpoint("best_model.pt")
        logger.info("Loaded best model from epoch %d (val_macro_f1=%.4f)",
                    self.best_epoch, self.best_val_macro_f1)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        torch.save({
            "model_state_dict":  self.model.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_epoch":        self.best_epoch,
            "history":           self.history,
        }, str(path))

    def _load_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        self.best_epoch        = ckpt.get("best_epoch", 0)
        logger.info("Checkpoint loaded from %s", path)
