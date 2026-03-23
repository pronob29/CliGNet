"""
loss.py
-------
Focal Binary Cross-Entropy loss for multi-class clinical specialty classification.

Focal loss (Lin et al., 2017) addresses class imbalance by down-weighting
easy examples and focusing learning on hard, misclassified ones. Combined
with per-class alpha weighting (inverse frequency), it replaces the SMOTE
oversampling used in the original class project — without introducing data
leakage.

Formula
-------
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

For binary predictions per label:
  FL_k = -sum over batch [
      alpha_k * (1 - p_k)^gamma * y_k * log(p_k)
    + (1 - alpha_k) * p_k^gamma * (1 - y_k) * log(1 - p_k)
  ]

where p_k = sigmoid(logit_k), y_k in {0, 1}, alpha_k is the class weight
for label k, and gamma is the focusing parameter.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy loss with per-class alpha weighting.

    Parameters
    ----------
    gamma         : focusing parameter; higher = more focus on hard examples
                    typical values: 1.0, 2.0, 3.0 (tuned on val macro-F1)
    alpha_weights : (num_classes,) tensor of per-class alpha weights
                    (typically inverse class frequency, normalised to sum=1 or =K)
                    If None, alpha=0.5 (uniform) is used.
    reduction     : 'mean' (default), 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha_weights is not None:
            self.register_buffer("alpha", alpha_weights.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,    # (B, K) raw logits
        targets: torch.Tensor,   # (B, K) one-hot float labels
    ) -> torch.Tensor:
        """Compute focal BCE loss.

        Parameters
        ----------
        logits  : (B, K) — output of model before sigmoid
        targets : (B, K) — float one-hot targets (0.0 or 1.0)

        Returns
        -------
        loss scalar (or per-element tensor if reduction='none')
        """
        B, K = logits.shape
        targets = targets.float()

        # Sigmoid probabilities
        probs = torch.sigmoid(logits)                          # (B, K)

        # Standard BCE (numerically stable via log_sigmoid)
        bce_pos = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                                       # (B, K)

        # Focal weights: (1 - p_t)^gamma where p_t is prob of correct class
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)  # (B, K)
        focal = (1.0 - p_t).pow(self.gamma)                         # (B, K)

        loss  = focal * bce_pos                                     # (B, K)

        # Apply per-class alpha weights
        if self.alpha is not None:
            # alpha shape: (K,) -> broadcast to (B, K)
            alpha_t = self.alpha.unsqueeze(0).expand(B, -1)    # (B, K)
            loss    = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_focal_loss(
    class_weights: Optional[torch.Tensor],
    gamma: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> FocalBCELoss:
    """Convenience factory for FocalBCELoss.

    Parameters
    ----------
    class_weights : (K,) numpy array or tensor of per-class weights.
                    Typically computed as total / (K * count_k).
    gamma         : focusing parameter (default 2.0)
    device        : device to move the loss module to

    Returns
    -------
    FocalBCELoss instance on `device`
    """
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights.to(device)
        logger.info("Focal loss: gamma=%.1f, alpha weights loaded (%d classes)",
                    gamma, len(class_weights))
    else:
        logger.info("Focal loss: gamma=%.1f, no alpha weights (uniform)", gamma)

    return FocalBCELoss(gamma=gamma, alpha_weights=class_weights).to(device)


class StandardBCELoss(nn.Module):
    """Standard BCE with class weights (used in Ablation A2 — no focal)."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("pos_weight", class_weights.float())
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight,
            reduction="mean",
        )
