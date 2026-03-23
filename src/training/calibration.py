"""
calibration.py
--------------
Per-label Platt scaling for post-hoc probability calibration.

After training CLiGNet (B8) or any BERT baseline, the raw sigmoid probabilities
are often miscalibrated, especially for rare classes. Platt scaling learns a
scalar temperature T_k per label k from the validation set, then transforms:

    p_calibrated_k = sigmoid(logit_k / T_k)

This improves Expected Calibration Error (ECE) and optimises per-label
decision thresholds simultaneously.

Usage
-----
    calibrator = LabelCalibrator(num_labels=40)
    calibrator.fit(val_logits, val_labels)   # val_logits: (N, 40), val_labels: (N, 40)
    cal_probs = calibrator.calibrate(test_logits)
    thresholds = calibrator.optimal_thresholds   # (40,) per-label thresholds
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

logger = logging.getLogger(__name__)


class LabelCalibrator:
    """Per-label Platt scaling calibration for CLiGNet.

    Learns one temperature parameter T_k per label by minimising NLL on the
    validation set. Also computes per-label optimal classification thresholds
    by grid search on val macro-F1.

    Parameters
    ----------
    num_labels : number of specialty classes
    max_iter   : maximum L-BFGS iterations for temperature fitting
    """

    def __init__(self, num_labels: int, max_iter: int = 50):
        self.num_labels = num_labels
        self.max_iter   = max_iter
        self.temperatures:      Optional[np.ndarray] = None   # (K,)
        self.optimal_thresholds: Optional[np.ndarray] = None  # (K,)

    def fit(
        self,
        val_logits: np.ndarray,    # (N, K) raw logits from model
        val_labels: np.ndarray,    # (N, K) one-hot ground truth (int or float)
    ) -> "LabelCalibrator":
        """Fit per-label temperature parameters on validation set.

        Uses L-BFGS to minimise per-label binary cross-entropy.
        """
        N, K = val_logits.shape
        assert K == self.num_labels, f"Expected {self.num_labels} labels, got {K}"

        temperatures = np.ones(K, dtype=np.float32)

        for k in range(K):
            logits_k = torch.tensor(val_logits[:, k], dtype=torch.float32)
            labels_k = torch.tensor(val_labels[:, k], dtype=torch.float32)

            # T_k > 0 via exp parameterisation
            log_T = nn.Parameter(torch.zeros(1))
            optim = LBFGS([log_T], lr=0.01, max_iter=self.max_iter, line_search_fn="strong_wolfe")

            def closure():
                optim.zero_grad()
                T     = log_T.exp()
                cal_logits = logits_k / T
                loss  = F.binary_cross_entropy_with_logits(cal_logits, labels_k)
                loss.backward()
                return loss

            optim.step(closure)
            temperatures[k] = log_T.exp().item()

        self.temperatures = temperatures
        logger.info("Calibration: temperatures fitted. Mean T=%.4f, std=%.4f",
                    temperatures.mean(), temperatures.std())

        # Grid search for per-label optimal thresholds
        self.optimal_thresholds = self._find_optimal_thresholds(val_logits, val_labels)
        return self

    def _find_optimal_thresholds(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-label threshold grid search on validation F1.

        Parameters
        ----------
        val_logits : (N, K) raw logits
        val_labels : (N, K) binary ground truth
        grid       : threshold values to try (default: 0.10 to 0.90 step 0.01)

        Returns
        -------
        thresholds : (K,) optimal threshold per label
        """
        if grid is None:
            grid = np.arange(0.10, 0.91, 0.01)

        cal_probs = self.calibrate(val_logits)   # (N, K)
        K = cal_probs.shape[1]
        thresholds = np.full(K, 0.5)

        for k in range(K):
            best_f1  = -1.0
            best_thr = 0.5
            probs_k  = cal_probs[:, k]
            labels_k = val_labels[:, k]

            for thr in grid:
                preds = (probs_k >= thr).astype(int)
                tp = int(((preds == 1) & (labels_k == 1)).sum())
                fp = int(((preds == 1) & (labels_k == 0)).sum())
                fn = int(((preds == 0) & (labels_k == 1)).sum())
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1  = f1
                    best_thr = thr

            thresholds[k] = best_thr

        logger.info("Optimal thresholds: mean=%.3f, min=%.3f, max=%.3f",
                    thresholds.mean(), thresholds.min(), thresholds.max())
        return thresholds

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw logits.

        Parameters
        ----------
        logits : (N, K) raw logits

        Returns
        -------
        probs : (N, K) calibrated probabilities
        """
        if self.temperatures is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        # Divide logits by per-label temperature and apply sigmoid
        cal_logits = logits / self.temperatures[None, :]      # (N, K) broadcast
        probs      = 1.0 / (1.0 + np.exp(-cal_logits))       # sigmoid
        return probs

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate and apply per-label optimal thresholds.

        Returns
        -------
        predictions : (N, K) binary predictions
        """
        if self.optimal_thresholds is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        probs = self.calibrate(logits)
        return (probs >= self.optimal_thresholds[None, :]).astype(int)

    def save(self, path: str) -> None:
        """Save calibration parameters to .npz file."""
        np.savez(
            path,
            temperatures=self.temperatures,
            optimal_thresholds=self.optimal_thresholds,
        )
        logger.info("Calibrator saved to %s", path)

    def load(self, path: str) -> "LabelCalibrator":
        """Load calibration parameters from .npz file."""
        data = np.load(path)
        self.temperatures       = data["temperatures"]
        self.optimal_thresholds = data["optimal_thresholds"]
        logger.info("Calibrator loaded from %s", path)
        return self


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Parameters
    ----------
    probs  : (N, K) predicted probabilities
    labels : (N, K) binary ground truth
    n_bins : number of probability bins (default 15)

    Returns
    -------
    ece : scalar float
    """
    probs_flat  = probs.flatten()
    labels_flat = labels.flatten()
    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    ece         = 0.0
    total       = len(probs_flat)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        conf     = probs_flat[mask].mean()
        accuracy = labels_flat[mask].mean()
        ece     += (mask.sum() / total) * abs(conf - accuracy)

    return float(ece)
