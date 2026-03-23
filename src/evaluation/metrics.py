"""
metrics.py
----------
Comprehensive evaluation metrics for the CLiGNet multi-class specialty
classification experiment.

All metric functions accept:
    y_true : (N, K) int array — one-hot ground truth
    y_pred : (N, K) int array — one-hot predictions

Primary metrics (ACM paper table):
    macro_f1     : mean F1 across all K labels (unweighted — required for ACM review)
    micro_f1     : pooled TP/FP/FN across all labels
    per_label_f1 : F1 for each individual specialty
    hamming_loss : fraction of incorrectly predicted labels
    accuracy     : exact-match multi-class accuracy
    ece          : Expected Calibration Error (requires probability inputs)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Compute all metrics for one-hot y_true and y_pred.

    Parameters
    ----------
    y_true       : (N, K) int — one-hot ground truth
    y_pred       : (N, K) int — one-hot predictions
    label_names  : list of K specialty name strings (optional, for per-label dict)

    Returns
    -------
    dict with keys:
        macro_f1, micro_f1, accuracy, hamming, per_label_f1 (array),
        per_label_precision (array), per_label_recall (array)
    """
    # Convert one-hot to class indices for some metrics
    y_true_idx = y_true.argmax(axis=1)
    y_pred_idx = y_pred.argmax(axis=1)

    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)
    acc      = accuracy_score(y_true_idx, y_pred_idx)
    h_loss   = hamming_loss(y_true, y_pred)

    # Per-label F1 / precision / recall
    K = y_true.shape[1]
    per_label_f1   = np.zeros(K)
    per_label_prec = np.zeros(K)
    per_label_rec  = np.zeros(K)

    for k in range(K):
        tp = int(((y_pred[:, k] == 1) & (y_true[:, k] == 1)).sum())
        fp = int(((y_pred[:, k] == 1) & (y_true[:, k] == 0)).sum())
        fn = int(((y_pred[:, k] == 0) & (y_true[:, k] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label_f1[k]   = f1
        per_label_prec[k] = prec
        per_label_rec[k]  = rec

    result = {
        "macro_f1":          float(macro_f1),
        "micro_f1":          float(micro_f1),
        "accuracy":          float(acc),
        "hamming_loss":      float(h_loss),
        "per_label_f1":      per_label_f1,
        "per_label_prec":    per_label_prec,
        "per_label_recall":  per_label_rec,
    }

    # Per-label dict with names if provided
    if label_names is not None:
        result["per_label_dict"] = {
            label_names[k]: {
                "f1":        round(float(per_label_f1[k]),   4),
                "precision": round(float(per_label_prec[k]), 4),
                "recall":    round(float(per_label_rec[k]),  4),
            }
            for k in range(K)
        }

    return result


def format_results_table(
    results: Dict[str, Dict],
    label_names: Optional[List[str]] = None,
) -> str:
    """Format a markdown/text results table for all models.

    Parameters
    ----------
    results : dict mapping model_name -> metrics_dict
    label_names : optional (for per-label table)

    Returns
    -------
    formatted string
    """
    lines = []
    lines.append(f"{'Model':<35} {'Macro-F1':>10} {'Micro-F1':>10} {'Accuracy':>10} {'Hamming':>10}")
    lines.append("-" * 80)
    for model_name, m in results.items():
        lines.append(
            f"{model_name:<35} "
            f"{m['macro_f1']:>10.4f} "
            f"{m['micro_f1']:>10.4f} "
            f"{m['accuracy']:>10.4f} "
            f"{m['hamming_loss']:>10.4f}"
        )
    return "\n".join(lines)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute confusion matrix for multi-class predictions.

    Parameters
    ----------
    y_true      : (N, K) one-hot ground truth
    y_pred      : (N, K) one-hot predictions
    label_names : K specialty name strings

    Returns
    -------
    cm         : (K, K) confusion matrix
    label_names : (K,) ordered labels
    """
    y_true_idx = y_true.argmax(axis=1)
    y_pred_idx = y_pred.argmax(axis=1)
    K          = y_true.shape[1]

    if label_names is None:
        label_names = [str(k) for k in range(K)]

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(K)))
    return cm, label_names


def most_confused_pairs(
    cm: np.ndarray,
    label_names: List[str],
    top_n: int = 20,
) -> List[Dict]:
    """Return top-N most confused specialty pairs (off-diagonal counts).

    Parameters
    ----------
    cm          : (K, K) confusion matrix
    label_names : K specialty name strings
    top_n       : number of pairs to return

    Returns
    -------
    list of dicts: [{true, predicted, count, pct_of_true}, ...]
    """
    K      = len(label_names)
    pairs  = []
    for i in range(K):
        row_total = cm[i, :].sum()
        for j in range(K):
            if i != j and cm[i, j] > 0:
                pct = cm[i, j] / row_total * 100 if row_total > 0 else 0.0
                pairs.append({
                    "true":         label_names[i],
                    "predicted":    label_names[j],
                    "count":        int(cm[i, j]),
                    "pct_of_true":  round(pct, 1),
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def f1_by_class_size(
    per_label_f1: np.ndarray,
    train_counts: np.ndarray,
    bins: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute mean F1 by training set class size bin.

    Parameters
    ----------
    per_label_f1 : (K,) per-label F1 scores
    train_counts : (K,) number of training samples per class
    bins         : class size bin boundaries (default: 0, 20, 50, 100, 500, inf)

    Returns
    -------
    dict mapping bin_label -> mean_f1
    """
    if bins is None:
        bins = [0, 20, 50, 100, 500, int(1e9)]

    result = {}
    for i in range(len(bins) - 1):
        lo, hi  = bins[i], bins[i + 1]
        mask    = (train_counts >= lo) & (train_counts < hi)
        if mask.sum() == 0:
            continue
        label   = f"{lo}-{hi if hi < 1e9 else 'inf'}"
        result[label] = float(per_label_f1[mask].mean())

    return result


def f1_by_document_length(
    per_sample_correct: np.ndarray,
    word_counts: np.ndarray,
    bins: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute mean accuracy by document word count bin (for Contribution C4-F3).

    Parameters
    ----------
    per_sample_correct : (N,) bool or 0/1 — correct/incorrect per document
    word_counts        : (N,) document word counts
    bins               : word count bin boundaries

    Returns
    -------
    dict mapping bin_label -> mean_accuracy
    """
    if bins is None:
        bins = [0, 200, 400, 600, 1000, int(1e9)]

    result = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask   = (word_counts >= lo) & (word_counts < hi)
        if mask.sum() == 0:
            continue
        label  = f"{lo}-{hi if hi < 1e9 else 'inf'}"
        result[label] = float(per_sample_correct[mask].mean())

    return result
