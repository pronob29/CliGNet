"""
significance.py
---------------
Statistical significance testing for the CLiGNet experiment.

Method: McNemar's test (with continuity correction) on paired per-document
correct/incorrect predictions, with Bonferroni correction for multiple
comparisons.

Required for ACM paper: all CLiGNet-Full (B8) vs. baseline comparisons
must report p-values with Bonferroni correction.
    - 7 comparisons (B8 vs B1, B2, B3, B4, B5, B6, B7)
    - alpha_corrected = 0.05 / 7 = 0.0071

Effect size: reported as Cohen's kappa.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true:   np.ndarray,
    preds_a:  np.ndarray,
    preds_b:  np.ndarray,
    use_continuity_correction: bool = True,
) -> Dict:
    """McNemar's test between two classifiers on the same test set.

    Compares classifier A (e.g., CLiGNet-Full B8) against classifier B
    (a baseline).

    Contingency table:
        n00 : both wrong
        n01 : A wrong, B correct
        n10 : A correct, B wrong
        n11 : both correct

    H0: n01 == n10 (both classifiers perform equivalently)

    Parameters
    ----------
    y_true  : (N,) or (N, K) ground truth (argmax taken if 2D)
    preds_a : (N,) or (N, K) predictions from model A (argmax taken if 2D)
    preds_b : (N,) or (N, K) predictions from model B

    Returns
    -------
    dict with: n01, n10, chi2_stat, p_value, reject_h0, cohen_kappa, note
    """
    # Flatten to 1D class indices if needed
    def _to_idx(arr):
        if arr.ndim == 2:
            return arr.argmax(axis=1)
        return arr.flatten()

    y_true_idx  = _to_idx(y_true)
    preds_a_idx = _to_idx(preds_a)
    preds_b_idx = _to_idx(preds_b)

    correct_a = (preds_a_idx == y_true_idx).astype(int)  # (N,)
    correct_b = (preds_b_idx == y_true_idx).astype(int)  # (N,)

    n01 = int(((correct_a == 0) & (correct_b == 1)).sum())  # A wrong, B correct
    n10 = int(((correct_a == 1) & (correct_b == 0)).sum())  # A correct, B wrong
    n00 = int(((correct_a == 0) & (correct_b == 0)).sum())
    n11 = int(((correct_a == 1) & (correct_b == 1)).sum())

    # Chi-squared statistic with continuity correction
    if n01 + n10 == 0:
        chi2_stat = 0.0
        p_value   = 1.0
    else:
        if use_continuity_correction:
            chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        else:
            chi2_stat = (n01 - n10) ** 2 / (n01 + n10)
        p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))

    # Cohen's kappa between A and B (as agreement measure)
    N   = len(y_true_idx)
    p_o = (n00 + n11) / N   # observed agreement
    # Expected agreement under independence
    p_a_pos = (n10 + n11) / N
    p_b_pos = (n01 + n11) / N
    p_e = (p_a_pos * p_b_pos) + ((1 - p_a_pos) * (1 - p_b_pos))
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    return {
        "n00":        n00,
        "n01":        n01,
        "n10":        n10,
        "n11":        n11,
        "chi2_stat":  round(chi2_stat, 4),
        "p_value":    round(p_value, 6),
        "cohen_kappa": round(kappa, 4),
    }


def run_all_significance_tests(
    y_true:             np.ndarray,
    model_predictions:  Dict[str, np.ndarray],
    reference_model:    str = "B8_CLiGNet_Full",
    alpha:              float = 0.05,
) -> Dict[str, Dict]:
    """Run McNemar tests: reference_model vs. all other models.

    Applies Bonferroni correction automatically based on number of comparisons.

    Parameters
    ----------
    y_true             : (N,) or (N, K) ground truth
    model_predictions  : dict mapping model_name -> predictions array
    reference_model    : name of the reference model (usually B8)
    alpha              : family-wise error rate (default 0.05)

    Returns
    -------
    results : dict mapping model_name -> mcnemar_result_dict
              (includes 'reject_h0' key with Bonferroni-corrected decision)
    """
    if reference_model not in model_predictions:
        raise KeyError(f"Reference model '{reference_model}' not in model_predictions")

    preds_ref  = model_predictions[reference_model]
    comparisons = {k: v for k, v in model_predictions.items() if k != reference_model}
    n_comparisons  = len(comparisons)
    alpha_corrected = alpha / n_comparisons

    logger.info(
        "McNemar tests: %s vs %d baselines | Bonferroni alpha=%.4f (%.4f / %d)",
        reference_model, n_comparisons, alpha_corrected, alpha, n_comparisons
    )

    results = {}
    for model_name, preds_b in comparisons.items():
        res = mcnemar_test(y_true, preds_ref, preds_b)
        res["reject_h0"]       = res["p_value"] < alpha_corrected
        res["alpha_corrected"] = round(alpha_corrected, 4)
        res["n_comparisons"]   = n_comparisons
        results[model_name]    = res

        status = "SIGNIFICANT" if res["reject_h0"] else "not significant"
        logger.info(
            "  %s vs %s | chi2=%.3f | p=%.4f | kappa=%.4f | %s",
            reference_model, model_name,
            res["chi2_stat"], res["p_value"], res["cohen_kappa"], status
        )

    return results


def format_significance_table(
    significance_results: Dict[str, Dict],
    reference_model: str = "B8_CLiGNet_Full",
) -> str:
    """Format a text table of McNemar test results.

    Returns
    -------
    formatted string
    """
    lines = [
        f"McNemar Test Results: {reference_model} vs. Baselines",
        f"Bonferroni correction: alpha / {list(significance_results.values())[0]['n_comparisons']}",
        "",
        f"{'Baseline':<35} {'chi2':>8} {'p-value':>10} {'kappa':>8} {'n01':>6} {'n10':>6} {'sig?':>8}",
        "-" * 85,
    ]
    for model_name, res in significance_results.items():
        sig = "***" if res["reject_h0"] else "ns"
        lines.append(
            f"{model_name:<35} "
            f"{res['chi2_stat']:>8.3f} "
            f"{res['p_value']:>10.6f} "
            f"{res['cohen_kappa']:>8.4f} "
            f"{res['n01']:>6d} "
            f"{res['n10']:>6d} "
            f"{sig:>8}"
        )
    return "\n".join(lines)
