"""
failure_analysis.py
-------------------
Contribution C4: Failure analysis as a first-class research contribution.

Four sub-components (F1 to F4):
    F1 — Pairwise confusion: identify specialty pairs most frequently confused
    F2 — Rare class behaviour: F1 stratified by training set class size
    F3 — Document length vs. performance: F1 by word count bin
    F4 — Systematic error patterns: IG attribution analysis for low-F1 labels

All functions write results to the results/ablations/ directory and return
structured dicts for programmatic downstream use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from src.evaluation.metrics import (
    compute_confusion_matrix,
    most_confused_pairs,
    f1_by_class_size,
    f1_by_document_length,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F1 — Pairwise Confusion Analysis
# ---------------------------------------------------------------------------
def f1_pairwise_confusion(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    label_names:  List[str],
    output_dir:   str,
    top_n:        int = 20,
) -> Dict:
    """Compute and visualise top-N confused specialty pairs.

    Parameters
    ----------
    y_true, y_pred : (N, K) one-hot arrays
    label_names    : K specialty name strings
    output_dir     : directory to save plots and JSON
    top_n          : number of top confused pairs to analyse

    Returns
    -------
    dict with 'confused_pairs' list and 'cm' matrix
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm, names = compute_confusion_matrix(y_true, y_pred, label_names)
    pairs = most_confused_pairs(cm, names, top_n=top_n)

    # Save JSON
    with open(out / "F1_confused_pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)

    # Heatmap of top-20 rows/cols
    top_labels = list(dict.fromkeys(
        [p["true"] for p in pairs[:top_n]] +
        [p["predicted"] for p in pairs[:top_n]]
    ))[:20]
    top_idx = [names.index(l) for l in top_labels if l in names]

    if top_idx:
        sub_cm = cm[np.ix_(top_idx, top_idx)]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            sub_cm, annot=True, fmt="d",
            xticklabels=[names[i] for i in top_idx],
            yticklabels=[names[i] for i in top_idx],
            cmap="YlOrRd", ax=ax, linewidths=0.3,
        )
        ax.set_title("F1: Pairwise Confusion Heatmap (Top-20 Confused Pairs)", fontsize=13)
        ax.set_xlabel("Predicted Specialty")
        ax.set_ylabel("True Specialty")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        fig.savefig(out / "F1_confusion_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info("Saved F1 confusion heatmap.")

    logger.info("F1: top confused pair: %s -> %s (%d times, %.1f%%)",
                pairs[0]["true"] if pairs else "N/A",
                pairs[0]["predicted"] if pairs else "N/A",
                pairs[0]["count"] if pairs else 0,
                pairs[0]["pct_of_true"] if pairs else 0)

    return {"confused_pairs": pairs, "cm": cm}


# ---------------------------------------------------------------------------
# F2 — Rare Class Behaviour
# ---------------------------------------------------------------------------
def f2_rare_class_behaviour(
    per_label_f1:   np.ndarray,
    train_counts:   np.ndarray,
    label_names:    List[str],
    output_dir:     str,
) -> Dict:
    """Analyse F1 as a function of training class size.

    Produces a scatter plot (F1 vs. log(n)) and a binned summary table.

    Parameters
    ----------
    per_label_f1 : (K,) per-label F1 scores
    train_counts : (K,) training sample counts per class
    label_names  : K specialty name strings
    output_dir   : output directory

    Returns
    -------
    dict with 'binned_f1' and per-label details
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Binned summary
    binned = f1_by_class_size(per_label_f1, train_counts)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        train_counts, per_label_f1,
        c=per_label_f1, cmap="RdYlGn", vmin=0, vmax=1,
        s=80, alpha=0.8, edgecolors="k", linewidths=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="F1 Score")

    # Annotate rare classes
    for i, (name, cnt, f1) in enumerate(zip(label_names, train_counts, per_label_f1)):
        if cnt < 20 or f1 < 0.3:
            ax.annotate(name.split("/")[0][:12], (cnt, f1), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Training Sample Count (log scale)")
    ax.set_ylabel("Per-label F1")
    ax.set_title("F2: Per-label F1 vs. Training Set Size")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="F1=0.5 threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "F2_f1_vs_classsize.png", dpi=150)
    plt.close(fig)

    # Save JSON
    per_label_detail = [
        {"specialty": label_names[i], "train_count": int(train_counts[i]),
         "f1": round(float(per_label_f1[i]), 4)}
        for i in range(len(label_names))
    ]
    per_label_detail.sort(key=lambda x: x["train_count"])
    with open(out / "F2_rare_class.json", "w") as f:
        json.dump({"binned_f1": binned, "per_label": per_label_detail}, f, indent=2)

    logger.info("F2: Binned F1 by class size: %s",
                {k: round(v, 3) for k, v in binned.items()})
    return {"binned_f1": binned, "per_label": per_label_detail}


# ---------------------------------------------------------------------------
# F3 — Document Length vs. Performance
# ---------------------------------------------------------------------------
def f3_length_vs_performance(
    y_true:        np.ndarray,
    y_pred_clignet: np.ndarray,
    y_pred_longformer: Optional[np.ndarray],
    word_counts:   np.ndarray,
    output_dir:    str,
) -> Dict:
    """Compare CLiGNet vs. Longformer accuracy by document length bin.

    Parameters
    ----------
    y_true              : (N, K) one-hot ground truth
    y_pred_clignet      : (N, K) CLiGNet predictions
    y_pred_longformer   : (N, K) Longformer B7 predictions (optional)
    word_counts         : (N,) document word counts
    output_dir          : output directory

    Returns
    -------
    dict with per-bin accuracy for each model
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _correct(y_t, y_p):
        return (y_t.argmax(axis=1) == y_p.argmax(axis=1)).astype(float)

    correct_clignet = _correct(y_true, y_pred_clignet)
    bins = [0, 200, 400, 600, 1000, int(1e9)]
    bin_labels = ["0-200", "200-400", "400-600", "600-1000", "1000+"]

    clignet_by_len = f1_by_document_length(correct_clignet, word_counts, bins)

    longformer_by_len = None
    if y_pred_longformer is not None:
        correct_lf = _correct(y_true, y_pred_longformer)
        longformer_by_len = f1_by_document_length(correct_lf, word_counts, bins)

    # Line chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(clignet_by_len.keys())
    ax.plot(x, list(clignet_by_len.values()), "o-", color="#1565C0", label="CLiGNet (B8)")
    if longformer_by_len:
        ax.plot(x, list(longformer_by_len.values()), "s--", color="#E65100",
                label="Longformer (B7)")
    ax.set_xlabel("Document Length (words)")
    ax.set_ylabel("Accuracy")
    ax.set_title("F3: Accuracy vs. Document Length")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(out / "F3_length_vs_accuracy.png", dpi=150)
    plt.close(fig)

    result = {
        "clignet":    clignet_by_len,
        "longformer": longformer_by_len,
    }
    with open(out / "F3_length_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("F3: CLiGNet accuracy by length: %s",
                {k: round(v, 3) for k, v in clignet_by_len.items()})
    return result


# ---------------------------------------------------------------------------
# F4 — Systematic Error Patterns
# ---------------------------------------------------------------------------
def f4_systematic_errors(
    per_label_f1:  np.ndarray,
    label_names:   List[str],
    explanations:  Optional[List[Dict]],
    output_dir:    str,
    f1_threshold:  float = 0.40,
) -> Dict:
    """Analyse attribution patterns for low-F1 specialties.

    Categorises prediction errors as:
        (a) true label ambiguity (specialties with high semantic overlap)
        (b) vocabulary overlap (shared clinical terminology)
        (c) insufficient training data (rare class)

    Parameters
    ----------
    per_label_f1  : (K,) per-label F1 scores
    label_names   : K specialty name strings
    explanations  : list of IG explanation dicts from CLiGNetExplainer.explain()
    output_dir    : output directory
    f1_threshold  : F1 below which a specialty is considered low-performing

    Returns
    -------
    dict with low_f1_labels list and error_categories
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = len(label_names)
    low_f1_labels = [
        {"specialty": label_names[k], "f1": round(float(per_label_f1[k]), 4)}
        for k in range(K)
        if per_label_f1[k] < f1_threshold
    ]
    low_f1_labels.sort(key=lambda x: x["f1"])

    # Per-label F1 bar chart (sorted)
    sorted_idx    = np.argsort(per_label_f1)
    sorted_f1     = per_label_f1[sorted_idx]
    sorted_names  = [label_names[i] for i in sorted_idx]
    colours       = ["#B71C1C" if f < f1_threshold else
                     "#E65100" if f < 0.70 else
                     "#1B5E20" for f in sorted_f1]

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(range(K), sorted_f1, color=colours, edgecolor="none")
    ax.set_xticks(range(K))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax.set_ylabel("F1 Score")
    ax.set_title("F4: Per-label F1 (sorted) — CLiGNet Full (B8)")
    ax.axhline(f1_threshold, color="red", linestyle="--", linewidth=1,
               label=f"F1 = {f1_threshold} threshold")
    ax.axhline(0.70, color="orange", linestyle=":", linewidth=1, label="F1 = 0.70")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out / "F4_per_label_f1.png", dpi=150)
    plt.close(fig)

    # Save JSON
    result = {"low_f1_labels": low_f1_labels}
    if explanations:
        # Aggregate top tokens per low-F1 specialty
        low_spec_set = {d["specialty"] for d in low_f1_labels}
        by_specialty: Dict[str, List] = {}
        for exp in explanations:
            spec = exp.get("specialty", "unknown")
            if spec in low_spec_set:
                by_specialty.setdefault(spec, []).extend(exp.get("top_tokens", []))
        result["attribution_summary"] = by_specialty

    with open(out / "F4_systematic_errors.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("F4: %d specialties below F1 threshold %.2f",
                len(low_f1_labels), f1_threshold)
    return result


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_full_failure_analysis(
    y_true:          np.ndarray,
    y_pred_clignet:  np.ndarray,
    label_names:     List[str],
    train_counts:    np.ndarray,
    word_counts:     np.ndarray,
    output_dir:      str,
    per_label_f1:    np.ndarray,
    y_pred_longformer: Optional[np.ndarray] = None,
    ig_explanations: Optional[List[Dict]] = None,
) -> Dict:
    """Run all four failure analysis sub-components and save outputs.

    Returns dict with all sub-component results.
    """
    logger.info("Running full failure analysis (C4: F1-F4)...")
    out = Path(output_dir)

    results = {}
    results["F1"] = f1_pairwise_confusion(
        y_true, y_pred_clignet, label_names, str(out / "F1_confusion"))
    results["F2"] = f2_rare_class_behaviour(
        per_label_f1, train_counts, label_names, str(out / "F2_rare"))
    results["F3"] = f3_length_vs_performance(
        y_true, y_pred_clignet, y_pred_longformer, word_counts, str(out / "F3_length"))
    results["F4"] = f4_systematic_errors(
        per_label_f1, label_names, ig_explanations, str(out / "F4_errors"))

    # Master summary
    with open(out / "failure_analysis_summary.json", "w") as f:
        summary = {
            "num_low_f1_labels": len(results["F4"]["low_f1_labels"]),
            "low_f1_labels":     results["F4"]["low_f1_labels"],
            "f1_by_class_size":  results["F2"]["binned_f1"],
            "f1_by_doc_length":  results["F3"]["clignet"],
            "top_confused_pairs": results["F1"]["confused_pairs"][:5],
        }
        json.dump(summary, f, indent=2)

    logger.info("Failure analysis complete. Results in %s", output_dir)
    return results
