"""
scripts/evaluate.py
-------------------
Aggregate evaluation: compare all 8 baselines, run McNemar significance tests,
and produce the ACM paper results table.

Usage
-----
    python scripts/evaluate.py --results results/ \
                                --processed data/processed/ \
                                --out results/

Expects
-------
    results/baselines/B1/predictions.npz
    results/baselines/B2/predictions.npz
    results/baselines/B3/predictions.npz
    results/baselines/B4/predictions.npz
    results/baselines/B5/predictions.npz   (InceptionXML — optional)
    results/clignet/B6/predictions.npz
    results/baselines/B7/predictions.npz
    results/clignet/B8/predictions.npz

Outputs
-------
    results/comparison_table.txt
    results/comparison_table.json
    results/significance_tests.txt
    results/per_label_f1_comparison.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_all_metrics, format_results_table
from src.evaluation.significance import run_all_significance_tests, format_significance_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model ID -> display name and results subdirectory
MODEL_REGISTRY = {
    "B1": {"name": "B1: TF-IDF + LR",              "subdir": "baselines/B1"},
    "B2": {"name": "B2: TF-IDF + SVC",             "subdir": "baselines/B2"},
    "B3": {"name": "B3: ClinicalBERT + OvR",        "subdir": "clignet/B3"},
    "B4": {"name": "B4: BioBERT + OvR",             "subdir": "clignet/B4"},
    "B5": {"name": "B5: InceptionXML",              "subdir": "baselines/B5"},
    "B6": {"name": "B6: CLiGNet (no calib)",        "subdir": "clignet/B6"},
    "B7": {"name": "B7: Longformer + BR",           "subdir": "clignet/B7"},
    "B8": {"name": "B8: CLiGNet Full",              "subdir": "clignet/B8"},
}
REFERENCE_MODEL = "B8"


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate evaluation and significance tests")
    p.add_argument("--results",   default="results/",      help="Root results directory")
    p.add_argument("--processed", default="data/processed/", help="Processed data directory")
    p.add_argument("--out",       default="results/",      help="Output directory")
    return p.parse_args()


def load_model_predictions(results_root: Path, model_id: str):
    """Load predictions.npz for a model. Returns (test_preds, test_true) or None."""
    subdir = MODEL_REGISTRY[model_id]["subdir"]
    path   = results_root / subdir / "predictions.npz"
    if not path.exists():
        logger.warning("Predictions not found for %s at %s — skipping.", model_id, path)
        return None, None
    data = np.load(str(path))
    return data["test_preds"], data["test_true"]


def main():
    args    = parse_args()
    results = Path(args.results)
    out     = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc          = Path(args.processed)
    label_classes = np.load(str(proc / "label_classes.npy"), allow_pickle=True).tolist()

    # ---- Load all predictions ----
    all_metrics     = {}
    all_preds       = {}
    test_true_ref   = None

    for model_id, cfg in MODEL_REGISTRY.items():
        test_preds, test_true = load_model_predictions(results, model_id)
        if test_preds is None:
            continue

        if test_true_ref is None:
            test_true_ref = test_true
        else:
            assert np.array_equal(test_true, test_true_ref), \
                f"Ground truth mismatch for {model_id}"

        metrics = compute_all_metrics(test_true, test_preds, label_classes)
        all_metrics[cfg["name"]] = metrics
        all_preds[cfg["name"]]   = test_preds
        logger.info("%s: macro_f1=%.4f | micro_f1=%.4f | accuracy=%.4f",
                    cfg["name"], metrics["macro_f1"], metrics["micro_f1"], metrics["accuracy"])

    if not all_metrics:
        logger.error("No model predictions found. Run training scripts first.")
        sys.exit(1)

    # ---- Comparison table ----
    table_str = format_results_table(all_metrics, label_classes)
    logger.info("\n\nRESULTS TABLE\n%s", table_str)

    with open(out / "comparison_table.txt", "w") as f:
        f.write(table_str)

    # JSON-serialisable version
    table_json = {
        name: {
            "macro_f1":     round(m["macro_f1"],   4),
            "micro_f1":     round(m["micro_f1"],   4),
            "accuracy":     round(m["accuracy"],   4),
            "hamming_loss": round(m["hamming_loss"], 5),
        }
        for name, m in all_metrics.items()
    }
    with open(out / "comparison_table.json", "w") as f:
        json.dump(table_json, f, indent=2)
    logger.info("Comparison table saved.")

    # ---- McNemar significance tests ----
    b8_name = MODEL_REGISTRY[REFERENCE_MODEL]["name"]
    if b8_name in all_preds and len(all_preds) > 1:
        sig_results = run_all_significance_tests(
            y_true=test_true_ref,
            model_predictions=all_preds,
            reference_model=b8_name,
            alpha=0.05,
        )
        sig_str = format_significance_table(sig_results, reference_model=b8_name)
        logger.info("\n\nSIGNIFICANCE TESTS\n%s", sig_str)

        with open(out / "significance_tests.txt", "w") as f:
            f.write(sig_str)

        sig_json = {
            model: {
                k: v for k, v in res.items()
                if k not in ("n00", "n11")  # omit concordant pairs
            }
            for model, res in sig_results.items()
        }
        with open(out / "significance_tests.json", "w") as f:
            json.dump(sig_json, f, indent=2)
        logger.info("Significance tests saved.")
    else:
        logger.warning("Reference model %s not found or only one model loaded — skipping significance.", b8_name)

    # ---- Per-label F1 comparison figure ----
    _plot_per_label_f1(all_metrics, label_classes, out)

    logger.info("Evaluation complete. Outputs in: %s", str(out))


def _plot_per_label_f1(all_metrics, label_classes, out):
    """Bar chart: per-label F1 for all models with CLiGNet Full highlighted."""
    import matplotlib.pyplot as plt
    K = len(label_classes)

    # Only plot models that have per_label_f1
    models_to_plot = {
        name: m["per_label_f1"] for name, m in all_metrics.items()
        if "per_label_f1" in m
    }
    if len(models_to_plot) < 2:
        return

    # Sort by CLiGNet F1
    b8_name = MODEL_REGISTRY[REFERENCE_MODEL]["name"]
    if b8_name in models_to_plot:
        sort_key = models_to_plot[b8_name]
    else:
        sort_key = list(models_to_plot.values())[0]

    sort_idx = np.argsort(sort_key)
    x        = np.arange(K)
    width    = 0.8 / len(models_to_plot)

    fig, ax = plt.subplots(figsize=(20, 7))
    colours = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))

    for i, (model_name, f1_arr) in enumerate(models_to_plot.items()):
        alpha = 1.0 if model_name == b8_name else 0.55
        lw    = 1.5 if model_name == b8_name else 0.5
        ax.bar(x + i * width, f1_arr[sort_idx], width, label=model_name,
               color=colours[i], alpha=alpha, edgecolor="none", linewidth=lw)

    ax.set_xticks(x + width * len(models_to_plot) / 2)
    ax.set_xticklabels([label_classes[i] for i in sort_idx], rotation=90, fontsize=6)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-label F1: All Models vs. CLiGNet Full (sorted by B8 F1)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(str(out / "per_label_f1_comparison.png"), dpi=150)
    plt.close(fig)
    logger.info("Per-label F1 comparison figure saved.")


if __name__ == "__main__":
    main()
