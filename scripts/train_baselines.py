"""
scripts/train_baselines.py
--------------------------
Train classical baselines B1 and B2 (TF-IDF-based).

Usage
-----
    python scripts/train_baselines.py --processed data/processed/ \
                                       --out results/baselines/ \
                                       --models B1 B2

Outputs (per model, in --out/<model_name>/)
-------------------------------------------
    model.pkl           trained pipeline
    val_metrics.json    validation metrics
    test_metrics.json   held-out test metrics
    predictions.npz     val_preds, test_preds (one-hot), val_true, test_true

Note: BERT baselines B3, B4, B7 are trained via train_clignet.py with
--mode baseline --baseline-id B3|B4|B7 to share the same training loop.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines import TFIDF_LR, TFIDF_SVC
from src.evaluation.metrics import compute_all_metrics, format_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train TF-IDF baselines B1, B2")
    p.add_argument("--processed", default="data/processed/",
                   help="Directory with train/val/test CSV files")
    p.add_argument("--out",       default="results/baselines/",
                   help="Output directory")
    p.add_argument("--models",    nargs="+", default=["B1", "B2"],
                   choices=["B1", "B2"],
                   help="Baselines to train")
    return p.parse_args()


def label_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def evaluate_baseline(model, texts, labels_int, label_classes, split_name):
    """Run predict, compute metrics, return (metrics_dict, preds_onehot)."""
    K = len(label_classes)

    # Build one-vs-rest integer label array
    proba      = model.predict_proba(texts)  # (N, K)
    preds_idx  = proba.argmax(axis=1)        # (N,)
    preds_oh   = label_onehot(preds_idx, K)
    labels_oh  = label_onehot(labels_int, K)

    metrics    = compute_all_metrics(labels_oh, preds_oh, label_classes.tolist())
    logger.info(
        "%s | macro_f1=%.4f | micro_f1=%.4f | accuracy=%.4f",
        split_name, metrics["macro_f1"], metrics["micro_f1"], metrics["accuracy"]
    )
    return metrics, preds_oh, labels_oh


def train_baseline(model_id, train_df, val_df, test_df, label_classes, out_dir):
    out = Path(out_dir) / model_id
    out.mkdir(parents=True, exist_ok=True)

    K              = len(label_classes)
    train_texts    = train_df["transcription"].tolist()
    train_labels   = train_df["label"].values
    val_texts      = val_df["transcription"].tolist()
    val_labels     = val_df["label"].values
    test_texts     = test_df["transcription"].tolist()
    test_labels    = test_df["label"].values

    # One-hot for multi-class OvR training
    train_oh = label_onehot(train_labels, K)

    # Build model
    if model_id == "B1":
        model = TFIDF_LR()
    else:
        model = TFIDF_SVC()

    # Train
    logger.info("Training %s on %d samples...", model_id, len(train_texts))
    model.fit(train_texts, train_oh)

    # Evaluate
    val_metrics,  val_preds,  val_true  = evaluate_baseline(
        model, val_texts,  val_labels,  label_classes, f"{model_id}/val")
    test_metrics, test_preds, test_true = evaluate_baseline(
        model, test_texts, test_labels, label_classes, f"{model_id}/test")

    # Save
    model.save(str(out / "model.pkl"))
    with open(out / "val_metrics.json",  "w") as f:
        json.dump({k: v for k, v in val_metrics.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2)
    with open(out / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2)
    np.savez(str(out / "predictions.npz"),
             val_preds=val_preds, val_true=val_true,
             test_preds=test_preds, test_true=test_true)
    # Save per-label F1 separately for comparison table
    np.save(str(out / "per_label_f1.npy"), test_metrics["per_label_f1"])

    logger.info("%s training complete. Outputs in %s", model_id, str(out))
    return test_metrics


def main():
    args = parse_args()
    proc = Path(args.processed)

    train_df      = pd.read_csv(str(proc / "train.csv"))
    val_df        = pd.read_csv(str(proc / "val.csv"))
    test_df       = pd.read_csv(str(proc / "test.csv"))
    label_classes = np.load(str(proc / "label_classes.npy"), allow_pickle=True)

    all_results = {}
    for model_id in args.models:
        logger.info("=" * 60)
        logger.info("Training baseline: %s", model_id)
        metrics = train_baseline(
            model_id, train_df, val_df, test_df, label_classes,
            out_dir=args.out
        )
        all_results[model_id] = metrics

    # Print comparison table
    logger.info("\n%s", format_results_table(all_results))


if __name__ == "__main__":
    main()
