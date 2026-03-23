"""
scripts/interpret.py
--------------------
Interpretability analysis and full failure analysis (Contributions C3 and C4).

Steps performed
---------------
1. Load trained CLiGNet B8 model and test set
2. Run Integrated Gradients on up to --num-samples documents per specialty
3. Extract top-5 tokens per label per document
4. Compute optional UMLS signal recovery rate (requires scispaCy)
5. Run all four failure analysis sub-components (F1–F4)
6. Save all outputs to results/interpretability/

Usage
-----
    python scripts/interpret.py --processed data/processed/ \
                                  --graph data/processed/label_graph.pt \
                                  --model results/clignet/B8/ \
                                  --predictions results/clignet/B8/predictions.npz \
                                  --out results/interpretability/ \
                                  --num-samples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import load_mtsamples, clean_dataset, encode_labels
from src.models.clignet import CLiGNet
from src.models.label_graph import load_label_graph
from src.interpretability.integrated_gradients import CLiGNetExplainer
from src.interpretability.failure_analysis import run_full_failure_analysis
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Interpretability and failure analysis")
    p.add_argument("--processed",    default="data/processed/")
    p.add_argument("--graph",        default="data/processed/label_graph.pt")
    p.add_argument("--model-dir",    default="results/clignet/B8/",
                   help="Directory containing best_model.pt and calibrator.npz")
    p.add_argument("--predictions",  default="results/clignet/B8/predictions.npz")
    p.add_argument("--longformer-preds", default=None,
                   help="Optional: results/baselines/B7/predictions.npz for F3 comparison")
    p.add_argument("--out",          default="results/interpretability/")
    p.add_argument("--bert",         default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--num-samples",  type=int, default=5,
                   help="Documents per specialty for IG (5 recommended for speed)")
    p.add_argument("--n-steps",      type=int, default=50,
                   help="IG integration steps (50 standard, 100 for higher precision)")
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def load_clignet(model_dir, graph, label_classes, bert_name, device):
    """Load trained CLiGNet from checkpoint."""
    K     = len(label_classes)
    model = CLiGNet(
        bert_model_name=bert_name,
        num_labels=K,
        node_features=graph["node_features"],
        adj_norm=graph["adj_norm"],
    )
    ckpt_path = Path(model_dir) / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded CLiGNet from %s", ckpt_path)
    return model


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    proc          = Path(args.processed)
    out           = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label_classes = np.load(str(proc / "label_classes.npy"), allow_pickle=True).tolist()
    K             = len(label_classes)

    # ---- Load test set ----
    test_df = pd.read_csv(str(proc / "test.csv"))
    logger.info("Test set: %d records", len(test_df))

    # ---- Load model ----
    graph = load_label_graph(args.graph, device)
    model = load_clignet(args.model_dir, graph, label_classes, args.bert, device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert)

    # ---- Load predictions ----
    preds_data = np.load(args.predictions)
    test_preds = preds_data["test_preds"]   # (N, K) one-hot
    test_true  = preds_data["test_true"]    # (N, K) one-hot

    # ---- Per-label F1 ----
    metrics       = compute_all_metrics(test_true, test_preds, label_classes)
    per_label_f1  = metrics["per_label_f1"]
    logger.info("Test macro_f1=%.4f | micro_f1=%.4f", metrics["macro_f1"], metrics["micro_f1"])

    # ---- Integrated Gradients ----
    explainer  = CLiGNetExplainer(model, tokenizer, device)
    ig_results = []

    logger.info("Running IG attribution (%d samples per specialty)...", args.num_samples)
    for label_idx, spec in enumerate(label_classes):
        # Sample up to num_samples documents from this specialty in test set
        spec_df = test_df[test_df["medical_specialty"] == spec].head(args.num_samples)
        if len(spec_df) == 0:
            continue

        for _, row in spec_df.iterrows():
            text = str(row["transcription"])
            try:
                exp       = explainer.explain(text, label_idx, n_steps=args.n_steps)
                top_toks  = explainer.top_k_tokens(exp, k=5)
                ig_entry  = {
                    "specialty":   spec,
                    "label_idx":   label_idx,
                    "top_tokens":  top_toks,
                    "convergence_delta": round(exp["convergence_delta"], 4),
                    "f1_of_specialty": round(float(per_label_f1[label_idx]), 4),
                }
                ig_results.append(ig_entry)
            except Exception as e:
                logger.warning("IG failed for %s: %s", spec, str(e))

    logger.info("IG complete: %d explanations generated", len(ig_results))

    # Save IG results
    with open(out / "ig_top_tokens.json", "w") as f:
        json.dump(ig_results, f, indent=2)

    # ---- Failure Analysis (C4: F1–F4) ----
    # Compute word counts for test set
    word_counts = test_df["transcription"].str.split().apply(len).values

    # Training class counts
    train_df     = pd.read_csv(str(proc / "train.csv"))
    train_counts = np.array([
        int((train_df["medical_specialty"] == spec).sum())
        for spec in label_classes
    ])

    # Optional Longformer predictions for F3
    y_pred_lf = None
    if args.longformer_preds and Path(args.longformer_preds).exists():
        lf_data   = np.load(args.longformer_preds)
        y_pred_lf = lf_data["test_preds"]
        logger.info("Loaded Longformer predictions for F3 comparison.")

    failure_results = run_full_failure_analysis(
        y_true=test_true,
        y_pred_clignet=test_preds,
        label_names=label_classes,
        train_counts=train_counts,
        word_counts=word_counts,
        output_dir=str(out),
        per_label_f1=per_label_f1,
        y_pred_longformer=y_pred_lf,
        ig_explanations=ig_results,
    )

    logger.info("Interpretability and failure analysis complete. Outputs in: %s", str(out))

    # ---- Summary JSON ----
    summary = {
        "ig_explanations_count":  len(ig_results),
        "low_f1_specialties":     failure_results["F4"]["low_f1_labels"],
        "top_confused_pairs_5":   failure_results["F1"]["confused_pairs"][:5],
        "f1_by_class_size":       failure_results["F2"]["binned_f1"],
        "f1_by_doc_length":       failure_results["F3"]["clignet"],
    }
    with open(out / "interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
