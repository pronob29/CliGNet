"""
scripts/train_clignet.py
------------------------
Train CLiGNet (B6, B8) and BERT baselines (B3, B4, B7).

Usage — CLiGNet Full (B8, recommended):
    python scripts/train_clignet.py --processed data/processed/ \
                                     --graph data/processed/label_graph.pt \
                                     --out results/clignet/ \
                                     --mode clignet \
                                     --gamma 2.0

Usage — BERT baseline (B3, B4, B7):
    python scripts/train_clignet.py --processed data/processed/ \
                                     --out results/baselines/ \
                                     --mode baseline \
                                     --baseline-id B3

Usage — CLiGNet without calibration (B6):
    python scripts/train_clignet.py ... --mode clignet --no-calibration

Key flags
---------
    --mode           : 'clignet' or 'baseline'
    --baseline-id    : 'B3', 'B4', 'B7' (only used when --mode=baseline)
    --gamma          : focal loss gamma (1.0, 2.0, 3.0 — tune on val)
    --no-calibration : disable Platt scaling (produces B6 from B8 config)
    --freeze-bert    : number of lower BERT layers to freeze (default 6)
    --device         : 'auto', 'cuda', 'cpu'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders
from src.models.baselines import BERT_BASELINE_CONFIGS, make_bert_baseline
from src.models.clignet import CLiGNet
from src.models.label_graph import load_label_graph
from src.training.calibration import LabelCalibrator, compute_ece
from src.training.loss import build_focal_loss
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_all_metrics, format_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train CLiGNet / BERT baselines")
    p.add_argument("--processed",      default="data/processed/")
    p.add_argument("--graph",          default="data/processed/label_graph.pt",
                   help="Label graph .pt (required for --mode=clignet)")
    p.add_argument("--out",            default="results/clignet/")
    p.add_argument("--mode",           choices=["clignet", "baseline"], default="clignet")
    p.add_argument("--baseline-id",    choices=["B3", "B4", "B7"],    default="B3")
    p.add_argument("--gamma",          type=float, default=2.0,
                   help="Focal loss gamma (1.0/2.0/3.0)")
    p.add_argument("--no-calibration", action="store_true",
                   help="Disable Platt scaling (produces B6 from CLiGNet config)")
    p.add_argument("--freeze-bert",    type=int,   default=6,
                   help="Number of lower BERT layers to freeze")
    p.add_argument("--bert",           default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--grad-accum",     type=int,   default=2,
                   help="Gradient accumulation steps")
    p.add_argument("--max-epochs",     type=int,   default=30)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--bert-lr",        type=float, default=2e-5)
    p.add_argument("--head-lr",        type=float, default=1e-3)
    p.add_argument("--max-length",     type=int,   default=512)
    p.add_argument("--stride",         type=int,   default=128)
    p.add_argument("--max-chunks",     type=int,   default=4)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--device",         default="auto")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_logits(model, loader, device):
    """Collect all raw logits from a DataLoader (no grad)."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            nchk  = batch["num_chunks"].to(device)
            lbl   = batch["label"].cpu().numpy()
            logit = model(ids, mask, nchk).cpu().numpy()
            all_logits.append(logit)
            all_labels.append(lbl)
    return np.concatenate(all_logits), np.concatenate(all_labels)


def logits_to_preds(logits: np.ndarray, K: int) -> np.ndarray:
    """Argmax -> one-hot (N, K)."""
    N = logits.shape[0]
    preds = np.zeros((N, K), dtype=int)
    preds[np.arange(N), logits.argmax(axis=1)] = 1
    return preds


def labels_to_onehot(labels: np.ndarray, K: int) -> np.ndarray:
    one_hot = np.zeros((len(labels), K), dtype=int)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    logger.info("Device: %s", device)

    proc          = Path(args.processed)
    label_classes = np.load(str(proc / "label_classes.npy"), allow_pickle=True)
    class_weights = np.load(str(proc / "class_weights.npy"))
    K             = len(label_classes)

    train_df = pd.read_csv(str(proc / "train.csv"))
    val_df   = pd.read_csv(str(proc / "val.csv"))
    test_df  = pd.read_csv(str(proc / "test.csv"))

    # ---- Determine tokenizer name ----
    if args.mode == "baseline":
        tok_name = BERT_BASELINE_CONFIGS[args.baseline_id]["model_name"]
        use_longformer = args.baseline_id == "B7"
        # Allow --max-length to override the Longformer default of 4096
        max_length = (4096 if args.max_length == 512 else args.max_length) if use_longformer else args.max_length
        model_tag  = args.baseline_id
    else:
        tok_name   = args.bert
        max_length = args.max_length
        model_tag  = "B8" if not args.no_calibration else "B6"

    out_dir = Path(args.out) / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DataLoaders ----
    logger.info("Building DataLoaders (tokenizer: %s, max_length: %d)...", tok_name, max_length)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        train_df, val_df, test_df,
        tokenizer_name=tok_name,
        batch_size=args.batch_size,
        max_length=max_length,
        stride=args.stride,
        max_chunks=args.max_chunks,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ---- Build model ----
    if args.mode == "baseline":
        model = make_bert_baseline(tok_name, K, args.baseline_id)
        is_bert_baseline = True
    else:
        graph = load_label_graph(args.graph, device)
        model = CLiGNet(
            bert_model_name=tok_name,
            num_labels=K,
            node_features=graph["node_features"],
            adj_norm=graph["adj_norm"],
            freeze_bert_layers=args.freeze_bert,
        )
        is_bert_baseline = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | trainable params: %d", model_tag, trainable)

    # ---- Loss ----
    loss_fn = build_focal_loss(class_weights, gamma=args.gamma, device=device)

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        num_labels=K,
        output_dir=str(out_dir),
        bert_lr=args.bert_lr,
        head_lr=args.head_lr,
        max_epochs=args.max_epochs,
        grad_accum_steps=args.grad_accum,
        patience=args.patience,
        device=device,
        seed=args.seed,
        is_bert_baseline=is_bert_baseline,
    )

    # ---- Train ----
    trainer.fit()
    trainer.load_best_model()

    # ---- Collect logits ----
    val_logits,  val_labels_int  = collect_logits(model, val_loader,  device)
    test_logits, test_labels_int = collect_logits(model, test_loader, device)

    val_true  = labels_to_onehot(val_labels_int,  K)
    test_true = labels_to_onehot(test_labels_int, K)

    # ---- Calibration (B8 only) ----
    if args.mode == "clignet" and not args.no_calibration:
        logger.info("Fitting Platt scaling calibration on validation set...")
        calibrator = LabelCalibrator(num_labels=K)
        calibrator.fit(val_logits, val_true)
        calibrator.save(str(out_dir / "calibrator.npz"))

        test_probs = calibrator.calibrate(test_logits)
        test_preds = calibrator.predict(test_logits)
        val_preds  = calibrator.predict(val_logits)

        ece = compute_ece(test_probs, test_true)
        logger.info("Post-calibration ECE: %.4f", ece)
        with open(out_dir / "calibration_ece.json", "w") as f:
            json.dump({"ece": round(ece, 5)}, f)
    else:
        # Argmax predictions, no calibration
        val_preds  = logits_to_preds(val_logits,  K)
        test_preds = logits_to_preds(test_logits, K)

    # ---- Final metrics ----
    val_metrics  = compute_all_metrics(val_true,  val_preds,  label_classes.tolist())
    test_metrics = compute_all_metrics(test_true, test_preds, label_classes.tolist())

    def _serialisable(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in d.items() if k != "per_label_dict"}

    with open(out_dir / "val_metrics.json",  "w") as f:
        json.dump(_serialisable(val_metrics),  f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(_serialisable(test_metrics), f, indent=2)

    np.savez(str(out_dir / "predictions.npz"),
             val_logits=val_logits,  val_preds=val_preds,  val_true=val_true,
             test_logits=test_logits, test_preds=test_preds, test_true=test_true)
    np.save(str(out_dir / "per_label_f1.npy"), test_metrics["per_label_f1"])

    logger.info("\nFINAL TEST METRICS for %s:", model_tag)
    logger.info("  macro_f1   = %.4f", test_metrics["macro_f1"])
    logger.info("  micro_f1   = %.4f", test_metrics["micro_f1"])
    logger.info("  accuracy   = %.4f", test_metrics["accuracy"])
    logger.info("  hamming    = %.5f", test_metrics["hamming_loss"])
    logger.info("Outputs saved to: %s", str(out_dir))


if __name__ == "__main__":
    main()
