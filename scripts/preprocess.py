"""
scripts/preprocess.py
---------------------
Data cleaning and stratified splitting for the MTSamples dataset.

Usage
-----
    python scripts/preprocess.py --csv data/raw/mtsamples.csv \
                                  --out data/processed/

Outputs (in --out directory)
----------------------------
    train.csv, val.csv, test.csv
    label_classes.npy     (40 specialty name strings)
    class_weights.npy     (40 inverse-frequency weights for focal loss)
    specialty_stats.csv   (EDA: count, pct, tier per specialty)
    preprocessing_log.json

Reproducibility
---------------
Random seed is fixed at 42 (matches configs/default.yaml).
All splits are completed before any tokenization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    clean_dataset,
    compute_class_weights,
    encode_labels,
    get_specialty_stats,
    load_mtsamples,
    save_splits,
    stratified_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MTSamples preprocessing pipeline")
    p.add_argument("--csv",   default="data/raw/mtsamples.csv",
                   help="Path to raw mtsamples.csv")
    p.add_argument("--out",   default="data/processed/",
                   help="Output directory for processed splits")
    p.add_argument("--train", type=float, default=0.70, help="Train fraction")
    p.add_argument("--val",   type=float, default=0.15, help="Val fraction")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    df = load_mtsamples(args.csv)
    logger.info("Raw dataset: %d rows", len(df))

    # ---- Clean ----
    df = clean_dataset(df)
    logger.info("After cleaning: %d rows, %d unique specialties",
                len(df), df["medical_specialty"].nunique())

    # ---- Encode labels ----
    df, le = encode_labels(df)

    # ---- Split ----
    train_df, val_df, test_df = stratified_split(
        df, train_ratio=args.train, val_ratio=args.val, seed=args.seed
    )

    # ---- Compute class weights (for focal loss alpha) ----
    weights = compute_class_weights(train_df["label"].values, len(le.classes_))
    np.save(str(out / "class_weights.npy"), weights)
    logger.info("Class weights: min=%.3f, max=%.3f, mean=%.3f",
                weights.min(), weights.max(), weights.mean())

    # ---- EDA: specialty stats ----
    stats = get_specialty_stats(df)
    stats.to_csv(out / "specialty_stats.csv", index=False)
    logger.info("Specialty stats saved.")

    # ---- Save splits ----
    save_splits(train_df, val_df, test_df, le, str(out))

    # ---- Log summary ----
    token_approx = df["transcription"].str.split().apply(len) * 1.3
    summary = {
        "total_records":    len(df),
        "train_records":    len(train_df),
        "val_records":      len(val_df),
        "test_records":     len(test_df),
        "num_specialties":  int(len(le.classes_)),
        "rare_specialties_lt20": int((stats["count"] < 20).sum()),
        "docs_over_512_tokens": int((token_approx > 512).sum()),
        "docs_over_1024_tokens": int((token_approx > 1024).sum()),
        "median_words":     float(df["transcription"].str.split().apply(len).median()),
        "max_words":        int(df["transcription"].str.split().apply(len).max()),
        "seed":             args.seed,
    }
    with open(out / "preprocessing_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Preprocessing complete. Outputs in: %s", str(out))
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
