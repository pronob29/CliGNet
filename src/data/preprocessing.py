"""
preprocessing.py
----------------
Data cleaning, label encoding, and stratified train/val/test split for the
MTSamples clinical transcription dataset.

All splitting is performed BEFORE any tokenization or augmentation to prevent
data leakage. SMOTE and oversampling are explicitly excluded from this module;
they appear only inside the ablation study (ablation.py).
"""

import re
import logging
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMPTY_SPECIALTY_FILL = "Unknown"
MIN_TRANSCRIPTION_CHARS = 10  # drop tokens shorter than this after stripping


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters.

    Preserves case (Bio_ClinicalBERT is cased). Preserves clinical punctuation
    (periods, commas, slashes). Does NOT lowercase.
    """
    if not isinstance(text, str):
        return ""
    # remove non-printable
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    # collapse multiple whitespace to single space, normalize line endings
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_mtsamples(csv_path: str) -> pd.DataFrame:
    """Load MTSamples CSV and return raw dataframe.

    The CSV has columns: [unnamed index, description, medical_specialty,
    sample_name, transcription, keywords].
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at {csv_path}.\n"
            "Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
            "Place as data/raw/mtsamples.csv"
        )
    df = pd.read_csv(str(path), index_col=0, low_memory=False)
    logger.info("Loaded %d raw rows from %s", len(df), csv_path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps in order.

    Steps
    -----
    C1  Drop rows with missing or empty transcription.
    C2  Strip leading/trailing whitespace from all string columns.
    C3  Normalize transcription text (whitespace, non-printable chars).
    C4  Fill missing specialty with EMPTY_SPECIALTY_FILL then strip.
    C5  Drop transcriptions shorter than MIN_TRANSCRIPTION_CHARS after cleaning.
    C6  Reset index.
    """
    original_len = len(df)

    # C1 — drop empty transcriptions
    df = df.dropna(subset=["transcription"])
    df = df[df["transcription"].str.strip() != ""]
    logger.info("C1: dropped %d rows with empty transcription (remaining: %d)",
                original_len - len(df), len(df))

    # C2 — strip string columns
    for col in ["medical_specialty", "sample_name", "description", "keywords"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # C3 — normalize transcription text
    df["transcription"] = df["transcription"].apply(_clean_text)

    # C4 — fill and strip specialty
    df["medical_specialty"] = df["medical_specialty"].replace("", EMPTY_SPECIALTY_FILL)
    df["medical_specialty"] = df["medical_specialty"].str.strip()

    # C5 — drop very short transcriptions after cleaning
    before = len(df)
    df = df[df["transcription"].str.len() >= MIN_TRANSCRIPTION_CHARS]
    logger.info("C5: dropped %d transcriptions shorter than %d chars",
                before - len(df), MIN_TRANSCRIPTION_CHARS)

    # C6 — reset index
    df = df.reset_index(drop=True)
    logger.info("Cleaned dataset: %d records, %d unique specialties",
                len(df), df["medical_specialty"].nunique())
    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encode specialty strings as integer labels.

    Adds column 'label' (int, 0-indexed) to the dataframe.
    Returns (modified_df, fitted_LabelEncoder).
    """
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df["medical_specialty"])
    logger.info("Label encoder: %d classes", len(le.classes_))
    return df, le


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights for focal loss alpha parameter.

    Returns weights array of shape (num_classes,) where each weight is
    total_samples / (num_classes * class_count). This is the balanced
    sklearn-style weighting.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return weights


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a stratified 70/15/15 train/val/test split.

    Uses sklearn StratifiedShuffleSplit on the integer label column.
    The test set is held out immediately and never used during training
    or threshold calibration.

    Parameters
    ----------
    df          : cleaned dataframe with 'label' column
    train_ratio : fraction for training  (default 0.70)
    val_ratio   : fraction for validation (default 0.15)
    seed        : random seed for reproducibility

    Returns
    -------
    train_df, val_df, test_df
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column. Call encode_labels() first.")

    test_ratio = 1.0 - train_ratio - val_ratio
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1.0"

    # Add a stable row id before any split to enable overlap checking
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    labels = df["label"].values

    # First split: separate test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df     = df.iloc[test_idx].reset_index(drop=True)

    # Second split: separate val from trainval
    relative_val = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    trainval_labels = trainval_df["label"].values
    train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_labels)), trainval_labels))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df   = trainval_df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df), 100 * len(train_df) / len(df),
        len(val_df),   100 * len(val_df)   / len(df),
        len(test_df),  100 * len(test_df)  / len(df),
    )

    # Verify no overlap using stable row IDs (not text keys)
    train_ids = set(train_df["_row_id"].values)
    val_ids   = set(val_df["_row_id"].values)
    test_ids  = set(test_df["_row_id"].values)
    assert len(train_ids & test_ids) == 0, "Leakage: train/test overlap detected"
    assert len(val_ids   & test_ids) == 0, "Leakage: val/test overlap detected"
    assert len(train_ids & val_ids)  == 0, "Leakage: train/val overlap detected"
    logger.info("Overlap check passed: no train/val/test leakage.")

    # Drop the helper column
    for _df in [train_df, val_df, test_df]:
        _df.drop(columns=["_row_id"], inplace=True, errors="ignore")

    return train_df, val_df, test_df


def get_specialty_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary dataframe of specialty distribution."""
    stats = (
        df.groupby("medical_specialty")
        .agg(count=("transcription", "count"),
             avg_len=("transcription", lambda x: x.str.len().mean()),
             avg_words=("transcription", lambda x: x.str.split().apply(len).mean()))
        .sort_values("count", ascending=False)
        .reset_index()
    )
    stats["pct"] = 100.0 * stats["count"] / stats["count"].sum()
    stats["tier"] = stats["count"].apply(
        lambda c: "dominant" if c >= 500 else
                  "major"    if c >= 100 else
                  "moderate" if c >= 50  else
                  "rare"     if c >= 20  else "very_rare"
    )
    return stats


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    le: LabelEncoder,
    output_dir: str,
) -> None:
    """Save split CSVs and label encoder classes to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out   / "val.csv",   index=False)
    test_df.to_csv(out  / "test.csv",  index=False)
    np.save(str(out / "label_classes.npy"), le.classes_)
    logger.info("Saved splits and label classes to %s", output_dir)
