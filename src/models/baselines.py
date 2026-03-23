"""
baselines.py
------------
Classical and BERT-based baselines B1 through B5 and B7.

B1 : TF-IDF (unigrams+bigrams) + One-vs-Rest Logistic Regression
B2 : TF-IDF + One-vs-Rest SVC (RBF kernel)
B3 : Bio_ClinicalBERT + Linear Head (One-vs-Rest, no GCN)
B4 : BioBERT-base + Linear Head (One-vs-Rest)
B7 : Clinical-Longformer (4096-token) + Linear Head (One-vs-Rest)

B6 (CLiGNet without calibration) and B8 (CLiGNet full) are in clignet.py +
trainer.py. Calibration is in training/calibration.py.

All baseline predict() methods return probability arrays of shape (N, num_classes)
so the evaluation module can compute metrics uniformly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B1 — TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
class TFIDF_LR:
    """Baseline B1: TF-IDF (1,2-gram) + One-vs-Rest Logistic Regression."""

    name = "B1_TFIDF_LR"

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", OneVsRestClassifier(
                LogisticRegression(C=C, max_iter=max_iter, solver="saga",
                                   class_weight="balanced", n_jobs=-1),
                n_jobs=-1,
            )),
        ])

    def fit(self, texts: List[str], labels: np.ndarray) -> "TFIDF_LR":
        logger.info("Training %s...", self.name)
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability matrix (N, num_classes)."""
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: str) -> "TFIDF_LR":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# B2 — TF-IDF + Linear SVC (calibrated for probability output)
# ---------------------------------------------------------------------------
class TFIDF_SVC:
    """Baseline B2: TF-IDF + One-vs-Rest SVC with Platt calibration."""

    name = "B2_TFIDF_SVC"

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
    ):
        base_svc = CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=2000, class_weight="balanced"),
            cv=3, method="sigmoid",
        )
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", OneVsRestClassifier(base_svc, n_jobs=-1)),
        ])

    def fit(self, texts: List[str], labels: np.ndarray) -> "TFIDF_SVC":
        logger.info("Training %s...", self.name)
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: str) -> "TFIDF_SVC":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ---------------------------------------------------------------------------
# Shared BERT Classifier for B3, B4, B7
# ---------------------------------------------------------------------------
class BERTClassifier(nn.Module):
    """Linear classification head on top of a HuggingFace BERT-family model.

    Used for baselines B3 (Bio_ClinicalBERT), B4 (BioBERT), B7 (Longformer).

    For B7 (Longformer), sliding window is disabled; the model receives the
    full document up to 4096 tokens.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        max_length: int = 512,
        use_longformer: bool = False,
    ):
        super().__init__()
        self.model_name     = model_name
        self.num_labels     = num_labels
        self.max_length     = max_length
        self.use_longformer = use_longformer

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim   = self.encoder.config.hidden_size

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        num_chunks:     torch.Tensor,    # ignored for Longformer (single chunk)
    ) -> torch.Tensor:
        B, C, L = input_ids.shape

        if self.use_longformer:
            # B7: single-chunk, full document up to 4096 tokens
            ids   = input_ids[:, 0, :]        # (B, L)
            mask  = attention_mask[:, 0, :]   # (B, L)
            # Longformer requires global_attention_mask for [CLS]
            global_attention = torch.zeros_like(mask)
            global_attention[:, 0] = 1        # [CLS] token gets global attention
            out   = self.encoder(input_ids=ids, attention_mask=mask,
                                 global_attention_mask=global_attention)
            cls_emb = out.last_hidden_state[:, 0, :]   # (B, 768)
        else:
            # B3, B4: sliding-window mean-pool (same as CLiGNet encoder)
            flat_ids  = input_ids.view(B * C, L)
            flat_mask = attention_mask.view(B * C, L)
            out       = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
            cls_all   = out.last_hidden_state[:, 0, :].view(B, C, -1)  # (B, C, 768)
            doc_embs  = []
            for i in range(B):
                n = num_chunks[i].item()
                doc_embs.append(cls_all[i, :n, :].mean(dim=0))
            cls_emb = torch.stack(doc_embs)   # (B, 768)

        cls_emb = self.dropout(cls_emb)
        logits  = self.classifier(cls_emb)    # (B, num_labels)
        return logits

    def get_probabilities(self, input_ids, attention_mask, num_chunks):
        logits = self.forward(input_ids, attention_mask, num_chunks)
        return torch.sigmoid(logits)


def make_bert_baseline(
    model_name: str,
    num_labels: int,
    baseline_id: str,
) -> BERTClassifier:
    """Factory for BERT baselines.

    Parameters
    ----------
    model_name   : HuggingFace model identifier
    num_labels   : number of specialty classes
    baseline_id  : one of 'B3', 'B4', 'B7'
    """
    use_longformer = baseline_id == "B7"
    max_length     = 4096 if use_longformer else 512

    model = BERTClassifier(
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length,
        use_longformer=use_longformer,
    )
    logger.info("Created baseline %s: %s (%d trainable params)",
                baseline_id, model_name,
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
BERT_BASELINE_CONFIGS = {
    "B3": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "Bio_ClinicalBERT + One-vs-Rest linear head, sliding window",
    },
    "B4": {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "description": "BioBERT-base + One-vs-Rest linear head, sliding window",
    },
    "B7": {
        "model_name": "allenai/longformer-base-4096",
        "description": "Clinical-Longformer 4096-token, no GCN, global [CLS] attention",
    },
}
