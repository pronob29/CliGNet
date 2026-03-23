"""
dataset.py
----------
PyTorch Dataset classes for the MTSamples clinical transcription dataset.

MTSamplesDataset returns (input_ids, attention_mask, label) for each record.
For documents longer than max_length tokens, a sliding window strategy
produces multiple overlapping chunks whose [CLS] embeddings are later
mean-pooled inside the model forward pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MTSamplesDataset(Dataset):
    """PyTorch Dataset wrapping MTSamples clinical transcriptions.

    Each item returned is a dict:
        input_ids       : LongTensor of shape (num_chunks, max_length)
        attention_mask  : LongTensor of shape (num_chunks, max_length)
        label           : LongTensor scalar (0 to num_classes-1)
        num_chunks      : int, actual number of non-padding chunks
        text            : raw transcription string (for interpretability)

    The num_chunks dimension allows the model to mean-pool [CLS] embeddings
    across all chunks from a single document.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        stride: int = 128,
        max_chunks: int = 4,
        label_col: str = "label",
        text_col: str = "transcription",
    ):
        """
        Parameters
        ----------
        df          : DataFrame with text_col and label_col columns
        tokenizer   : HuggingFace tokenizer (Bio_ClinicalBERT)
        max_length  : maximum tokens per chunk (default 512)
        stride      : overlap between consecutive chunks in tokens (default 128)
        max_chunks  : maximum number of chunks per document (default 4)
        label_col   : column name for integer labels
        text_col    : column name for raw text
        """
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stride     = stride
        self.max_chunks = max_chunks
        self.label_col  = label_col
        self.text_col   = text_col
        self._cache: Dict[int, Dict] = {}

        logger.info("MTSamplesDataset: %d records, max_length=%d, stride=%d, max_chunks=%d",
                    len(self.df), max_length, stride, max_chunks)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]

        row   = self.df.iloc[idx]
        text  = str(row[self.text_col])
        label = int(row[self.label_col])

        item = self._encode_with_sliding_window(text, label)
        # Cache is disabled by default to save memory; enable via enable_cache()
        return item

    def enable_cache(self) -> None:
        """Pre-encode and cache all items. Call only if RAM is sufficient."""
        logger.info("Pre-encoding %d items into cache...", len(self))
        for i in range(len(self)):
            self._cache[i] = self.__getitem__(i)
        logger.info("Cache complete.")

    def _encode_with_sliding_window(
        self, text: str, label: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text using a sliding window.

        For texts <= max_length tokens: one chunk.
        For longer texts: multiple overlapping chunks of size max_length
        with stride overlap, capped at max_chunks.

        Returns padded tensors so all documents have shape (max_chunks, max_length),
        with a num_chunks integer indicating real chunk count.
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=False,          # do not truncate here; we slide manually
            padding=False,
        )

        input_ids_full   = encoded["input_ids"][0]       # (seq_len,)
        attention_full   = encoded["attention_mask"][0]  # (seq_len,)
        seq_len          = input_ids_full.size(0)

        chunks_ids  = []
        chunks_mask = []

        if seq_len <= self.max_length:
            # Single chunk — pad to max_length
            pad_len  = self.max_length - seq_len
            ids      = torch.cat([input_ids_full,
                                  torch.full((pad_len,), self.tokenizer.pad_token_id,
                                             dtype=torch.long)])
            mask     = torch.cat([attention_full,
                                  torch.zeros(pad_len, dtype=torch.long)])
            chunks_ids.append(ids)
            chunks_mask.append(mask)
        else:
            # Sliding window
            step = self.max_length - self.stride
            start = 0
            while start < seq_len and len(chunks_ids) < self.max_chunks:
                end   = min(start + self.max_length, seq_len)
                chunk = input_ids_full[start:end]
                msk   = attention_full[start:end]

                if len(chunk) < 32:
                    # Skip trivially short trailing chunks
                    break

                # Pad if last chunk is shorter than max_length
                pad_len = self.max_length - len(chunk)
                if pad_len > 0:
                    chunk = torch.cat([chunk,
                                       torch.full((pad_len,), self.tokenizer.pad_token_id,
                                                  dtype=torch.long)])
                    msk   = torch.cat([msk,
                                       torch.zeros(pad_len, dtype=torch.long)])

                chunks_ids.append(chunk)
                chunks_mask.append(msk)
                start += step

        num_chunks = len(chunks_ids)

        # Pad to max_chunks so batches can be stacked
        while len(chunks_ids) < self.max_chunks:
            chunks_ids.append(torch.full((self.max_length,),
                                         self.tokenizer.pad_token_id, dtype=torch.long))
            chunks_mask.append(torch.zeros(self.max_length, dtype=torch.long))

        return {
            "input_ids":      torch.stack(chunks_ids),          # (max_chunks, max_length)
            "attention_mask": torch.stack(chunks_mask),         # (max_chunks, max_length)
            "label":          torch.tensor(label, dtype=torch.long),
            "num_chunks":     torch.tensor(num_chunks, dtype=torch.long),
            "text":           text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader.

    Stacks tensors; passes text as a list of strings.
    """
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),   # (B, C, L)
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),   # (B, C, L)
        "label":          torch.stack([b["label"]          for b in batch]),   # (B,)
        "num_chunks":     torch.stack([b["num_chunks"]     for b in batch]),   # (B,)
        "text":           [b["text"] for b in batch],
    }


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    stride: int = 128,
    max_chunks: int = 4,
    num_workers: int = 4,
    seed: int = 42,
):
    """Build and return train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, tokenizer
    """
    import torch
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _make_loader(df: pd.DataFrame, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = MTSamplesDataset(df, tokenizer, max_length, stride, max_chunks)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=lambda wid: np.random.seed(seed + wid),
        )

    return (
        _make_loader(train_df, shuffle=True),
        _make_loader(val_df,   shuffle=False),
        _make_loader(test_df,  shuffle=False),
        tokenizer,
    )
