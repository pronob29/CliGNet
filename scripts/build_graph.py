"""
scripts/build_graph.py
-----------------------
Build the label co-occurrence graph for the CLiGNet GCN module.

Usage
-----
    python scripts/build_graph.py --processed data/processed/ \
                                   --out data/processed/label_graph.pt

Outputs
-------
    data/processed/label_graph.pt   — torch.save dict with:
        node_features   (40, 768)
        adj_raw         (40, 40)
        adj_norm        (40, 40)
        specialties     list of 40 specialty strings

Method
------
1. Load Bio_ClinicalBERT
2. For each specialty: mean-pool [CLS] embeddings over up to 30 training docs
3. Compute cosine-similarity adjacency (threshold=0.30)
4. Add ICD-10 chapter structural prior (bonus=0.20)
5. Symmetric normalisation: A_hat = D^{-1/2}(A+I)D^{-1/2}
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.label_graph import build_label_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build CLiGNet label graph")
    p.add_argument("--processed", default="data/processed/",
                   help="Directory with train.csv and label_classes.npy")
    p.add_argument("--out",       default="data/processed/label_graph.pt",
                   help="Output path for label_graph.pt")
    p.add_argument("--bert",      default="emilyalsentzer/Bio_ClinicalBERT",
                   help="HuggingFace model name for node embedding")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Cosine similarity threshold for edge creation")
    p.add_argument("--max-docs",  type=int,   default=30,
                   help="Max documents per specialty for node embedding")
    p.add_argument("--device",    default="auto",
                   help="Device: 'cpu', 'cuda', or 'auto'")
    return p.parse_args()


def main():
    args = parse_args()
    proc = Path(args.processed)

    # ---- Determine device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ---- Load data ----
    train_df = pd.read_csv(str(proc / "train.csv"))
    label_classes = np.load(str(proc / "label_classes.npy"), allow_pickle=True).tolist()
    logger.info("Loaded %d training records, %d specialties", len(train_df), len(label_classes))

    # ---- Load Bio_ClinicalBERT ----
    logger.info("Loading %s...", args.bert)
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    model     = AutoModel.from_pretrained(args.bert).to(device)
    model.eval()

    # ---- Build graph ----
    graph = build_label_graph(
        specialties=label_classes,
        train_df=train_df,
        bert_model=model,
        tokenizer=tokenizer,
        device=device,
        threshold=args.threshold,
        max_docs_per_specialty=args.max_docs,
        save_path=args.out,
    )

    logger.info("Graph built: %d nodes, adjacency shape %s",
                len(graph["specialties"]), graph["adj_norm"].shape)
    n_edges = int((graph["adj_raw"] > 0).sum().item())
    logger.info("Edges above threshold %.2f: %d / %d possible",
                args.threshold, n_edges, len(label_classes) * (len(label_classes) - 1))
    logger.info("Label graph saved to %s", args.out)


if __name__ == "__main__":
    main()
