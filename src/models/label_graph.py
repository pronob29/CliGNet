"""
label_graph.py
--------------
Constructs the specialty label co-occurrence graph used by the GCN module in
CLiGNet.

Because MTSamples is single-label (each document has exactly one specialty),
true co-occurrence is zero for all off-diagonal pairs. Instead, we build the
graph using semantic similarity between specialty-specific document embeddings.
An optional ICD-10 chapter adjacency can be added as a structural prior.

Graph representation
--------------------
- Nodes  : 40 medical specialties
- Edges  : cosine-similarity edge if similarity > threshold
- Node features : 768-dim mean-pool of Bio_ClinicalBERT [CLS] embeddings over
                  all training documents for that specialty
- Self-loops added; adjacency symmetrically normalised: A_hat = D^{-1/2} (A+I) D^{-1/2}
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICD-10 chapter-level specialty groupings
# Used as a structural prior: specialties in the same chapter get a bonus edge.
# ---------------------------------------------------------------------------
ICD10_GROUPS: Dict[str, List[str]] = {
    "cardiovascular": [
        "Cardiovascular / Pulmonary",
        "General Medicine",
    ],
    "musculoskeletal": [
        "Orthopedic",
        "Physical Medicine - Rehab",
        "Chiropractic",
        "Podiatry",
        "Pain Management",
        "Rheumatology",
    ],
    "neurological": [
        "Neurology",
        "Neurosurgery",
        "Psychiatry / Psychology",
        "Sleep Medicine",
        "Speech - Language",
    ],
    "gastrointestinal": [
        "Gastroenterology",
        "Bariatrics",
        "Diets and Nutritions",
    ],
    "genitourinary": [
        "Urology",
        "Obstetrics / Gynecology",
        "Nephrology",
    ],
    "head_neck": [
        "ENT - Otolaryngology",
        "Ophthalmology",
        "Dentistry",
    ],
    "oncology_hematology": [
        "Hematology - Oncology",
        "Lab Medicine - Pathology",
        "Autopsy",
        "Radiology",
    ],
    "endocrine_metabolic": [
        "Endocrinology",
        "Allergy / Immunology",
    ],
    "surgery": [
        "Surgery",
        "Cosmetic / Plastic Surgery",
        "Neurosurgery",
    ],
    "administrative": [
        "SOAP / Chart / Progress Notes",
        "Office Notes",
        "Consult - History and Phy.",
        "Discharge Summary",
        "Letters",
        "IME-QME-Work Comp etc.",
    ],
}


def compute_node_features(
    specialties: List[str],
    train_df,
    model,
    tokenizer,
    device: torch.device,
    max_docs_per_specialty: int = 30,
    max_length: int = 128,
) -> torch.Tensor:
    """Compute 768-dim node feature for each specialty.

    Method: mean-pool Bio_ClinicalBERT [CLS] embeddings over up to
    `max_docs_per_specialty` training documents for that specialty.

    If a specialty has zero training documents (rare case), uses the
    tokenized specialty name string as a fallback.

    Parameters
    ----------
    specialties : list of all 40 specialty name strings
    train_df    : training DataFrame with 'medical_specialty' and 'transcription'
    model       : Bio_ClinicalBERT model (in eval mode, on device)
    tokenizer   : corresponding tokenizer
    device      : torch device
    max_docs_per_specialty : cap documents per specialty to avoid imbalance
    max_length  : token length for node feature computation (128 suffices)

    Returns
    -------
    node_features : Tensor of shape (num_specialties, 768)
    """
    model.eval()
    node_feats = []

    with torch.no_grad():
        for spec in specialties:
            docs = train_df[train_df["medical_specialty"] == spec]["transcription"].tolist()

            if len(docs) == 0:
                # Fallback: embed the specialty name itself
                docs = [spec]

            # Cap
            docs = docs[:max_docs_per_specialty]

            cls_embs = []
            for text in docs:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                ).to(device)
                out = model(**enc, output_hidden_states=False)
                cls_emb = out.last_hidden_state[:, 0, :]    # (1, 768)
                cls_embs.append(cls_emb.squeeze(0))         # (768,)

            node_feat = torch.stack(cls_embs).mean(dim=0)   # (768,)
            node_feats.append(node_feat)

    return torch.stack(node_feats)   # (num_specialties, 768)


def compute_semantic_adjacency(
    node_features: torch.Tensor,
    threshold: float = 0.30,
    icd10_bonus: float = 0.20,
    specialty_list: Optional[List[str]] = None,
) -> torch.Tensor:
    """Build a symmetric adjacency matrix from cosine similarity.

    Parameters
    ----------
    node_features   : (N, 768) tensor of specialty embeddings
    threshold       : cosine-similarity threshold for edge creation
    icd10_bonus     : added to similarity for specialties in the same ICD-10 group
    specialty_list  : ordered list of specialty names (for ICD-10 bonus lookup)

    Returns
    -------
    adj : (N, N) float tensor, symmetric, zero diagonal (self-loops added later)
    """
    N = node_features.size(0)
    norms = F.normalize(node_features, p=2, dim=1)   # (N, 768)
    sim   = torch.mm(norms, norms.t())                # (N, N) cosine similarity

    # Add ICD-10 group bonus
    if specialty_list is not None:
        for group_specs in ICD10_GROUPS.values():
            indices = [i for i, s in enumerate(specialty_list) if s in group_specs]
            for i in indices:
                for j in indices:
                    if i != j:
                        sim[i, j] = min(sim[i, j].item() + icd10_bonus, 1.0)

    # Apply threshold
    adj = (sim > threshold).float() * sim
    # Zero out diagonal (self-loops added in normalise_adjacency)
    adj.fill_diagonal_(0.0)
    # Symmetrise
    adj = (adj + adj.t()) / 2.0

    n_edges = (adj > 0).sum().item()
    logger.info("Adjacency: %d / %d possible edges above threshold %.2f",
                int(n_edges), N * (N - 1), threshold)
    return adj


def normalise_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Add self-loops and apply symmetric normalisation.

    A_hat = D^{-1/2} (A + I) D^{-1/2}

    Parameters
    ----------
    adj : (N, N) adjacency matrix (no self-loops)

    Returns
    -------
    A_hat : (N, N) normalised adjacency
    """
    N   = adj.size(0)
    A_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
    degree  = A_tilde.sum(dim=1)                      # (N,)
    D_inv_sqrt = torch.diag(degree.pow(-0.5))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat


def build_label_graph(
    specialties: List[str],
    train_df,
    bert_model,
    tokenizer,
    device: torch.device,
    threshold: float = 0.30,
    max_docs_per_specialty: int = 30,
    save_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Full pipeline: compute node features, build and normalise adjacency.

    Parameters
    ----------
    specialties             : ordered list of specialty name strings
    train_df                : training DataFrame
    bert_model              : Bio_ClinicalBERT model
    tokenizer               : corresponding tokenizer
    device                  : torch device
    threshold               : cosine-similarity threshold (default 0.30)
    max_docs_per_specialty  : documents per specialty for embedding (default 30)
    save_path               : if provided, saves graph dict to this .pt path

    Returns
    -------
    dict with keys:
        node_features   : (N, 768)
        adj_raw         : (N, N) thresholded adjacency (no self-loops)
        adj_norm        : (N, N) normalised adjacency with self-loops
        specialties     : list of specialty names (ordering)
    """
    logger.info("Building label graph for %d specialties...", len(specialties))

    node_features = compute_node_features(
        specialties, train_df, bert_model, tokenizer, device, max_docs_per_specialty
    )

    adj_raw  = compute_semantic_adjacency(node_features, threshold, specialty_list=specialties)
    adj_norm = normalise_adjacency(adj_raw)

    graph = {
        "node_features": node_features.cpu(),
        "adj_raw":       adj_raw.cpu(),
        "adj_norm":      adj_norm.cpu(),
        "specialties":   specialties,
    }

    if save_path is not None:
        torch.save(graph, save_path)
        logger.info("Label graph saved to %s", save_path)

    return graph


def load_label_graph(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a saved label graph and move tensors to device."""
    graph = torch.load(path, map_location=device)
    graph["node_features"] = graph["node_features"].to(device)
    graph["adj_raw"]       = graph["adj_raw"].to(device)
    graph["adj_norm"]      = graph["adj_norm"].to(device)
    logger.info("Loaded label graph from %s (%d specialties)", path,
                len(graph["specialties"]))
    return graph
