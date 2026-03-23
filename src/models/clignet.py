"""
clignet.py
----------
CLiGNet: Clinical Label-interaction Graph Network.

Architecture
------------
1.  Text Encoder   : Bio_ClinicalBERT with sliding-window mean-pool [CLS]
2.  GCN            : 2-layer graph convolutional network on the label graph
                     GCNConv(768, gcn_hidden) -> ReLU -> Dropout
                     GCNConv(gcn_hidden, gcn_out) -> ReLU -> Dropout
3.  Fusion         : Per-label attention gate combining doc_emb and node_emb
                     z_k = alpha_k * doc_emb_projected + (1 - alpha_k) * node_k
4.  Classifier     : Linear(gcn_out, 1) per label -> sigmoid
5.  Loss           : Focal BCE (see training/loss.py)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCN Layer (manual implementation; no torch_geometric dependency)
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """Single graph convolutional layer.

    Implements: H' = sigma( A_hat * H * W )

    where A_hat is the pre-normalised adjacency (with self-loops, symmetric norm).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H     : (N, in_dim)  node feature matrix
        A_hat : (N, N)       normalised adjacency

        Returns
        -------
        H_out : (N, out_dim)
        """
        support = self.linear(H)                    # (N, out_dim)
        H_out   = torch.mm(A_hat, support)          # (N, out_dim)
        H_out   = F.relu(H_out)
        H_out   = self.dropout(H_out)
        return H_out


# ---------------------------------------------------------------------------
# Per-label Attention Gate
# ---------------------------------------------------------------------------
class LabelAttentionGate(nn.Module):
    """Compute a scalar gate alpha_k for each of the num_labels labels.

    alpha_k = sigmoid(W_k * [doc_emb; node_k_emb])

    Outputs the fused label-specific representation:
        z_k = alpha_k * doc_proj + (1 - alpha_k) * node_k
    """

    def __init__(self, doc_dim: int, node_dim: int, num_labels: int):
        super().__init__()
        self.doc_proj  = nn.Linear(doc_dim, node_dim)
        self.gate_w    = nn.Linear(node_dim * 2, 1, bias=True)
        self.num_labels = num_labels

    def forward(
        self,
        doc_emb:   torch.Tensor,   # (B, doc_dim)
        node_embs: torch.Tensor,   # (K, node_dim)
    ) -> torch.Tensor:
        """
        Returns
        -------
        fused : (B, K, node_dim)  label-specific fused representations
        """
        B   = doc_emb.size(0)
        K   = node_embs.size(0)

        doc_p = self.doc_proj(doc_emb)                 # (B, node_dim)
        doc_p = doc_p.unsqueeze(1).expand(-1, K, -1)   # (B, K, node_dim)
        node  = node_embs.unsqueeze(0).expand(B, -1, -1)  # (B, K, node_dim)

        concat = torch.cat([doc_p, node], dim=-1)      # (B, K, 2*node_dim)
        alpha  = torch.sigmoid(self.gate_w(concat))    # (B, K, 1)

        fused  = alpha * doc_p + (1.0 - alpha) * node  # (B, K, node_dim)
        return fused


# ---------------------------------------------------------------------------
# CLiGNet
# ---------------------------------------------------------------------------
class CLiGNet(nn.Module):
    """Full CLiGNet model.

    Parameters
    ----------
    bert_model_name : HuggingFace model name for the text encoder
    num_labels      : number of specialty classes (40 for MTSamples)
    node_features   : (N, 768) initial label node embeddings
    adj_norm        : (N, N) normalised adjacency matrix
    gcn_hidden      : hidden dim for first GCN layer (default 512)
    gcn_out         : output dim for second GCN layer (default 256)
    gcn_dropout     : dropout rate in GCN layers (default 0.3)
    bert_dropout    : additional dropout on BERT [CLS] output (default 0.1)
    freeze_bert_layers : number of lower BERT layers to freeze (default 6)
    """

    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        node_features: torch.Tensor,
        adj_norm: torch.Tensor,
        gcn_hidden: int = 512,
        gcn_out: int = 256,
        gcn_dropout: float = 0.3,
        bert_dropout: float = 0.1,
        freeze_bert_layers: int = 6,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ---- Text Encoder ----
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim  = self.bert.config.hidden_size   # 768

        # Freeze lower layers to speed training and prevent catastrophic forgetting
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        self.bert_dropout = nn.Dropout(bert_dropout)

        # ---- Label Graph (non-trainable buffers) ----
        self.register_buffer("node_features_init", node_features)  # (K, 768)
        self.register_buffer("adj_norm", adj_norm)                  # (K, K)

        # Learnable label node embeddings initialized from pre-computed features
        self.node_embed = nn.Parameter(node_features.clone())       # (K, 768)

        # ---- GCN ----
        self.gcn1 = GCNLayer(bert_dim,   gcn_hidden, gcn_dropout)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out,    gcn_dropout)

        # ---- Fusion ----
        self.fusion = LabelAttentionGate(bert_dim, gcn_out, num_labels)

        # ---- Classification Heads ----
        # One head per label (shared weight matrix, label-indexed)
        self.classifier = nn.Linear(gcn_out, num_labels)

        self._init_weights()

    def _freeze_bert_layers(self, n: int) -> None:
        """Freeze the first n transformer encoder layers of BERT."""
        # Always keep embedding layer frozen
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers 0..n-1
        for layer_idx in range(min(n, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info("Froze %d BERT layers. Trainable params: %d / %d", n, trainable, total)

    def _init_weights(self) -> None:
        """Xavier initialisation for GCN and classifier heads."""
        for m in [self.gcn1.linear, self.gcn2.linear,
                  self.fusion.doc_proj, self.fusion.gate_w, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_text(
        self,
        input_ids:      torch.Tensor,    # (B, C, L)
        attention_mask: torch.Tensor,    # (B, C, L)
        num_chunks:     torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """Encode a batch of documents using sliding-window mean-pool.

        Parameters
        ----------
        input_ids      : (B, C, L)  — B=batch, C=max_chunks, L=max_length
        attention_mask : (B, C, L)
        num_chunks     : (B,) actual chunk counts per document

        Returns
        -------
        doc_emb : (B, 768) mean-pooled [CLS] embeddings
        """
        B, C, L = input_ids.shape
        flat_ids  = input_ids.view(B * C, L)       # (B*C, L)
        flat_mask = attention_mask.view(B * C, L)  # (B*C, L)

        out      = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_all  = out.last_hidden_state[:, 0, :]  # (B*C, 768) [CLS] embeddings
        cls_all  = cls_all.view(B, C, -1)          # (B, C, 768)
        cls_all  = self.bert_dropout(cls_all)

        # Mean-pool over valid chunks only
        doc_embs = []
        for i in range(B):
            n = num_chunks[i].item()
            doc_emb = cls_all[i, :n, :].mean(dim=0)   # (768,)
            doc_embs.append(doc_emb)
        doc_emb = torch.stack(doc_embs)                # (B, 768)
        return doc_emb

    def encode_labels(self) -> torch.Tensor:
        """Run two GCN layers on label node embeddings.

        Returns
        -------
        node_embs : (K, gcn_out)
        """
        node_embs = self.gcn1(self.node_embed, self.adj_norm)   # (K, gcn_hidden)
        node_embs = self.gcn2(node_embs, self.adj_norm)         # (K, gcn_out)
        return node_embs

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        num_chunks:     torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Returns
        -------
        logits : (B, num_labels) — unnormalised scores (pass through sigmoid for probs)
        """
        # Document embedding
        doc_emb   = self.encode_text(input_ids, attention_mask, num_chunks)  # (B, 768)

        # Label graph embedding
        node_embs = self.encode_labels()                                     # (K, gcn_out)

        # Fuse
        fused     = self.fusion(doc_emb, node_embs)                          # (B, K, gcn_out)

        # Classify: dot product of each fused label repr with classifier weight
        # fused: (B, K, gcn_out), classifier.weight: (K, gcn_out)
        # per-label logit: fused[b, k, :] @ classifier.weight[k, :]
        logits = (fused * self.classifier.weight.unsqueeze(0)).sum(dim=-1)   # (B, K)
        if self.classifier.bias is not None:
            logits = logits + self.classifier.bias.unsqueeze(0)

        return logits   # (B, K), apply sigmoid externally

    def forward_for_label(
        self,
        embedding_input: torch.Tensor,
        label_idx: int,
    ) -> torch.Tensor:
        """Forward pass returning scalar logit for label `label_idx`.

        Used by captum Integrated Gradients which requires a scalar output.
        `embedding_input` is the input embedding tensor (from BERT embedding layer).

        Note: this method is called by the interpretability module; it expects
        that self.bert's forward accepts inputs_embeds directly.
        """
        out       = self.bert(inputs_embeds=embedding_input)
        cls_emb   = out.last_hidden_state[:, 0, :]            # (B, 768)
        cls_emb   = self.bert_dropout(cls_emb)
        node_embs = self.encode_labels()                       # (K, gcn_out)
        fused     = self.fusion(cls_emb, node_embs)            # (B, K, gcn_out)
        logits    = (fused * self.classifier.weight.unsqueeze(0)).sum(dim=-1)
        if self.classifier.bias is not None:
            logits = logits + self.classifier.bias.unsqueeze(0)
        return logits[:, label_idx].unsqueeze(-1)              # (B, 1)

    def get_probabilities(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        num_chunks:     torch.Tensor,
    ) -> torch.Tensor:
        """Return per-label sigmoid probabilities."""
        logits = self.forward(input_ids, attention_mask, num_chunks)
        return torch.sigmoid(logits)
