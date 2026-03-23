"""
integrated_gradients.py
-----------------------
Integrated Gradients (IG) attribution for CLiGNet.

IG computes the token-level attribution scores for a given label prediction
by integrating gradients along a straight path from a baseline input (zero
embeddings) to the actual input embedding.

This implements Contribution C3 of the CLiGNet paper:
    Per-label token attributions with a 4-step clinical validation protocol.

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks",
ICML 2017.

Usage
-----
    explainer = CLiGNetExplainer(model, tokenizer, device)
    attributions = explainer.explain(text, label_idx, n_steps=50)
    top_tokens = explainer.top_k_tokens(attributions, text, k=5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CLiGNetExplainer:
    """Integrated Gradients explainer for CLiGNet.

    Parameters
    ----------
    model     : trained CLiGNet model (in eval mode)
    tokenizer : Bio_ClinicalBERT tokenizer
    device    : torch device
    """

    def __init__(self, model, tokenizer, device: torch.device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.model.eval()

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get BERT token embeddings for input_ids."""
        return self.model.bert.embeddings.word_embeddings(input_ids)

    def _forward_from_embeddings(
        self,
        embeddings: torch.Tensor,   # (1, seq_len, 768)
        label_idx: int,
    ) -> torch.Tensor:
        """Forward pass using input embeddings directly (for IG).

        Returns scalar logit for label_idx.
        """
        # Pass embeddings to BERT
        attention_mask = torch.ones(
            embeddings.shape[0], embeddings.shape[1], device=self.device
        )
        out    = self.model.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]   # (1, 768)

        # GCN on label graph
        node_embs = self.model.encode_labels()     # (K, gcn_out)

        # Fusion
        fused  = self.model.fusion(cls_emb, node_embs)   # (1, K, gcn_out)

        # Logit for target label
        logit = (fused[:, label_idx, :] * self.model.classifier.weight[label_idx]).sum()
        if self.model.classifier.bias is not None:
            logit = logit + self.model.classifier.bias[label_idx]
        return logit

    @torch.no_grad()
    def _get_baseline_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Zero embedding baseline (all-pad baseline)."""
        pad_id   = self.tokenizer.pad_token_id
        baseline = torch.full_like(input_ids, pad_id)
        return self._get_embeddings(baseline)

    def explain(
        self,
        text: str,
        label_idx: int,
        n_steps: int = 50,
        max_length: int = 512,
    ) -> Dict:
        """Compute Integrated Gradients attributions for `label_idx`.

        Parameters
        ----------
        text      : raw transcription string
        label_idx : integer index of the specialty label to explain
        n_steps   : number of IG integration steps (default 50)
        max_length : tokenization max length

        Returns
        -------
        dict with:
            tokens       : list of subword token strings
            attributions : (seq_len,) float numpy array of attribution scores
            convergence_delta : |sum(attr) - (F(x) - F(baseline))| (should be small)
            label_idx    : int
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        input_ids    = enc["input_ids"]            # (1, L)
        token_ids    = input_ids[0].cpu().tolist()
        tokens       = self.tokenizer.convert_ids_to_tokens(token_ids)

        input_embeds    = self._get_embeddings(input_ids)     # (1, L, 768)
        baseline_embeds = self._get_baseline_embeddings(input_ids)  # (1, L, 768)

        # Integrated Gradients: approximate integral with n_steps
        integrated_grads = torch.zeros_like(input_embeds)
        for alpha in torch.linspace(0, 1, n_steps):
            interp = baseline_embeds + alpha.item() * (input_embeds - baseline_embeds)
            interp = interp.detach().requires_grad_(True)
            logit  = self._forward_from_embeddings(interp, label_idx)
            logit.backward()
            integrated_grads += interp.grad / n_steps

        attributions = (integrated_grads * (input_embeds - baseline_embeds)).detach()
        # Summarise over embedding dimension -> per-token scalar
        attr_scores = attributions[0].sum(dim=-1).cpu().numpy()   # (L,)

        # Convergence check
        f_x        = self._forward_from_embeddings(input_embeds, label_idx).item()
        f_baseline = self._forward_from_embeddings(baseline_embeds, label_idx).item()
        completeness_delta = abs(attr_scores.sum() - (f_x - f_baseline))

        return {
            "tokens":             tokens,
            "attributions":       attr_scores,
            "convergence_delta":  float(completeness_delta),
            "label_idx":          label_idx,
            "f_x":                float(f_x),
            "f_baseline":         float(f_baseline),
        }

    def top_k_tokens(
        self,
        explanation: Dict,
        k: int = 5,
        exclude_special: bool = True,
    ) -> List[Dict]:
        """Return the top-k tokens by absolute attribution score.

        Parameters
        ----------
        explanation      : result dict from explain()
        k                : number of top tokens to return
        exclude_special  : if True, skip [CLS], [SEP], [PAD] tokens

        Returns
        -------
        list of dicts: [{token, attribution, position}, ...]
        """
        tokens       = explanation["tokens"]
        attr_scores  = explanation["attributions"]
        special_toks = {"[CLS]", "[SEP]", "[PAD]"}

        indexed = [
            {"token": tok, "attribution": float(attr_scores[i]), "position": i}
            for i, tok in enumerate(tokens)
            if not (exclude_special and tok in special_toks)
        ]

        indexed.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return indexed[:k]

    def explain_batch(
        self,
        texts: List[str],
        label_indices: List[int],
        n_steps: int = 50,
        max_length: int = 512,
    ) -> List[Dict]:
        """Explain a list of (text, label_idx) pairs.

        Returns list of explanation dicts, one per text.
        """
        results = []
        for i, (text, label_idx) in enumerate(zip(texts, label_indices)):
            logger.debug("Explaining document %d / %d (label=%d)", i+1, len(texts), label_idx)
            exp = self.explain(text, label_idx, n_steps, max_length)
            exp["text_preview"] = text[:100]
            results.append(exp)
        return results


def compute_umls_signal_rate(
    top_tokens: List[Dict],
    specialty_name: str,
    nlp=None,   # optional spaCy NLP with scispaCy UMLS linker
) -> float:
    """Estimate UMLS signal recovery rate for top tokens.

    If scispaCy + UMLS linker is available, checks whether each top token
    appears as a UMLS clinical concept relevant to the specialty.
    Without scispaCy, returns -1.0 (not computed).

    Parameters
    ----------
    top_tokens     : list of dicts from top_k_tokens()
    specialty_name : name of the predicted specialty
    nlp            : scispaCy pipeline with UMLS linker (optional)

    Returns
    -------
    rate : fraction of top tokens that are UMLS concepts (0.0–1.0),
           or -1.0 if nlp not available
    """
    if nlp is None:
        logger.debug("UMLS signal check skipped: no scispaCy pipeline provided.")
        return -1.0

    matched = 0
    for tok_info in top_tokens:
        token_text = tok_info["token"].replace("##", "")   # remove BPE prefix
        doc = nlp(token_text)
        for ent in doc.ents:
            if hasattr(ent._, "umls_ents") and ent._.umls_ents:
                matched += 1
                break

    rate = matched / len(top_tokens) if top_tokens else 0.0
    return rate
