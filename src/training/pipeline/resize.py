"""Embedding resize utility for SFT stage transitions."""

from __future__ import annotations

import torch
import torch.nn as nn


def _resize_embeddings(model: nn.Module, new_vocab_size: int):
    """Resize token embedding and output head to accommodate new tokens.

    Preserves existing weights and initializes new rows with small random values.
    Handles weight-tied models (where head.weight is token_emb.weight).
    """
    if not hasattr(model, "token_emb"):
        raise ValueError("Model must have a 'token_emb' attribute for embedding resize")

    old_emb = model.token_emb
    old_vocab_size, d_model = old_emb.weight.shape

    if new_vocab_size <= old_vocab_size:
        return

    # Create new embedding
    new_emb = nn.Embedding(new_vocab_size, d_model, device=old_emb.weight.device)
    nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = old_emb.weight
    model.token_emb = new_emb

    # Update head if it exists and was weight-tied
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        old_head = model.head
        weight_tied = old_head.weight.data_ptr() == old_emb.weight.data_ptr()

        if weight_tied:
            model.head = nn.Linear(d_model, new_vocab_size, bias=False, device=old_emb.weight.device)
            model.head.weight = model.token_emb.weight
        else:
            new_head = nn.Linear(d_model, new_vocab_size, bias=old_head.bias is not None, device=old_emb.weight.device)
            nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                new_head.weight[:old_vocab_size] = old_head.weight
                if old_head.bias is not None:
                    new_head.bias[:old_vocab_size] = old_head.bias
            model.head = new_head
