"""Multi-token prediction: predict t+1, t+2, ..., t+k from each position.

Standard next-token prediction extracts one supervision signal per position.
Multi-token prediction adds k-1 additional prediction heads, giving k× the
gradient signal per forward pass at minimal extra cost.

For byte-level models this is especially valuable: predicting the next letter
of a known word is trivial, but predicting 4 bytes ahead forces the hidden
state to encode higher-level structure (word boundaries, next word choice).

Usage:
    heads = MultiTokenHeads(d_model=128, vocab_size=256, n_ahead=4)
    # In training loop:
    loss, metrics = compute_multi_token_loss(hidden_states, labels, heads)

    # For generation:
    tokens = generate_multi(model, heads, input_ids, max_new_tokens=100)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenHeads(nn.Module):
    """Prediction heads for t+2, t+3, ..., t+k.

    Head for t+1 is the model's own LM head (not duplicated here).
    Each additional head is a small linear layer: d_model -> vocab_size.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_ahead: int = 4,
    ):
        super().__init__()
        self.n_ahead = n_ahead
        # Only heads for t+2 through t+k (t+1 uses the model's own head)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_ahead - 1)
        ])
        # Small init so auxiliary losses don't dominate early
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Compute logits for t+2, t+3, ..., t+k.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            List of (batch, seq_len, vocab_size) logit tensors for offsets 2..k
        """
        return [head(hidden_states) for head in self.heads]


def compute_multi_token_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    heads: MultiTokenHeads,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute auxiliary multi-token prediction loss (t+2 through t+k).

    This is meant to be ADDED to the standard next-token loss, not replace it.
    The standard model head handles t+1 prediction; this adds supervision for
    further-ahead predictions.

    Args:
        hidden_states: (batch, seq_len, d_model) — final hidden states
        labels: (batch, seq_len) — target token IDs
        heads: MultiTokenHeads module
        weights: per-horizon weights for t+2, t+3, ..., t+k.
                 Defaults to [0.5, 0.25, 0.125, ...]

    Returns:
        (auxiliary_loss, metrics_dict)
    """
    n_aux = heads.n_ahead - 1  # number of auxiliary heads (t+2 through t+k)
    if weights is None:
        weights = [1.0 / (2 ** (i + 1)) for i in range(n_aux)]

    batch, seq_len = labels.shape
    metrics: dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=labels.device)

    # t+2 through t+k losses (from auxiliary heads)
    # labels are already shifted: labels[j] is the target for position j (= t+1)
    # So to predict t+2, we use hidden[j] to predict labels[j+1]
    aux_logits = heads(hidden_states)
    for i, aux in enumerate(aux_logits):
        offset = i + 1  # hidden[j] predicts labels[j + offset]
        horizon = i + 2  # this corresponds to predicting t+horizon
        if seq_len <= offset:
            continue
        pred = aux[:, :-offset]   # (batch, seq_len - offset, vocab)
        target = labels[:, offset:]  # (batch, seq_len - offset)
        loss_i = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1),
        )
        w = weights[i] if i < len(weights) else weights[-1]
        total_loss = total_loss + w * loss_i
        metrics[f"multi_token/t{horizon}_loss"] = loss_i.item()

    metrics["multi_token/aux_loss"] = total_loss.item()
    return total_loss, metrics


@torch.no_grad()
def generate_multi(
    model: nn.Module,
    heads: MultiTokenHeads,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    confidence_threshold: float = 0.8,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Generate tokens using multi-token prediction with confidence gating.

    At each step:
    1. Forward pass to get hidden states
    2. Predict t+1 from model head, t+2..t+k from auxiliary heads
    3. Accept consecutive predictions while confidence > threshold
    4. Append accepted tokens and repeat

    Args:
        model: language model with forward() returning ModelOutput
        heads: MultiTokenHeads module
        input_ids: (1, seq_len) input token IDs
        max_new_tokens: maximum tokens to generate
        confidence_threshold: min softmax probability to accept a prediction
        temperature: sampling temperature
        top_k: top-k filtering (None = no filtering)

    Returns:
        (1, seq_len + generated) tensor of token IDs
    """
    model.eval()
    heads.eval()
    max_ctx = getattr(model, "max_seq_len", None)
    generated = 0

    while generated < max_new_tokens:
        ctx = input_ids if max_ctx is None else input_ids[:, -max_ctx:]
        output = model(ctx)
        hidden = output.hidden_states  # (1, T, d_model)

        # t+1 prediction from model's own logits
        t1_logits = output.logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(t1_logits, min(top_k, t1_logits.size(-1)))
            t1_logits[t1_logits < v[:, [-1]]] = float("-inf")

        t1_probs = F.softmax(t1_logits, dim=-1)
        t1_conf, t1_token = t1_probs.max(dim=-1)

        # Always accept t+1 (sample if below threshold, greedy if above)
        if t1_conf.item() >= confidence_threshold:
            next_token = t1_token.unsqueeze(-1)
        else:
            next_token = torch.multinomial(t1_probs, num_samples=1)

        new_tokens = [next_token]
        generated += 1

        # Try to accept t+2, t+3, ... if confident
        if t1_conf.item() >= confidence_threshold and generated < max_new_tokens:
            aux_logits_list = heads(hidden[:, -1:, :])  # (1, 1, vocab) per head
            for aux_logits in aux_logits_list:
                if generated >= max_new_tokens:
                    break
                aux_l = aux_logits[:, -1, :] / temperature
                aux_probs = F.softmax(aux_l, dim=-1)
                aux_conf, aux_token = aux_probs.max(dim=-1)

                if aux_conf.item() >= confidence_threshold:
                    new_tokens.append(aux_token.unsqueeze(-1))
                    generated += 1
                else:
                    break

        input_ids = torch.cat([input_ids] + new_tokens, dim=1)

    return input_ids
