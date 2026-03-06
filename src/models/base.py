"""Model interface — any model must satisfy this contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.types.flops import FlopsEstimate


@dataclass
class ModelOutput:
    """Standard model output."""
    loss: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None


class BaseModel(nn.Module):
    """Interface for all models. Subclass this and implement forward."""

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with optional top-k sampling."""
        self.eval()
        with torch.no_grad():
            max_ctx = getattr(self, "max_seq_len", None)
            for _ in range(max_new_tokens):
                # Crop to max sequence length if needed
                ctx = input_ids if max_ctx is None else input_ids[:, -max_ctx:]
                logits = self.forward(ctx).logits
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def grow(self, **kwargs) -> BaseModel:
        """Return a new, larger model with weights copied from self.

        Subclasses should override this for architecture-specific weight surgery.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support grow()")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_flops(self, seq_len: int) -> FlopsEstimate:
        """Estimate per-token FLOPs for forward and backward passes.

        Walks modules and sums contributions:
        - nn.Linear: fwd = 2*m*n, bwd = 4*m*n per token
        - nn.Conv1d: fwd = 2*(in/groups)*out*kernel, bwd = 4x that
        - Attention (QK^T + attn@V): fwd = 4*seq_len*d_attn, bwd = 8x
        - SSM scan: fwd = 4*d_inner*d_state, bwd = 8x

        Weight-tied modules are deduplicated by data_ptr.
        """
        fwd = 0
        bwd = 0
        seen_ptrs: set[int] = set()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                ptr = module.weight.data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                mn = module.in_features * module.out_features
                fwd += 2 * mn
                bwd += 4 * mn

            elif isinstance(module, nn.Conv1d):
                k = module.kernel_size[0]
                flops = (module.in_channels // module.groups) * module.out_channels * k
                fwd += 2 * flops
                bwd += 4 * flops

        # Attention quadratic term (QK^T + attn@V)
        for module in self.modules():
            cls_name = type(module).__name__.lower()
            if "attention" not in cls_name:
                continue
            if hasattr(module, "n_heads") and hasattr(module, "head_dim"):
                d_attn = module.n_heads * module.head_dim
            elif hasattr(module, "qkv"):
                d_attn = module.qkv.in_features
            else:
                continue
            # Two matmuls (QK^T and attn@V), each 2*seq_len*d_attn per token
            fwd += 4 * seq_len * d_attn
            bwd += 8 * seq_len * d_attn

        # SSM scan cost
        for module in self.modules():
            if hasattr(module, "d_inner") and hasattr(module, "d_state"):
                cls_name = type(module).__name__.lower()
                if "mixer" in cls_name or "block" in cls_name:
                    dn = module.d_inner * module.d_state
                    fwd += 4 * dn
                    bwd += 8 * dn

        return FlopsEstimate(forward=fwd, backward=bwd)
