"""Pure-PyTorch Mamba (selective state space model). No CUDA kernels — runs on MPS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel, ModelOutput


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """One Mamba block: norm -> selective SSM with gating -> residual."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 8,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.norm = RMSNorm(d_model)

        # Two-branch input projection
        self.in_proj_x = nn.Linear(d_model, d_inner)
        self.in_proj_z = nn.Linear(d_model, d_inner)

        # Causal depthwise conv1d
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner,
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)

        # Learnable SSM parameters
        # A: log-parameterized diagonal state matrix (HiPPO-inspired init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Two branches
        x_branch = self.in_proj_x(x)
        z = self.in_proj_z(x)

        # Causal conv1d: (B, L, D) -> (B, D, L) -> conv -> trim -> (B, L, D)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x_branch = F.silu(x_branch)

        # Project to SSM params: dt, B, C
        x_dbl = self.x_proj(x_branch)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Delta (timestep)
        dt = F.softplus(self.dt_proj(dt))

        # State matrix
        A = -torch.exp(self.A_log)

        # Selective scan
        y = self._selective_scan(x_branch, dt, A, B, C)

        # Gate and project
        output = y * F.silu(z)
        output = self.out_proj(output)

        return residual + output

    def _selective_scan(self, x, delta, A, B, C):
        """Pure PyTorch selective scan (sequential over time).

        x:     (B, L, d_inner)
        delta: (B, L, d_inner)
        A:     (d_inner, d_state)
        B:     (B, L, d_state)
        C:     (B, L, d_state)
        """
        batch, seq_len, d_inner = x.shape

        # Discretize: deltaA = exp(delta * A)
        # delta: (B, L, d_inner, 1) * A: (1, 1, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB * x: (B, L, d_inner, d_state)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        # Sequential scan
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class TinyMamba(BaseModel):
    """Small Mamba model. Same interface as TinyTransformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 7,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: int = 0,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        d_inner = d_model * expand_factor

        if dt_rank <= 0:
            dt_rank = math.ceil(d_model / 16)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
            for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.drop(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return ModelOutput(loss=loss, logits=logits, hidden_states=x)
