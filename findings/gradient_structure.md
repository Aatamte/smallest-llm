# Gradient Structure Analysis

Empirical measurements of gradient information flow during training.

## Setup

- Model: `improved_mamba3`, d_model=128, n_layers=4, d_state=16, expand_factor=2, chunk_size=64, mlp_factor=4
- Vocab: 256 (byte-level)
- Batch: 4 sequences x 64 tokens
- Data: random tokens (untrained model, measuring structural properties of gradient flow)
- Reproduce: `uv run python findings/scripts/gradient_structure.py`

## 1. Information Compression in the Loss

The model outputs 65,536 logit values (B=4, L=64, V=256). Cross-entropy reduces this to 1 scalar. The gradient then expands back to 1,238,832 values (one per parameter), but all derived from that single scalar via chain rule.

| Stage | Dimensionality | Notes |
|-------|---------------|-------|
| Model output (logits) | 65,536 | Everything the model knows |
| Loss (scalar) | 1 | 65,536:1 compression |
| Gradient | 1,238,832 | Derived from the 1 scalar |

## 2. Gradient Signal-to-Noise by Parameter Group

Per-group gradient statistics from a single batch. |Mean|/Std measures how much of the gradient is coherent signal vs noise.

| Group | #Params | Grad L2 | |Mean|/Std |
|-------|---------|---------|----------|
| embeddings | 32,768 | 1.9611 | 0.0095 |
| layer_norms | 128 | 0.0130 | 0.2031 |
| mlp | 786,944 | 0.1741 | 0.0031 |
| ssm_dynamics | 48 | 0.0157 | 0.0041 |
| ssm_norms | 1,152 | 0.0481 | 0.0019 |
| ssm_projections | 417,792 | 2.6218 | 0.0007 |

Signal-to-noise is below 0.01 for all groups except layer_norms (0.20). The gradient at any single step is dominated by noise.

## 3. Gradient Consistency Across Batches

Cosine similarity of gradients from 10 different random batches, measured per parameter group. High cosine = consistent signal, low cosine = noise.

| Group | Avg Cosine | Min | Max |
|-------|-----------|-----|-----|
| embeddings | -0.0004 | -0.0156 | 0.0230 |
| layer_norms | 0.1039 | -0.0543 | 0.3175 |
| mlp | -0.0015 | -0.0196 | 0.0110 |
| ssm_dynamics | -0.0367 | -0.6197 | 0.3700 |
| ssm_norms | -0.0093 | -0.1215 | 0.1116 |
| ssm_projections | 0.0000 | -0.0106 | 0.0092 |

All groups except layer_norms show near-zero batch-to-batch consistency. SSM dynamics are the worst — highly variable, sometimes strongly anti-correlated across batches (min cosine -0.62).

## 4. Hard vs Easy Token Gradient Divergence

Tokens split by per-token loss magnitude (top 25% = hard, bottom 25% = easy). Cosine similarity between the gradient produced by hard tokens vs easy tokens.

| Group | Cosine(hard, easy) | Hard L2 | Easy L2 |
|-------|--------------------|---------|---------|
| embeddings | 0.0534 | 4.8336 | 4.4807 |
| mlp | 0.0385 | 0.4143 | 0.3959 |
| ssm_dynamics | -0.0132 | 0.0494 | 0.0513 |
| ssm_projections | 0.0318 | 6.7494 | 6.3731 |

Cosine near zero for all groups. Hard and easy tokens produce nearly orthogonal gradient signals. SSM dynamics cosine is slightly negative — hard and easy tokens push these parameters in weakly opposing directions.

## 5. Early vs Late Position Gradient Divergence

Tokens split by sequence position: first quarter (pos 0-15) vs last quarter (pos 48-63). Cosine similarity of gradients per layer.

| Group | Cosine(early, late) | Early L2 | Late L2 |
|-------|---------------------|----------|---------|
| embeddings | 0.0069 | 3.9107 | 4.8716 |
| layer_0 | 0.0111 | 4.4080 | 5.5611 |
| layer_1 | 0.0092 | 2.8236 | 3.5518 |
| layer_2 | 0.0008 | 1.6616 | 2.2565 |
| layer_3 | 0.0049 | 1.2868 | 1.7052 |

Position-based gradient divergence is even stronger than difficulty-based. Early and late positions produce almost completely orthogonal gradients. Deeper layers are more position-agnostic (layer_2 cosine 0.0008) than shallow layers (layer_0 cosine 0.011). Late positions produce larger gradient norms across all layers.

## 6. A_log Gradient by Position

A_log controls SSM state decay rate. Gradient mean from early (pos 0-15) vs late (pos 48-63) positions, per layer.

| Layer | Early grad_mean | Late grad_mean | Direction |
|-------|-----------------|----------------|-----------|
| 0 | +0.000739 | +0.002309 | same sign, different magnitude |
| 1 | +0.000627 | -0.000554 | opposing |
| 2 | +0.000020 | +0.002106 | same sign, different magnitude |
| 3 | -0.000325 | -0.001086 | same sign, different magnitude |

Layer 1 shows opposing A_log gradients: early positions push toward slower decay (more retention), late positions push toward faster decay (more forgetting). Standard training averages these to near-zero net gradient.

## Summary

1. Single-batch gradients are mostly noise (SNR < 0.01) for all parameter groups.
2. Gradients are inconsistent across batches (cosine ~0) — signal emerges only through long averaging.
3. Hard vs easy tokens produce nearly orthogonal gradients (cosine ~0.03). Averaging them causes destructive interference.
4. Early vs late positions produce even more orthogonal gradients (cosine ~0.005-0.01). The standard loss averages away positional structure.
5. SSM decay parameters (A_log) receive conflicting gradients from different sequence positions, particularly in layer 1 where early and late positions push in opposite directions.
