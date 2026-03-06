# Findings

Empirical results from ablation experiments and diagnostic probes. All experiments use improved_mamba3, d_model=64, n_layers=4, tiny_shakespeare (char tokenizer), 500 steps, seed=42.

## Ablation Results

### What Works

| Technique | Val Loss | Δ vs Baseline | Wall Time | Verdict |
|-----------|----------|---------------|-----------|---------|
| **Muon optimizer** | 1.73 | -0.49 | 1.15x | **Clear winner** |
| **Muon + Echo Loss** | 1.72 | -0.50 | 1.25x | **Best combo** |

Muon converges in ~2x fewer steps than AdamW. Echo loss adds a marginal gain on top of Muon but is neutral without it.

### What's Neutral

| Technique | Val Loss | Δ vs Baseline | Wall Time | Notes |
|-----------|----------|---------------|-----------|-------|
| Echo Loss (alone) | 2.31 | +0.09 | 1.03x | Neutral without Muon |
| Phantom Batches | 2.23 | +0.01 | 1.05x | Correlated pseudo-gradients ≠ real data diversity |
| State Anchoring | 2.24 | +0.02 | 1.13x | Auxiliary loss can't fix architectural retention problem |
| Hydra Training | 2.17 | -0.05 | 2.45x | Tiny quality gain, 2.5x wall-clock cost |

### What Hurts

| Technique | Val Loss | Δ vs Baseline | Wall Time | Notes |
|-----------|----------|---------------|-----------|-------|
| Neuroplasticity | 2.36 | +0.14 | 0.58x | Needs more steps to recover from growth |
| Gradient Sharpening | 2.34 | +0.12 | 1.10x | At this scale, "noisy" gradient components are signal |
| Muon + Sharpening | 4.42 | +2.20 | 1.26x | **Catastrophic.** Sharpening destroys Muon's orthogonalized gradients |
| New-stack (all new) | 4.78 | +2.56 | 1.42x | Gradient norms explode to 2M+, loss diverges |

### Key Insight: Gradient Sharpening + Muon Are Incompatible

Muon orthogonalizes gradients via Newton-Schulz iterations, carefully structuring the update. Gradient sharpening then zeros 90% of components, destroying that structure. The result is catastrophic divergence with gradient norms exploding from 127K → 2.1M.

## Diagnostic Probes

### Per-Token Loss Distribution
- mean=3.65, std=0.43, median=3.69
- 0% of tokens below loss 1.0
- **All tokens are hard.** No easy tokens to deprioritize. Importance weighting won't help.

### Gradient Cosine Similarity
- mean=0.34, range=[0.19, 0.57]
- **Moderate agreement.** Gradients point in roughly similar directions across examples. Gradient surgery would provide modest benefit at best.

### Layer Gradient Magnitudes
- Layer 0: 0.244, Layer 1: 0.168, Layer 2: 0.087, Layer 3: 0.104
- First/Last ratio: 2.36
- **Reasonably balanced.** No severe vanishing/exploding gradient across layers. Layer-local auxiliary heads not needed.

### SSM State Retention (the critical finding)
- k=1: 0.209 accuracy
- k=2: 0.181 accuracy
- k=4: 0.152 accuracy
- k=8: 0.142 accuracy
- k=16: 0.140 accuracy

A linear probe on the final hidden state can barely recover tokens from even 1 step back (21% accuracy on a 65-char vocab where random ≈ 1.5%). **The SSM state is not retaining information.** This is the fundamental bottleneck.

State anchoring (auxiliary loss to encourage retention) didn't help. The problem is architectural — the state doesn't have enough capacity or the discretization loses information too quickly.

## Conclusions

1. **Muon is the only training technique that clearly helps.** Use it for all future runs.
2. **Training tricks are exhausted at this scale.** Echo, phantom, anchoring, sharpening — none move the needle meaningfully.
3. **The bottleneck is state retention.** The SSM forgets almost everything beyond the last 1-2 tokens.
4. **The path forward is architectural:** bigger d_state, different discretization, attention hybrids, or fundamentally different state update mechanisms.
5. **Scale experiments are needed:** is the retention problem specific to d_model=64/L=4, or does it persist at d_model=128/L=8?
