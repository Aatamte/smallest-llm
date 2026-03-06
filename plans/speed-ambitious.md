# Ambitious Speed Plan: 10x-100x Training Acceleration

## Context

We have Mamba-1, Mamba-2, and Mamba-3 models training on Apple Silicon (MPS).
Current baseline: Mamba-3 at d_model=64, n_layers=4 does ~60K tok/sec, ~68ms/step.
Goal: explore every lever to push this 10x-100x faster.

## Phase 1: Low-Hanging Fruit (2-5x each, compound to ~10x)

### 1.1 torch.compile the models
- `TrainingConfig` already has `compile_model: bool` but it's never used in `trainer.py`
- Wire it up: `model = torch.compile(model)` in `src/training/run.py` when `config.training.compile_model=True`
- On MPS this may not help yet (compile support is limited), but on CUDA it's 1.5-3x
- Files: `src/training/run.py` (add compile call after model creation)

### 1.2 Mixed Precision (AMP)
- `TrainingConfig.mixed_precision` exists but is never used in `trainer.py`
- Add `torch.autocast` context in the training loop's forward/backward
- Add `GradScaler` for fp16 on CUDA
- Expected: 1.5-2x on CUDA, modest gains on MPS
- Files: `src/training/trainer.py` (wrap forward/backward in autocast)

### 1.3 Muon Optimizer
- Drop-in replacement for AdamW that converges in 2-5x fewer steps
- Orthogonalizes the momentum update, giving better gradient signal per step
- Implement as new option in `src/training/optimizer.py`
- Add `optimizer.name = "muon"` support in `OptimizerConfig`
- Files: `src/training/optimizer.py`, `src/config/base.py`

### 1.4 Fuse the SSD Inner Loop
- Current `_ssd()` in mamba3.py does: segsum -> exp -> einsum(CB) -> multiply(decay) -> einsum(Y)
- That's 5+ kernel launches and 5+ memory round-trips
- Rewrite as a single fused operation using `torch.compile` or manual fusion
- At minimum: precompute `CB * L` as one step, avoid materializing the full (cs, cs) matrix
- Files: `src/models/mamba3.py` (_ssd function)

### 1.5 Gradient Accumulation Tuning
- Already supported via `gradient_accumulation_steps`
- Use larger effective batch size with fewer optimizer steps
- Fewer steps = fewer scheduler/optimizer overhead calls
- This is a config change, not code change

## Phase 2: Architectural Speedups (5-20x)

### 2.1 Progressive Model Growing
- Start with d_model=16, n_layers=1 for the first 30% of training
- Double dimensions/layers at scheduled intervals
- Copy weights from smaller model into larger (with noise for new params)
- The early steps are 50-100x faster because the model is tiny
- Implementation:
  - New `ProgressiveGrowthCallback` in `src/training/callbacks.py`
  - `_grow_model()` function that creates a larger model and copies weights
  - Schedule defined in `StageConfig` or a new `GrowthConfig`
- Files: `src/training/callbacks.py`, new helper in `src/models/`

### 2.2 Token Merging / Sequence Compression
- Before feeding tokens to the SSM, merge similar adjacent token embeddings
- Reduce effective seq_len by 2-4x using cosine similarity threshold
- Unpool before computing loss (or compute loss on merged positions only)
- SSM runtime is roughly O(seq_len), so 2x shorter = ~2x faster
- Implementation:
  - New `TokenMerger` module inserted between embedding and first layer
  - Learnable merge threshold or fixed cosine-similarity cutoff
  - `TokenUnmerger` before the LM head for loss computation
- Files: new `src/models/token_merge.py`, modify model forward passes

### 2.3 Sparse Mixture-of-SSM-Experts (MoSSM)
- Each layer has K SSM experts (K=4 or 8), routes to top-1 per token
- Get K times the capacity for ~1x the compute
- Router: small linear layer from input -> K logits, pick top-1
- Load balancing loss to prevent expert collapse
- Implementation:
  - New `MoSSMLayer` wrapping multiple `Mamba3Mixer` instances
  - Top-k routing with straight-through estimator
  - Auxiliary load-balancing loss added to main loss
- Files: new `src/models/mossm.py`, integrate into mamba3 as option

### 2.4 Shared-State Across Layers
- All layers read/write a single shared SSM hidden state
- Each layer gets its own projection (view) of the shared state
- Dramatically fewer parameters, richer state per-parameter
- Implementation:
  - Modify `TinyMamba3.forward()` to pass state tensor between layers
  - Each `Mamba3Mixer` takes and returns the shared state
  - Projections per-layer to map shared state -> layer-specific B, C
- Files: `src/models/mamba3.py` (or new `src/models/mamba3_shared.py`)

## Phase 3: Radical Changes (50-100x)

### 3.1 BitNet / Ternary Training
- Replace all nn.Linear with ternary weight layers {-1, 0, +1}
- Forward: sign(weight) * scale, no multiplications needed
- Backward: straight-through estimator for gradients
- Matmuls become additions — massive speedup on any hardware
- Implementation:
  - New `TernaryLinear` module replacing `nn.Linear`
  - Custom autograd function for STE backward
  - Scale factor per output channel (learned)
  - Apply to in_proj, out_proj, SwiGLU projections
- Files: new `src/models/bitnet.py`, apply to any model via wrapper

### 3.2 Synthetic Pre-curriculum
- Before real data, train on synthetic algorithmic tasks:
  - Copying: "abc -> abc" (teaches state retention)
  - Reversal: "abc -> cba" (teaches state manipulation)
  - Counting: "aaabbb -> 3a2b" (teaches counting/accumulation)
  - Parity: "10110 -> 1" (state tracking — Mamba-3's strength)
- 100-500 steps on synthetic = 5000 steps on real data for core capabilities
- Implementation:
  - New `src/data/synthetic.py` generating algorithmic tasks
  - New stage type `"synthetic"` in StageConfig
  - Pipeline runs synthetic first, then transitions to real data
- Files: new `src/data/synthetic.py`, modify `src/training/pipeline.py`

### 3.3 Knowledge Distillation from Pretrained Teacher
- Use a pretrained GPT-2 (or any HF model) as teacher
- Student trains on soft targets (KL divergence) instead of hard labels
- Soft targets contain ~10x more information per token than one-hot labels
- Can also do progressive distillation: GPT-2 -> Mamba-3 medium -> Mamba-3 tiny
- Implementation:
  - New `DistillationTrainer` or callback that loads teacher model
  - Modified loss: `α * CE(student, labels) + (1-α) * KL(student, teacher)`
  - Teacher runs in eval mode with no_grad
  - Temperature parameter for softening distributions
- Files: new `src/training/distillation.py`, hook into trainer

### 3.4 Matmul-Free SSM
- Replace ALL linear projections with additive/ternary operations
- The SSM recurrence h = Ah + Bx becomes h = shift(h) + select(x)
- Inspired by "Scalable MatMul-Free Language Modeling" (2024)
- On Apple Silicon Neural Engine: additions at ~10x throughput of multiplications
- Implementation:
  - New `MatMulFreeSSM` module using ternary weights + additive mixing
  - Custom forward that avoids torch.mm/torch.einsum entirely
  - Activation functions via lookup tables instead of compute
- Files: new `src/models/matmul_free.py`

## Recommended Execution Order

```
Week 1: Phase 1 (all items) — compound to ~10x
  1.2 Mixed precision (quick win, already has config flag)
  1.1 torch.compile (one line, test if MPS supports it)
  1.3 Muon optimizer (drop-in, biggest step-efficiency gain)
  1.4 Fuse SSD (moderate effort, good payoff)

Week 2: Phase 2 picks — another 5-10x
  2.2 Token merging (relatively simple, immediate 2x)
  2.1 Progressive growing (ambitious but huge payoff)
  3.2 Synthetic pre-curriculum (data-side speedup)

Week 3: Phase 3 picks — push toward 100x
  3.1 BitNet ternary (radical but proven concept)
  3.3 Distillation (if teacher model available)
  2.3 MoSSM (research-grade, high risk/reward)
```

## Measurement Plan

After each change, benchmark with:
```bash
uv run python scripts/rate_model.py --model mamba3 --d_model 128 --n_layers 7
```

Track:
- tok/sec (throughput)
- ms/step (latency)
- loss at step N (convergence speed — fewer steps to same loss = faster)
- Total wall-clock to reach target loss (the real metric)

The compound effect matters: 2x from AMP * 2x from Muon * 2x from token merging * 2x from progressive growing = 16x, and that's before the radical stuff.
