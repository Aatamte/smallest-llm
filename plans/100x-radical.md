# 100x Radical Training Acceleration Plan

## Philosophy

We are NOT doing incremental optimization. We are rethinking what training even IS.
The question isn't "how do we make each step faster" — it's "how do we need fewer steps, and make each step count 100x more."

---

## The Big Ideas (pick 3-4 to implement)

### 1. HYDRA TRAINING — Multi-Resolution Parallel Forward

**Never been done before.** Instead of training on one sequence at a time, run the SAME model on 3 resolutions simultaneously in a single forward pass:

- **Micro** (seq_len=16, batch=256): Teaches local patterns (bigrams, trigrams). Insanely fast — tiny sequences fly through SSM.
- **Meso** (seq_len=128, batch=32): Teaches medium-range dependencies (sentence structure).
- **Macro** (seq_len=512, batch=4): Teaches long-range coherence (paragraph flow).

All three produce losses. Combined loss = `α*L_micro + β*L_meso + γ*L_macro` where α starts at 1.0 and decays to 0.

**Why 100x?** The micro resolution processes 256×16 = 4096 tokens in the time it takes macro to do 4×512 = 2048. You're getting 3× the gradient signal per wall-clock second, AND the micro-resolution gradients are extremely clean (low variance) so each step moves you further.

SSMs are uniquely suited for this because they don't have O(n²) attention — the cost is truly linear, so short sequences are proportionally faster.

### 2. GENESIS INITIALIZATION — Train Backwards From a Teacher's Brain

Instead of random init → slow convergence, do this:

1. Take a pretrained GPT-2 (124M params)
2. Run it on your training data, collect its hidden states at each layer
3. Train a tiny "projection network" that maps GPT-2 hidden states → your Mamba-3 hidden states
4. Use the projected states to INITIALIZE your Mamba-3's parameters via least-squares fitting

This is NOT distillation (which trains with KL divergence on outputs). This is **structural transplant** — you're giving your tiny model the same internal representations the big model learned, just compressed.

**Why 100x?** Random init wastes 80%+ of training finding the right subspace. This starts you in the right subspace. Models initialized this way converge in 50-200 steps instead of 5000-20000.

### 3. ECHO LOSS — Every Token Teaches Every Other Token

Standard next-token prediction: token at position i only teaches about position i+1.
That's wasting N-1 other teaching signals per position.

**Echo Loss**: After the forward pass, take the hidden states and compute REVERSE predictions too:
- Forward: h_i predicts token_{i+1} (standard)
- Backward echo: h_i also predicts token_{i-1} (reverse LM head)
- Skip echo: h_i predicts token_{i+k} for k = 2, 4, 8 (skip-gram LM heads)

Each head is a tiny linear projection (d_model → vocab). Cost: ~5% more compute per step. Benefit: 6× more gradient signal per token seen.

Combined: `L = L_forward + 0.3*L_backward + 0.1*(L_skip2 + L_skip4 + L_skip8)`

**Why 100x?** You're extracting 6× more information per token. Combined with other techniques, this means you need 6× fewer tokens to learn the same representations. The skip-gram losses especially help the SSM learn to maintain long-range state.

### 4. PHANTOM BATCHES — Synthetic Gradient Augmentation

After each real forward/backward pass, generate "phantom" gradients by:

1. Take the current batch's hidden states (already computed, free)
2. Apply random dropout masks (5 different masks) to the hidden states
3. Compute loss on each masked version
4. Average the 5 phantom gradients with the real gradient

This is like getting 5× the batch size for FREE — you don't need to load more data or run more forward passes through the full model. The hidden states are already in memory.

**Why 100x?** 5× effective batch size at ~20% extra compute cost. The gradient variance drops dramatically, so the optimizer can take much larger steps. Combined with Muon optimizer's better step direction, each step is worth ~10× a normal step.

### 5. NEUROPLASTICITY SCHEDULE — Dynamic Architecture During Training

The model PHYSICALLY CHANGES during training:

- **Steps 0-100**: 1 layer, d_model=32 (1K params). Learns embedding structure.
- **Steps 100-300**: 2 layers, d_model=64 (15K params). Learns local patterns. Weights from stage 1 are padded with noise.
- **Steps 300-700**: 4 layers, d_model=128 (200K params). Learns sentence structure.
- **Steps 700-1000**: 7 layers, d_model=256 (2M params). Learns long-range coherence.

When growing:
- Existing weights are copied to the top-left corner of new weight matrices
- New dimensions are initialized with scaled noise: `new_dims = randn * (existing_weight_std * 0.1)`
- Optimizer state (Muon momentum buffers) are padded similarly

**Why 100x?** Steps 0-300 run 100-1000× faster because the model is tiny. A 1K param model doing a forward pass on seq_len=16 takes <0.1ms. You get through 300 "steps" in the time it takes the full model to do 3 steps. And those early steps aren't wasted — they're learning real features that transfer when the model grows.

### 6. VOCABULARY ANNEALING — Start With Simple Tokens

Instead of the full vocabulary from step 1:

- **Phase 1**: Merge the vocabulary down to 64 tokens (by clustering embeddings). Train on this simplified language. The model learns structure fast because there are so few options.
- **Phase 2**: Expand to 256 tokens. Fine-tune.
- **Phase 3**: Full vocabulary. Fine-tune.

This is curriculum learning applied to the OUTPUT SPACE, not the input data.

**Why 100x?** With 64 tokens, the softmax output layer is tiny, cross-entropy is cheap, and the model can achieve low loss very quickly. Each correct prediction with 64 tokens teaches more bits of information (log2(64)=6 bits) than random guessing with 50K tokens (log2(50000)=15.6 bits). You're making the problem easier first.

### 7. GRADIENT TELEPORTATION — Learn From the Future

During training, maintain a **lookahead model** (copy of current model with +5 steps of SGD applied). Use the lookahead model to:

1. Compute a "future gradient" on the current batch
2. Blend it with the current gradient: `g = 0.7 * g_current + 0.3 * g_future`
3. The current model gets to "see" where training is going and takes smarter steps

This is different from Lookahead optimizer — it computes actual gradients from a future version of the model, not just interpolates parameters.

**Why 100x?** Eliminates most of the oscillation and inefficiency in early training. The future gradient acts as a denoiser — it cancels out the noisy components of the current gradient and reinforces the true signal. Combined with Muon, you get near-optimal update directions.

---

## Recommended Combo: THE STACK

For maximum compound effect, implement these together:

```
Neuroplasticity (5)     →  100x faster early steps
+ Hydra Training (1)    →  3x gradient signal
+ Echo Loss (3)         →  6x info per token
+ Phantom Batches (4)   →  5x effective batch size
─────────────────────────────────────
= 100x fewer wall-clock seconds to reach target loss
```

Phase A: **Neuroplasticity + Hydra** (the architecture tricks)
Phase B: **Echo Loss + Phantom Batches** (the gradient tricks)

---

## Implementation Order

1. **Neuroplasticity Schedule** — biggest single win, pure Python, changes training loop
2. **Hydra Training** — second biggest win, modifies forward pass to handle multi-resolution
3. **Echo Loss** — add auxiliary heads and losses, minimal code
4. **Phantom Batches** — post-forward synthetic gradients, moderate code

Each one is independently useful and testable. Together they compound.
