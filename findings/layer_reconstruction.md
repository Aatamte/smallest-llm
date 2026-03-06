# Layer Reconstruction Loss Experiment

Testing whether an auxiliary per-layer reconstruction loss improves training by penalizing information destruction.

## Setup

- Model: improved_mamba3, d_model=128, n_layers=4, d_state=16, expand_factor=2
- Data: TinyStories (streaming), byte tokenizer, seq_len=64, batch=16
- Optimizer: Adam, lr=1e-3, grad_clip=1.0
- 500 training steps, eval every 50 on fixed validation batches
- All runs start from identical model weights
- Reproduce: `uv run python findings/scripts/layer_reconstruction.py`

## Method

For each layer transition (i → i+1), a learned linear map reconstructs the input from the output. Loss = normalized MSE, weighted and added to the standard cross-entropy loss. The reconstruction map is initialized to identity.

## Results: Validation Loss

| Step | standard | recon_0.01 | recon_0.1 |
|------|----------|------------|-----------|
| 0 | 5.319 | 5.497 | 5.526 |
| 50 | 2.289 | 2.283 | 2.282 |
| 100 | 1.948 | 1.884 | 1.861 |
| 150 | 1.752 | 1.692 | 1.675 |
| 200 | 1.606 | 1.572 | 1.589 |
| 250 | 1.525 | 1.522 | 1.497 |
| 300 | 1.464 | 1.467 | 1.477 |
| 350 | 1.412 | 1.458 | 1.450 |
| 400 | 1.406 | 1.395 | 1.389 |
| 450 | 1.368 | 1.337 | 1.394 |
| 500 | 1.291 | 1.358 | 1.340 |

## Results: Layer Invertibility (R²) After Training

| Layer | standard | recon_0.01 | recon_0.1 |
|-------|----------|------------|-----------|
| L0→L1 | 89.8% | 99.0% | 99.6% |
| L1→L2 | 96.8% | 94.5% | 96.4% |
| L2→L3 | 96.0% | 96.3% | 96.2% |
| L3→L4 | 94.9% | 92.0% | 92.1% |
| Overall | 80.2% | 79.8% | 83.9% |

## Observations

1. **Reconstruction loss helps early training, hurts late.** Both recon variants converge faster initially (1.861-1.884 at step 100 vs 1.948 standard) but standard catches up and slightly wins by step 500 (1.291 vs 1.340-1.358).

2. **First layer invertibility dramatically improves.** L0→L1 goes from 89.8% (standard) to 99.0-99.6% (recon). The reconstruction loss successfully prevents the first layer from destroying information.

3. **Overall invertibility is similar.** The recon_0.1 variant has slightly better overall R² (83.9% vs 80.2%), but recon_0.01 is similar to standard (79.8%).

4. **The reconstruction loss may be constraining the model too much in later training.** By forcing layers to be invertible, we prevent them from learning useful compressions that discard task-irrelevant information. The initial speed boost (layers preserve info → downstream layers have more to work with) gives way to a representational bottleneck (layers can't specialize).

## Conclusions

- Layer reconstruction is a valid auxiliary loss that demonstrably improves first-layer information preservation.
- It provides faster early convergence but may constrain late-stage optimization.
- A potential improvement: anneal the reconstruction weight to zero over training (preserve info early when the model doesn't know what's useful yet, allow compression later when it does).
- Another angle: apply reconstruction loss only to the first 1-2 layers where information destruction is worst, rather than all layers.
