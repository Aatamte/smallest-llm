# Token Selection Training Strategies

Testing whether training on subsets of tokens (instead of all tokens) improves learning efficiency.

## Setup

- Model: transformer, d_model=128, n_heads=4, n_layers=4 (834K params)
- Data: TinyStories (streaming), byte tokenizer, seq_len=64, batch=16
- Optimizer: Adam, lr=1e-3, grad_clip=1.0
- 300 training steps, eval every 50 steps on fixed validation batch
- All strategies start from identical model weights
- Reproduce: `uv run python findings/scripts/token_selection.py`

## Strategies

| Strategy | Description |
|----------|-------------|
| standard | Normal training, loss averaged over all tokens |
| top_50pct_loss | Only backprop through top 50% highest-loss tokens |
| bottom_50pct_loss | Only backprop through bottom 50% lowest-loss tokens |
| random_25pct | Randomly sample 25% of tokens per step |
| alternating_halves | Even steps: first half of sequence. Odd steps: second half |

## Results

| Strategy | Step 0 | Step 100 | Step 200 | Step 300 | Total Δ |
|----------|--------|----------|----------|----------|---------|
| standard | 5.073 | 2.517 | 2.424 | 2.318 | -2.755 |
| top_50pct_loss | 5.163 | 2.934 | 2.761 | 2.688 | -2.475 |
| bottom_50pct_loss | 5.116 | 4.789 | 5.185 | 5.507 | +0.391 |
| random_25pct | 5.143 | 2.599 | 2.490 | 2.459 | -2.683 |
| alternating_halves | 5.096 | 2.524 | 2.466 | 2.369 | -2.727 |

## Observations

1. **Standard training wins.** Using all tokens with uniform averaging (2.318) beats every selective strategy.

2. **Training only on hard tokens hurts** (2.688 vs 2.318). Hard tokens produce high-magnitude gradients but they're the least reliable — the model doesn't know enough yet to predict them, so their gradients are noisy. Focusing on them amplifies noise.

3. **Training only on easy tokens diverges** (5.507, worse than start). Easy tokens have low loss and contribute almost no gradient. The model never gets signal about what it's doing wrong.

4. **Random 25% subset works surprisingly well** (2.459 vs 2.318). Using only 25% of tokens gets 97% of the learning with 25% of the gradient computation. The lost 75% of tokens were mostly adding noise that cancels out anyway.

5. **Alternating halves nearly matches standard** (2.369 vs 2.318). Training on first-half and second-half tokens on alternate steps — so each step has less internal conflict — performs almost identically to standard. This suggests the interference between position groups, while measurable, doesn't significantly hurt when Adam's momentum can smooth it out.

## Conclusions

- The gradient interference finding (90% waste) is real in terms of magnitude, but Adam's momentum buffer effectively recovers the signal over multiple steps. Simply dropping tokens doesn't beat letting momentum do its job.
- Hard-token-only training is counterproductive — it amplifies the noisiest gradients.
- Random subsampling (25%) provides nearly equal learning with less compute per step, suggesting most per-step gradient information is redundant.
- Position-based splitting doesn't help or hurt significantly — Adam already handles the conflict.
