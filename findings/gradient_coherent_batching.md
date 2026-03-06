# Gradient-Coherent Batching Experiment

Testing whether grouping training sequences by gradient similarity improves gradient efficiency beyond the 1/sqrt(n) bound.

## Setup

- Model: transformer, d_model=128, n_heads=4, n_layers=4 (834K params)
- Data: TinyStories (streaming), byte tokenizer, seq_len=64
- 64 sequences fingerprinted, clustered into 8 groups of 8
- Batch size: 8, giving 512 tokens per batch
- Reproduce: `uv run python findings/scripts/gradient_coherent_batching.py`

## Method

1. Compute per-sequence gradient directions (normalized gradient vectors) for 64 TinyStories sequences
2. Cluster sequences by cosine similarity of their gradient fingerprints
3. Measure gradient efficiency (||mean(g_i)|| / mean(||g_i||)) for:
   - Random batches (random 8 of 64 sequences)
   - Coherent batches (8 sequences from same cluster)
   - Adversarial batches (1 sequence from each of 8 different clusters)
4. Compare against the 1/sqrt(n_tokens) theoretical bound for random vectors

## Key Discovery: The 1/sqrt(n) Bound Doesn't Apply to Real Data

| Data Type | Efficiency | Theory (1/sqrt(512)) | Ratio | Waste |
|-----------|-----------|---------------------|-------|-------|
| Random tokens | 0.0431 | 0.0442 | 0.98x | 95.7% |
| Real text (TinyStories) | 0.1880 | 0.0442 | 4.25x | 81.2% |

The 1/sqrt(n) bound matches perfectly for random data (0.98x theory) but real text already exceeds it by 4.25x. The "90% gradient waste" finding from `gradient_interference.md` was measured on random data and **overstates the problem for real training**.

## Batching Strategy Results (Real Data)

| Strategy | Efficiency | Waste | vs Random Batch |
|----------|-----------|-------|-----------------|
| Random batching | 0.2087 | 79.1% | baseline |
| Coherent batching | 0.2144 | 78.6% | +2.7% |
| Adversarial batching | 0.2018 | 79.8% | -3.3% |

## Why Coherent Batching Barely Helps

Pairwise gradient cosine similarity between sequences:

| Metric | Value |
|--------|-------|
| Overall mean cosine | 0.6501 |
| Overall std | 0.0647 |
| Min cosine (most different pair) | 0.4109 |
| Max cosine (most similar pair) | 0.8815 |
| Within-cluster cosine | 0.6860 |
| Between-cluster cosine | 0.6456 |
| Random batch avg cosine | 0.6532 |
| Coherent batch avg cosine | 0.6947 |

All TinyStories sequences already produce highly correlated gradients (mean cosine 0.65). Even the most dissimilar pair has 0.41 cosine similarity. There's simply not enough variance in gradient directions to exploit via clustering.

## Conclusions

1. **The 1/sqrt(n) bound only applies to random/uncorrelated data.** Real language data has inherently correlated gradients because the model needs similar updates for similar text.

2. **The "90% gradient waste" finding is misleading for real training.** With real TinyStories data, waste is ~80% (not 95%), and the efficiency is 4x higher than the random-vector prediction.

3. **Gradient-coherent batching provides negligible benefit (+2.7%).** The gradients are already so aligned across all TinyStories sequences that there's almost no room to improve by clustering.

4. **Data diversity, not data similarity, is the bottleneck.** The remaining ~80% waste comes from within-sequence token diversity (different positions in the same sequence want different things), not from between-sequence conflicts.

5. **This experiment would be more interesting with highly diverse data** (e.g., mixing code, prose, math, dialogue). In that case, between-sequence gradient conflict would be much larger, and coherent batching could matter more. For a homogeneous dataset like TinyStories, it's a non-starter.
