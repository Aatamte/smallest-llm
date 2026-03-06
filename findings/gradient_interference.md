# Gradient Interference: A General Deep Learning Problem

## The Finding

~90% of gradient signal is destroyed by destructive interference when averaging across tokens in a single training step. This is architecture-independent and follows a precise mathematical law.

## Setup

Tested on both transformer (834K params) and Mamba SSM (1.2M params), byte-level vocab (256), various sequence lengths. Random data with untrained models to measure structural properties of gradient flow.

Reproduce: `uv run python findings/scripts/gradient_interference.py`

## 1. The Problem Is Universal

Early-position vs late-position gradient cosine similarity, and batch-to-batch gradient cosine:

| Architecture | Params | Pos Cosine | Batch Cosine |
|-------------|--------|-----------|--------------|
| transformer | 834K | 0.0833 | 0.0129 |
| improved_mamba3 | 1.2M | -0.0158 | 0.0004 |

Both architectures show near-zero cosine similarity. This is not an SSM problem or a transformer problem — it's a backprop-with-averaged-loss problem.

## 2. 90% of Gradient Signal Is Wasted

Efficiency = ||mean(g_i)|| / mean(||g_i||), where g_i is the gradient from token i.

| Architecture | Averaged Grad Norm | Avg Per-Token Norm | Efficiency | Waste |
|-------------|-------------------|-------------------|-----------|-------|
| transformer | 2.91 | 31.10 | 0.094 | 90.6% |
| improved_mamba3 | 4.91 | 55.87 | 0.088 | 91.2% |

Each individual token produces a gradient with ~30-55 norm. But when averaged across all tokens, the result has norm ~3-5. The per-token gradients are nearly orthogonal, so they mostly cancel out.

## 3. Efficiency Follows 1/sqrt(n)

| Seq Length | Measured Efficiency | Theory (1/sqrt(2*L)) | Waste |
|-----------|--------------------|--------------------|-------|
| 8 | 0.262 | 0.250 | 73.8% |
| 16 | 0.182 | 0.177 | 81.8% |
| 32 | 0.130 | 0.125 | 87.0% |
| 64 | 0.078 | 0.088 | 92.2% |
| 128 | 0.067 | 0.063 | 93.3% |

Measured efficiency tracks the theoretical prediction for random vectors (1/sqrt(n_tokens)) almost exactly. This means per-token gradients behave like random vectors in parameter space — they carry useful information individually, but that information is almost entirely lost when averaged.

## 4. Implications

**Longer sequences make it worse.** Doubling sequence length increases waste by ~5 percentage points. At seq_len=512 with batch_size=4, efficiency would be ~3% — 97% of gradient signal destroyed.

**Larger batches make it worse.** Same math applies across the batch dimension. More examples per step = more random vectors averaged = more cancellation.

**This is the fundamental reason training requires so many steps.** Each step extracts only ~3-10% of the available gradient signal. The rest is noise that cancels out. We compensate with momentum (accumulates weak signal over many steps) and thousands of iterations.

## 5. What This Does NOT Mean

- This does NOT mean training is broken — averaging over many steps still extracts the signal.
- This does NOT mean each token's gradient is useless — individually they carry strong signal (norm ~30-55).
- This does NOT mean larger batches are bad — they reduce variance, just with diminishing returns.
- This is measured on an untrained model — a trained model may have more aligned gradients for "easy" tokens, but hard/novel tokens will still produce conflicting signals.
