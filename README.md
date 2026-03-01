# smallest-llm

The smallest language model that can do the most. Frontier techniques, crammed into the fewest parameters possible.

## Goal

Build a tiny transformer LLM — under 10M parameters — that punches absurdly above its weight. Not a toy. Not a tutorial. A serious attempt to answer: **how much capability can you squeeze into the smallest possible model?**

Every architectural choice, every training technique, every byte of data is chosen to maximize capability per parameter and sample efficiency.

## Philosophy

- **Every parameter earns its keep.** No wasted capacity. If a technique exists that gets more out of fewer params, we use it.
- **Sample efficiency over scale.** We don't have billions of tokens. We make every training example count.
- **Frontier techniques at tiny scale.** The same ideas powering the largest models — adapted and tuned for the smallest.
- **Stable foundation, long-term thinking.** The training infrastructure is built to last. Clean abstractions, reproducible experiments, easy iteration.

## Architecture

Modern transformer, every component chosen for small-scale efficiency:

| Component | Choice | Why |
|---|---|---|
| Normalization | RMSNorm | Fewer params than LayerNorm, equally effective |
| Positions | RoPE | No learned embeddings to waste params on |
| Activation | SwiGLU | Better performance per FLOP than ReLU/GELU |
| Attention | GQA / MQA | Saves params on KV heads without sacrificing quality |
| Embeddings | Tied input/output | Halves embedding param count |
| Shape | Deep & narrow | More layers with fewer dims beats the reverse at small scale |

## Training Strategy

- **Tokenizer:** Small BPE vocabulary trained on our data. Smaller vocab = more parameter budget for the model itself.
- **Curriculum learning:** Easy examples first, progressively harder. Accelerates convergence.
- **Knowledge distillation:** Compress knowledge from a large teacher model into our tiny student.
- **Cosine annealing with warmup:** Standard LR schedule, tuned for our scale.
- **Mixed precision on MPS:** fp16/bf16 where possible for speed without sacrificing stability.
- **Gradient accumulation:** Simulate larger batch sizes on limited hardware.
- **Aggressive regularization:** Weight decay, dropout, gradient clipping — preventing overfit on small data.

## Hardware Target

Designed to train on a single Apple Silicon Mac:

- Apple M-series chip (10 cores, MPS GPU)
- 16 GB unified memory
- ~3,800 GFLOPS via MPS (2.8x CPU speedup)
- Tier: MEDIUM — comfortably trains models under 10M params

No cloud required. No multi-GPU. Just your laptop.

## Project Structure

```
smallest-llm/
├── main.py              # Entry point
├── scripts/
│   └── rate_infra.py    # Hardware benchmarking & capability rating
├── pyproject.toml       # Dependencies (torch, psutil)
└── README.md
```

## Getting Started

```bash
# Clone and setup
git clone <repo-url>
cd smallest-llm
uv sync

# Check what your hardware can handle
uv run python scripts/rate_infra.py
```

## Status

Early stage. Infrastructure and architecture are being built out. The foundation comes first — training loops, config management, logging, checkpointing — then the model.

## License

TBD
