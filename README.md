# smallest-llm

Train real language models on a MacBook. Frontier techniques, crammed into the fewest parameters possible.

## What This Is

A complete LLM training framework — models, training loop, evaluation, and a real-time dashboard — all running locally on Apple Silicon. The goal: **how much capability can you squeeze into 300K-10M parameters?**

## Models

| Architecture | Description |
|---|---|
| **Transformer** | Baseline with RMSNorm, RoPE, SwiGLU, GQA, tied embeddings |
| **Mamba** | Selective state space model (SSM) |
| **Mamba-2** | Chunked SSD with multi-head state |
| **Mamba-3** | SSD + trapezoidal discretization + RoPE + MLP blocks |
| **Improved Mamba-3** | Fused SSD, gradient checkpointing |

## Training Innovations

**Proven** (from ablation experiments):
- **Muon optimizer** — 0.5 lower val loss than AdamW
- **Muon + Echo Loss** — marginal gain over Muon alone

**Experimental:**
- **State Anchoring** — forces SSM to retain information across distances
- **Gradient Sharpening** — keeps top-K% of gradient components, zeros noise
- **Neuroplasticity** — progressive model growing
- **Phantom Batches** — correlated pseudo-gradients from hidden states
- **Hydra Training** — multi-scale sequence training (micro + macro)

**Multi-stage pipelines:**
- Curriculum learning (short sequences → long)
- Multi-dataset training (TinyStories → MiniPile → OpenWebText)
- Supervised fine-tuning (SFT) stage with UltraChat

## Evaluation

- **Built-in tasks:** perplexity, BLiMP grammar probes, LAMBADA, generation quality, state tracking
- **lm-evaluation-harness:** HellaSwag, ARC, WinoGrande, PIQA, BoolQ, MMLU, TruthfulQA, GSM8K
- **Compare** trained checkpoints against HF reference models (SmolLM-135M, Qwen2.5-0.5B)

## Dashboard

React + TypeScript web UI with:
- Real-time training metrics via WebSocket (loss, LR, grad norm, tokens/sec)
- Loss curves and evaluation results
- Run management (start, stop, delete, compare)
- Model evaluation with progress tracking
- Chat/generation playground
- Checkpoint weight inspection

## Quick Start

```bash
# Setup
git clone <repo-url>
cd smallest-llm
uv sync

# Train a model
uv run python main.py                              # default config
uv run python main.py --preset mamba3-quick         # named preset
uv run python main.py --preset mamba3-curriculum    # multi-stage training
uv run python main.py --config configs/example.json # custom config
uv run python main.py --optimizer.lr 1e-3           # CLI overrides

# Dashboard
uv run python scripts/serve.py   # backend (localhost:8000)
cd dashboard && npm install && npm run dev  # frontend (localhost:5173)

# Tests
uv run pytest

# Evaluate
uv run python scripts/eval_harness.py  # lm-evaluation-harness benchmarks
```

## Presets

Quick experiments:
- `sanity-check` — overfit 1 batch, verify training works
- `transformer-quick` / `mamba-quick` / `mamba3-quick` — 200 steps

Standard training:
- `mamba3-default` — 1K steps on MiniPile
- `transformer-long` — 5K steps

Multi-stage:
- `mamba3-curriculum` — 4 stages, progressive sequence length (32→256)
- `mamba3-multistage` — 4 stages across TinyStories → MiniPile → OpenWebText
- `mamba3-sft` — pretrain + diversify + supervised fine-tuning

Kitchen sink:
- `improved-mamba3-100x` — Muon + Echo + Phantom + Hydra + Neuroplasticity + multi-stage + SFT

## Project Structure

```
smallest-llm/
├── main.py                  # CLI entry point
├── src/
│   ├── config/              # Dataclass configs & presets
│   ├── data/                # Dataset loading, tokenization, streaming
│   ├── evaluation/          # Eval framework + lm-harness integration
│   ├── models/              # Transformer, Mamba, Mamba-2, Mamba-3
│   ├── training/            # Training loop, optimizers, checkpointing, pipelines
│   ├── server/              # FastAPI + WebSocket backend
│   ├── storage/             # SQLite persistence (metrics, checkpoints, evals)
│   └── types/               # Shared type definitions
├── dashboard/               # React + Vite + Jotai frontend
├── configs/                 # JSON config files
├── scripts/                 # Utility scripts (serve, eval, ablation)
├── data/                    # Training data (Shakespeare, eval datasets)
├── checkpoints/             # Saved model checkpoints
└── tests/                   # pytest suite
```

## Tech Stack

- **Python** + **uv** for package management
- **PyTorch** (MPS backend for Apple Silicon)
- **FastAPI** + **WebSockets** for the dashboard server
- **React** + **TypeScript** + **Vite** + **Jotai** for the dashboard UI
- **SQLite** for metrics, checkpoints, and eval results
- **lm-eval** for standard benchmarks

## Hardware

Designed for Apple Silicon MacBooks. No cloud, no multi-GPU — just your laptop.

## License

TBD
