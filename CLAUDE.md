# CLAUDE.md

## CRITICAL RULES

- **NEVER use sub agents. EVER.** Do not use the Agent tool under any circumstances. Always do the work directly yourself.

## Project Goal

See `goal.md` for the full project vision, success tiers (statistical coherence → actual utility), current training innovations and ablation results, evaluation strategy, and open research questions.

## Repo Structure

```
small-llm/
├── main.py                     # Entry point
├── pyproject.toml               # Project config & dependencies (uv)
├── configs/                     # JSON training configs (example.json, tiny_stories.json)
├── src/
│   ├── config/                  # Training config dataclasses & presets
│   │   ├── base.py
│   │   └── presets.py
│   ├── data/                    # Data loading & tokenization
│   │   ├── datasets.py          # HuggingFace dataset loading
│   │   ├── streaming.py         # Streaming data pipeline
│   │   ├── text.py              # Raw text file loading
│   │   └── tokenizer.py         # Tokenizer wrapper
│   ├── evaluation/              # Model evaluation framework
│   │   ├── tasks/               # Eval tasks (blimp, lambada, perplexity)
│   │   ├── checkpoint_model.py  # Load models from checkpoints for eval
│   │   ├── config.py            # Eval config
│   │   ├── evaluator.py         # Core evaluator
│   │   ├── hf_model.py          # HuggingFace model wrapper for eval
│   │   ├── lm_harness_adapter.py # lm-evaluation-harness integration
│   │   ├── lm_harness_results.py # Harness results parsing
│   │   ├── protocol.py          # Eval model protocol
│   │   ├── results.py           # Results storage
│   │   └── runner.py            # Eval runner
│   ├── logging/                 # Training logger & metrics tracking
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── models/                  # Model architectures
│   │   ├── base.py              # Base model class
│   │   ├── mamba.py             # Mamba SSM model
│   │   └── tiny_transformer.py  # Transformer model
│   ├── server/                  # FastAPI server & WebSocket dashboard backend
│   │   ├── app.py               # FastAPI app with REST + WS endpoints
│   │   ├── broadcast.py         # WebSocket broadcast
│   │   ├── run_manager.py       # Training run lifecycle management
│   │   └── weights.py           # Weight inspection endpoints
│   ├── storage/                 # SQLite persistence
│   │   ├── checkpoint_db.py     # Checkpoint metadata DB
│   │   ├── database.py          # Main metrics DB
│   │   └── eval_db.py           # Eval results DB
│   ├── training/                # Training loop & utilities
│   │   ├── callbacks.py         # Training callbacks
│   │   ├── checkpointing.py     # Checkpoint save/load
│   │   ├── estimate.py          # Training time estimation
│   │   ├── export.py            # Model export (HF format)
│   │   ├── optimizer.py         # Optimizer & scheduler setup
│   │   ├── pipeline.py          # Full training pipeline
│   │   ├── run.py               # Training run orchestration
│   │   └── trainer.py           # Core training loop
│   ├── types/                   # Shared type definitions
│   │   ├── status.py
│   │   ├── training.py
│   │   └── ws.py
│   └── utils/                   # Utilities (device, env, reproducibility)
├── dashboard/                   # React + TypeScript frontend (Vite)
│   └── src/
│       ├── api/client.ts        # REST API client
│       ├── components/          # UI components (charts, pages, sidebar)
│       ├── containers/          # Container components (data fetching)
│       ├── hooks/               # Custom hooks (hash router, WebSocket)
│       ├── storage/             # Jotai atoms for state management
│       ├── types/               # TypeScript type definitions
│       └── ws/client.ts         # WebSocket client
├── scripts/                     # Utility scripts (eval, serve, rate infra)
├── data/                        # Training & eval data
│   ├── eval/                    # Eval datasets (blimp, lambada, wikitext2)
│   └── tiny_shakespeare.txt
├── checkpoints/                 # Saved model checkpoints
├── eval_results/                # Eval result JSONs
└── tests/                       # Pytest test suite
```

## Tech Stack

- **Python** with **uv** for package management
- **PyTorch** for model training
- **FastAPI** + **WebSockets** for the server
- **React** + **TypeScript** + **Vite** + **Jotai** for the dashboard
- **SQLite** for metrics/checkpoint/eval storage
- **pytest** for tests

## Common Commands

- `uv run python main.py` — Start training
- `uv run python scripts/serve.py` — Start the dashboard server
- `uv run pytest` — Run all tests
- `cd dashboard && npm run dev` — Start dashboard frontend dev server
