"""Entry point for standalone training (no server)."""

from __future__ import annotations

import argparse

from src.config.base import ExperimentConfig, apply_cli_overrides
from src.config.presets import get_preset, get_presets
from src.training.run import start_run


def main():
    parser = argparse.ArgumentParser(description="smallest-llm training")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--preset", type=str, default=None, help="Named preset config")
    args, unknown = parser.parse_known_args()

    if args.preset:
        config = get_preset(args.preset)
        if config is None:
            available = [p["name"] for p in get_presets()]
            raise ValueError(f"Unknown preset: {args.preset!r}. Available: {available}")
    elif args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    config = apply_cli_overrides(config, unknown)

    run_id = start_run(config=config)
    print(f"Training complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()
