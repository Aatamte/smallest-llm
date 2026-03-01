"""Entry point for standalone training (no server)."""

from __future__ import annotations

import argparse

from src.config.base import ExperimentConfig, apply_cli_overrides
from src.training.run import start_run


def main():
    parser = argparse.ArgumentParser(description="smallest-llm training")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    args, unknown = parser.parse_known_args()

    config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    config = apply_cli_overrides(config, unknown)

    run_id = start_run(config=config)
    print(f"Training complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()
