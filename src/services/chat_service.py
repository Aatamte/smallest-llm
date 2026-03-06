"""Chat model loading, generation, and state management."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from src.services.model_loader import load_eval_model

if TYPE_CHECKING:
    from src.services.run_service import RunManager


class ChatService:
    """Manages a single loaded chat model for interactive generation."""

    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager
        self._lock = threading.Lock()
        self._state: dict = {"model": None, "tokenizer": None, "source": None, "name": None}

    def get_status(self) -> dict:
        with self._lock:
            return {
                "loaded": self._state["model"] is not None,
                "source": self._state["source"],
                "name": self._state["name"],
            }

    def load(self, source: str, *, model_name: str | None = None, run_id: int | None = None, step: int | None = None) -> dict:
        """Load a model for chat. Source is 'hf' or 'checkpoint'."""
        with self._lock:
            self._state.update(model=None, tokenizer=None, source=None, name=None)

        if source == "hf":
            model_row = self.run_manager.db.get_model_by_name(model_name)
            if model_row is None:
                raise ValueError(f"Unknown model: {model_name}")

            chat_model = load_eval_model(model_row["path"])

            with self._lock:
                self._state.update(
                    model=chat_model, tokenizer=chat_model.tokenizer,
                    source="hf", name=model_name,
                )
            return {"status": "loaded", "name": model_name}

        elif source == "checkpoint":
            if run_id is None or step is None:
                raise ValueError("run_id and step required for checkpoint source")

            model, tokenizer, device = self._load_checkpoint_model(run_id, step)
            display_name = f"run-{run_id} step {step}"

            with self._lock:
                self._state.update(
                    model=model, tokenizer=tokenizer,
                    source="checkpoint", name=display_name,
                )
            return {"status": "loaded", "name": display_name}

        else:
            raise ValueError(f"Unknown source: {source}")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.8, top_k: int = 50) -> dict:
        """Generate text from the loaded model."""
        import torch

        with self._lock:
            model = self._state["model"]
            tokenizer = self._state["tokenizer"]
            source = self._state["source"]

        if model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        token_ids = tokenizer.encode(prompt)

        if source == "hf":
            output_ids = model.generate(
                prompt_ids=token_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            text = tokenizer.decode(output_ids)
        else:
            device = next(model.parameters()).device
            input_tensor = torch.tensor([token_ids], device=device)
            output_tensor = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            new_ids = output_tensor[0, len(token_ids):].tolist()
            text = tokenizer.decode(new_ids)

        return {"text": text, "prompt": prompt}

    def unload(self) -> dict:
        with self._lock:
            self._state.update(model=None, tokenizer=None, source=None, name=None)
        return {"status": "unloaded"}

    def _load_checkpoint_model(self, run_id: int, step: int):
        """Load a trained model + tokenizer from a checkpoint."""
        import os
        import torch
        from src.config.base import ExperimentConfig
        from src.data.tokenizer import build_tokenizer
        from src.data.datasets import load_dataset, TextFileDataset
        from src.training.run import _build_model
        from src.utils.device import resolve_device

        checkpoints = self.run_manager.db.get_checkpoints(run_id)
        match = next((c for c in checkpoints if c["step"] == step), None)
        if match is None:
            raise ValueError(f"No checkpoint at step {step} for run {run_id}")

        path = match["path"]
        if not os.path.exists(path):
            raise ValueError(f"Checkpoint file not found: {path}")

        device = resolve_device("auto")
        state = torch.load(path, map_location=device, weights_only=False)
        config = ExperimentConfig.from_dict(state["config"])

        tok_name = config.data.tokenizer_name
        if tok_name == "char":
            dataset = load_dataset(config.data.dataset_name)
            if isinstance(dataset, TextFileDataset):
                tokenizer = build_tokenizer(tok_name, text=dataset.text)
            else:
                tokenizer = build_tokenizer(tok_name, text="")
        else:
            tokenizer = build_tokenizer(tok_name)

        model = _build_model(config, tokenizer.vocab_size).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        return model, tokenizer, device
