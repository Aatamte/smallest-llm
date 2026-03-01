"""Checkpoint model wrapper satisfying the Evaluatable protocol."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from src.config.base import ExperimentConfig
from src.data.datasets import TextFileDataset, load_dataset
from src.data.tokenizer import Tokenizer, build_tokenizer
from src.training.run import _build_model
from src.utils.device import resolve_device


class CheckpointModel:
    """Wraps a trained checkpoint for evaluation.

    Loads the model + tokenizer from a checkpoint file and exposes the
    Evaluatable protocol (device, tokenizer, loglikelihood, loglikelihood_rolling,
    generate).

    Usage:
        model = CheckpointModel("/path/to/checkpoint-500.pt")
        results = evaluate(model, EvalConfig(tasks=["perplexity"]))
    """

    def __init__(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        self._device = resolve_device("auto")
        state = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        config = ExperimentConfig.from_dict(state["config"])

        # Rebuild tokenizer
        tok_name = config.data.tokenizer_name
        if tok_name == "char":
            dataset = load_dataset(config.data.dataset_name)
            if isinstance(dataset, TextFileDataset):
                self._tokenizer = build_tokenizer(tok_name, text=dataset.text)
            else:
                self._tokenizer = build_tokenizer(tok_name, text="")
        else:
            self._tokenizer = build_tokenizer(tok_name)

        self._model = _build_model(config, self._tokenizer.vocab_size).to(self._device)
        self._model.load_state_dict(state["model_state_dict"])
        self._model.eval()
        self._max_seq_len = getattr(self._model, "max_seq_len", None)

        param_count = sum(p.numel() for p in self._model.parameters())
        print(f"Loaded checkpoint ({param_count / 1e6:.1f}M params) on {self._device}")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @torch.inference_mode()
    def loglikelihood(self, context_ids: list[int], target_ids: list[int]) -> float:
        input_ids = context_ids + target_ids
        # Crop to max_seq_len if needed
        if self._max_seq_len and len(input_ids) > self._max_seq_len:
            input_ids = input_ids[-self._max_seq_len:]
            context_len = max(0, len(input_ids) - len(target_ids))
        else:
            context_len = len(context_ids)

        ids_tensor = torch.tensor([input_ids], device=self._device)
        logits = self._model(ids_tensor).logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Vectorized scoring
        n_targets = min(len(target_ids), log_probs.size(1) - context_len + 1)
        if n_targets > 0 and context_len - 1 >= 0:
            positions = torch.arange(context_len - 1, context_len - 1 + n_targets, device=self._device)
            positions = positions.clamp(0, log_probs.size(1) - 1)
            targets = torch.tensor(target_ids[:n_targets], device=self._device)
            total_ll = log_probs[0, positions, targets].sum().item()
        else:
            total_ll = 0.0

        return total_ll

    @torch.inference_mode()
    def loglikelihood_rolling(self, token_ids: list[int], on_progress=None) -> float:
        total_to_score = len(token_ids) - 1

        # For sequences longer than max_seq_len, use sliding window
        if self._max_seq_len and len(token_ids) > self._max_seq_len:
            return self._loglikelihood_rolling_chunked(token_ids, on_progress=on_progress)

        ids_tensor = torch.tensor([token_ids], device=self._device)
        logits = self._model(ids_tensor).logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Vectorized scoring
        positions = torch.arange(0, len(token_ids) - 1, device=self._device)
        targets = torch.tensor(token_ids[1:], device=self._device)
        total_ll = log_probs[0, positions, targets].sum().item()

        if on_progress:
            on_progress(total_to_score, total_to_score)
        return total_ll

    def _loglikelihood_rolling_chunked(self, token_ids: list[int], on_progress=None) -> float:
        """Sliding-window log-likelihood for sequences exceeding max_seq_len."""
        stride = self._max_seq_len // 2
        total_ll = 0.0
        prev_end = 0
        tokens_scored = 0
        total_to_score = len(token_ids) - 1

        for begin in range(0, len(token_ids), stride):
            end = min(begin + self._max_seq_len, len(token_ids))
            chunk = token_ids[begin:end]

            ids_tensor = torch.tensor([chunk], device=self._device)
            logits = self._model(ids_tensor).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Only score tokens we haven't scored yet (vectorized)
            score_start = max(1, prev_end - begin)
            if score_start < len(chunk):
                positions = torch.arange(score_start - 1, len(chunk) - 1, device=self._device)
                targets = torch.tensor(chunk[score_start:], device=self._device)
                total_ll += log_probs[0, positions, targets].sum().item()
                tokens_scored += len(chunk) - score_start

            if on_progress:
                on_progress(tokens_scored, total_to_score)

            prev_end = end
            if end >= len(token_ids):
                break

        return total_ll

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        stop_ids: list[int] | None = None,
    ) -> list[int]:
        ids = torch.tensor([prompt_ids], device=self._device)

        for _ in range(max_new_tokens):
            ctx = ids if self._max_seq_len is None else ids[:, -self._max_seq_len:]
            logits = self._model(ctx).logits[:, -1, :]

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if stop_ids and token_id in stop_ids:
                break

            ids = torch.cat([ids, next_token], dim=1)

        return ids[0, len(prompt_ids):].tolist()
