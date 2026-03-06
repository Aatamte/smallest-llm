"""Lightweight Evaluatable wrapper for a live trainer model + tokenizer.

Unlike CheckpointModel (which loads from disk), this wraps the in-memory
model directly so we can run eval tasks mid-training without saving a checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.tokenizer import Tokenizer


class TrainerModelWrapper:
    """Wraps a live model + tokenizer for the Evaluatable protocol.

    Automatically puts the model in eval mode for inference and
    restores train mode afterwards.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_seq_len = getattr(model, "max_seq_len", None)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @torch.inference_mode()
    def loglikelihood_choices(self, context_ids: list[int], choices: list[list[int]]) -> list[float]:
        """Score multiple choices for the same context in a single batched forward pass."""
        was_training = self._model.training
        self._model.eval()
        try:
            # Build sequences: context + each choice
            sequences = []
            context_lens = []
            for choice_ids in choices:
                seq = context_ids + choice_ids
                if self._max_seq_len and len(seq) > self._max_seq_len:
                    seq = seq[-self._max_seq_len:]
                    ctx_len = max(0, len(seq) - len(choice_ids))
                else:
                    ctx_len = len(context_ids)
                sequences.append(seq)
                context_lens.append(ctx_len)

            # Pad to same length
            max_len = max(len(s) for s in sequences)
            padded = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self._device)
            for j, seq in enumerate(sequences):
                padded[j, :len(seq)] = torch.tensor(seq, device=self._device)

            # Single batched forward pass
            logits = self._model(padded).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # Extract log-likelihood for each choice's target tokens
            results = []
            for j, (choice_ids, ctx_len) in enumerate(zip(choices, context_lens)):
                n_targets = min(len(choice_ids), log_probs.size(1) - ctx_len + 1)
                if n_targets > 0 and ctx_len - 1 >= 0:
                    positions = torch.arange(ctx_len - 1, ctx_len - 1 + n_targets, device=self._device)
                    positions = positions.clamp(0, log_probs.size(1) - 1)
                    targets = torch.tensor(choice_ids[:n_targets], device=self._device)
                    ll = log_probs[j, positions, targets].sum().item()
                else:
                    ll = 0.0
                results.append(ll)
            return results
        finally:
            if was_training:
                self._model.train()

    @torch.inference_mode()
    def loglikelihood(self, context_ids: list[int], target_ids: list[int]) -> float:
        was_training = self._model.training
        self._model.eval()
        try:
            input_ids = context_ids + target_ids
            if self._max_seq_len and len(input_ids) > self._max_seq_len:
                input_ids = input_ids[-self._max_seq_len:]
                context_len = max(0, len(input_ids) - len(target_ids))
            else:
                context_len = len(context_ids)

            ids_tensor = torch.tensor([input_ids], device=self._device)
            logits = self._model(ids_tensor).logits
            log_probs = F.log_softmax(logits, dim=-1)

            n_targets = min(len(target_ids), log_probs.size(1) - context_len + 1)
            if n_targets > 0 and context_len - 1 >= 0:
                positions = torch.arange(context_len - 1, context_len - 1 + n_targets, device=self._device)
                positions = positions.clamp(0, log_probs.size(1) - 1)
                targets = torch.tensor(target_ids[:n_targets], device=self._device)
                total_ll = log_probs[0, positions, targets].sum().item()
            else:
                total_ll = 0.0

            return total_ll
        finally:
            if was_training:
                self._model.train()

    @torch.inference_mode()
    def loglikelihood_rolling(self, token_ids: list[int], on_progress=None) -> float:
        was_training = self._model.training
        self._model.eval()
        try:
            total_to_score = len(token_ids) - 1

            if self._max_seq_len and len(token_ids) > self._max_seq_len:
                return self._loglikelihood_rolling_chunked(token_ids, on_progress=on_progress)

            ids_tensor = torch.tensor([token_ids], device=self._device)
            logits = self._model(ids_tensor).logits
            log_probs = F.log_softmax(logits, dim=-1)

            positions = torch.arange(0, len(token_ids) - 1, device=self._device)
            targets = torch.tensor(token_ids[1:], device=self._device)
            total_ll = log_probs[0, positions, targets].sum().item()

            if on_progress:
                on_progress(total_to_score, total_to_score)
            return total_ll
        finally:
            if was_training:
                self._model.train()

    def _loglikelihood_rolling_chunked(self, token_ids: list[int], on_progress=None) -> float:
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
        was_training = self._model.training
        self._model.eval()
        try:
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
        finally:
            if was_training:
                self._model.train()
