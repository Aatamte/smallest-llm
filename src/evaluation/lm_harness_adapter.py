"""Adapter bridging CheckpointModel to lm-evaluation-harness's LM interface."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from src.evaluation.checkpoint_model import CheckpointModel

# Type for progress callbacks: (current, total) -> None
ProgressCallback = Optional[Callable[[int, int], None]]


@register_model("smallest-llm")
class CheckpointLMAdapter(LM):
    """Wraps a CheckpointModel for use with lm-evaluation-harness.

    Usage:
        model = CheckpointModel("/path/to/checkpoint.pt")
        adapter = CheckpointLMAdapter(checkpoint_model=model)
        results = lm_eval.simple_evaluate(model=adapter, tasks=["hellaswag"])
    """

    def __init__(
        self,
        checkpoint_model: CheckpointModel | None = None,
        checkpoint_path: str | None = None,
        batch_size: int = 1,
        on_progress: ProgressCallback = None,
        **kwargs,
    ):
        super().__init__()
        if checkpoint_model is not None:
            self._model = checkpoint_model
        elif checkpoint_path is not None:
            self._model = CheckpointModel(checkpoint_path)
        else:
            raise ValueError("Provide either checkpoint_model or checkpoint_path")

        self._batch_size = batch_size
        self._on_progress = on_progress

    @property
    def eot_token_id(self) -> int:
        tok = self._model.tokenizer
        if hasattr(tok, "_tok") and hasattr(tok._tok, "eos_token_id"):
            return tok._tok.eos_token_id or 0
        return 0

    @property
    def max_length(self) -> int:
        return self._model._max_seq_len or 512

    @property
    def max_gen_toks(self) -> int:
        return 64

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._model.device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self._model.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self._model.tokenizer.decode(tokens)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        total = len(requests)
        for i, req in enumerate(requests):
            context, continuation = req.args
            ctx_ids = self.tok_encode(context) if context else [self.eot_token_id]
            cont_ids = self.tok_encode(continuation)

            ll = self._model.loglikelihood(ctx_ids, cont_ids)

            # Check greedy: generate same length and compare
            generated = self._model.generate(
                ctx_ids, max_new_tokens=len(cont_ids), temperature=0.0
            )
            is_greedy = generated[: len(cont_ids)] == cont_ids

            results.append((ll, is_greedy))

            if self._on_progress:
                self._on_progress(i + 1, total)
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        total = len(requests)
        for i, req in enumerate(requests):
            (text,) = req.args
            token_ids = self.tok_encode(text)
            if not token_ids:
                results.append(0.0)
            else:
                ll = self._model.loglikelihood_rolling(token_ids)
                results.append(ll)

            if self._on_progress:
                self._on_progress(i + 1, total)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        total = len(requests)
        for i, req in enumerate(requests):
            context, gen_kwargs = req.args
            ctx_ids = self.tok_encode(context)

            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)

            # Map single-token stop strings to stop IDs
            stop_ids = []
            for stop_str in until:
                encoded = self.tok_encode(stop_str)
                if len(encoded) == 1:
                    stop_ids.append(encoded[0])

            generated_ids = self._model.generate(
                ctx_ids,
                max_new_tokens=max_gen_toks,
                temperature=temperature,
                stop_ids=stop_ids or None,
            )

            generated_text = self.tok_decode(generated_ids)

            # Truncate at first stop string
            for stop_str in until:
                idx = generated_text.find(stop_str)
                if idx != -1:
                    generated_text = generated_text[:idx]

            results.append(generated_text)

            if self._on_progress:
                self._on_progress(i + 1, total)
        return results


class ProgressLMWrapper(LM):
    """Wraps any LM to add per-request progress reporting.

    Splits request lists into sub-batches and reports progress between
    each batch. Works with any LM implementation (HFLM, custom, etc.).
    """

    def __init__(self, inner: LM, on_progress: ProgressCallback = None, chunk_size: int = 16):
        super().__init__()
        self._inner = inner
        self._on_progress = on_progress
        self._chunk_size = chunk_size

    # Delegate required properties to inner model
    @property
    def eot_token_id(self):
        return self._inner.eot_token_id

    @property
    def max_length(self):
        return self._inner.max_length

    @property
    def max_gen_toks(self):
        return self._inner.max_gen_toks

    @property
    def batch_size(self):
        return self._inner.batch_size

    @property
    def device(self):
        return self._inner.device

    def tok_encode(self, string, **kwargs):
        return self._inner.tok_encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self._inner.tok_decode(tokens, **kwargs)

    def _chunked_call(self, method_name: str, requests: list[Instance]) -> list:
        """Call a method on the inner model in chunks, reporting progress."""
        method = getattr(self._inner, method_name)
        total = len(requests)
        results = []

        for start in range(0, total, self._chunk_size):
            chunk = requests[start : start + self._chunk_size]
            chunk_results = method(chunk)
            results.extend(chunk_results)

            if self._on_progress:
                self._on_progress(min(start + len(chunk), total), total)

        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return self._chunked_call("loglikelihood", requests)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        return self._chunked_call("loglikelihood_rolling", requests)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        return self._chunked_call("generate_until", requests)
