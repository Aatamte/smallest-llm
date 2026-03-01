"""Protocol that models must satisfy to be evaluated."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from src.data.tokenizer import Tokenizer


@runtime_checkable
class Evaluatable(Protocol):
    """Interface a model must expose for evaluation.

    All methods work at the token-ID level. Evaluation tasks handle
    text-to-token conversion via the model's tokenizer.
    """

    @property
    def device(self) -> torch.device: ...

    @property
    def tokenizer(self) -> Tokenizer: ...

    def loglikelihood(self, context_ids: list[int], target_ids: list[int]) -> float:
        """Return log P(target_ids | context_ids).

        The model should compute the sum of log probabilities for each token
        in target_ids, conditioned on context_ids concatenated with the
        preceding target tokens.
        """
        ...

    def loglikelihood_rolling(self, token_ids: list[int]) -> float:
        """Return sum of log P(token_i | token_<i) over the full sequence.

        Used for perplexity computation and minimal-pair scoring.
        The first token is conditioned on nothing (or BOS).
        """
        ...

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        stop_ids: list[int] | None = None,
    ) -> list[int]:
        """Autoregressively generate token IDs given a prompt.

        Args:
            prompt_ids: Context token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. 0.0 = greedy.
            stop_ids: Stop generation if any of these tokens are produced.

        Returns:
            Generated token IDs (not including the prompt).
        """
        ...
