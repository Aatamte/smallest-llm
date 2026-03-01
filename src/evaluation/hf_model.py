"""HuggingFace model wrapper satisfying the Evaluatable protocol."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFTokenizerAdapter:
    """Adapts a HuggingFace tokenizer to our Tokenizer protocol."""

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.vocab_size = hf_tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class HFModel:
    """Wraps any HuggingFace causal LM for evaluation.

    Usage:
        model = HFModel("HuggingFaceTB/SmolLM-135M")
        results = evaluate(model, EvalConfig(tasks=["perplexity"]))
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        self._model_name = model_name

        # Resolve device
        if device == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # MPS doesn't support all fp16 ops reliably — use float32 there
        if self._device.type == "mps":
            dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

        hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer = HFTokenizerAdapter(hf_tok)

        param_count = sum(p.numel() for p in self._model.parameters())
        print(f"Loaded {model_name} ({param_count / 1e6:.1f}M params) on {self._device}")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self) -> HFTokenizerAdapter:
        return self._tokenizer

    @torch.no_grad()
    def loglikelihood(self, context_ids: list[int], target_ids: list[int]) -> float:
        input_ids = context_ids + target_ids
        ids_tensor = torch.tensor([input_ids], device=self._device)

        logits = self._model(ids_tensor).logits  # (1, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log P(target_i | context + target_<i)
        # Target tokens start at position len(context_ids)
        total_ll = 0.0
        start = len(context_ids)
        for i, target_id in enumerate(target_ids):
            # Logit at position (start + i - 1) predicts token at position (start + i)
            pos = start + i - 1
            total_ll += log_probs[0, pos, target_id].item()

        return total_ll

    @torch.no_grad()
    def loglikelihood_rolling(self, token_ids: list[int]) -> float:
        ids_tensor = torch.tensor([token_ids], device=self._device)

        logits = self._model(ids_tensor).logits  # (1, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log P(token_i | token_<i) for i = 1..N-1
        # logits[0, i-1, :] predicts token_ids[i]
        total_ll = 0.0
        for i in range(1, len(token_ids)):
            total_ll += log_probs[0, i - 1, token_ids[i]].item()

        return total_ll

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        stop_ids: list[int] | None = None,
    ) -> list[int]:
        ids_tensor = torch.tensor([prompt_ids], device=self._device)

        if temperature == 0.0:
            # Greedy
            output = self._model.generate(
                ids_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._model.config.eos_token_id,
            )
        else:
            output = self._model.generate(
                ids_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self._model.config.eos_token_id,
            )

        # Return only the newly generated tokens
        generated = output[0, len(prompt_ids):].tolist()

        # Apply stop_ids
        if stop_ids:
            for i, tok_id in enumerate(generated):
                if tok_id in stop_ids:
                    generated = generated[:i]
                    break

        return generated
