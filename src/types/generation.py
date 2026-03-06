"""GenerationRecord — a sample text generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationRecord:
    step: int
    prompt: str
    output: str

    def to_wire(self) -> dict:
        return {"step": self.step, "prompt": self.prompt, "output": self.output}
