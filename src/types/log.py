"""LogRecord — a log line."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogRecord:
    level: str  # "info" | "warn" | "error"
    message: str

    def to_wire(self) -> dict:
        return {"level": self.level, "message": self.message}
