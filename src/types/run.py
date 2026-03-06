"""RunRecord — mirrors a row in runs."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class RunRecord:
    id: int
    name: str
    status: str
    config: dict
    env: dict
    created_at: str
    finished_at: str | None = None

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "config": self.config,
            "env": self.env,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> RunRecord:
        config = row.get("config", "{}")
        env = row.get("env", "{}")
        return cls(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            config=json.loads(config) if isinstance(config, str) else (config or {}),
            env=json.loads(env) if isinstance(env, str) else (env or {}),
            created_at=row.get("created_at", ""),
            finished_at=row.get("finished_at"),
        )
