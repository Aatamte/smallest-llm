"""Models table — saved/exported model records."""

from __future__ import annotations

import json
import os

from src.storage.table import Column, Table


class ModelsTable(Table):
    name = "models"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER"),
        Column("name", "TEXT", not_null=True),
        Column("path", "TEXT", not_null=True),
        Column("config", "TEXT"),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = []

    def create_model(
        self,
        run_id: int | None,
        name: str,
        path: str,
        config: dict | None = None,
    ) -> int:
        return self.insert(
            run_id=run_id,
            name=name,
            path=path,
            config=json.dumps(config) if config else None,
        )

    def list_all(self) -> list[dict]:
        return self.select(
            columns="id, run_id, name, path, created_at",
            order_by="id DESC",
        )

    def get(self, model_id: int) -> dict | None:
        rows = self.select(where="id = ?", params=[model_id])
        return rows[0] if rows else None

    def get_by_name(self, name: str) -> dict | None:
        rows = self.select(where="name = ?", params=[name])
        return rows[0] if rows else None

    def delete_model(self, model_id: int):
        self.delete(where="id = ?", params=[model_id])

    def sync_dir(self, models_dir: str = "models"):
        """Sync the models table with subdirectories in models_dir."""
        os.makedirs(models_dir, exist_ok=True)

        disk_models: dict[str, str] = {}
        for entry in os.listdir(models_dir):
            full_path = os.path.join(models_dir, entry)
            if os.path.isdir(full_path):
                disk_models[entry] = full_path

        db_rows = self.select(columns="id, name, path")
        db_by_name = {r["name"]: r for r in db_rows}

        for name, row in db_by_name.items():
            if name not in disk_models:
                self.delete(where="id = ?", params=[row["id"]])

        for name, path in disk_models.items():
            if name not in db_by_name:
                self.insert(name=name, path=path)
