"""Shared server state — imported by route modules."""

from __future__ import annotations

import os

from src.server.broadcast import broadcaster
from src.services.run_service import RunManager
from src.services.eval_service import EvalService
from src.services.chat_service import ChatService
from src.storage import EvalDatabase

DB_PATH = "smallest_llm.db"
EVAL_DB_PATH = "eval.db"
MODELS_DIR = "models"

run_manager = RunManager(DB_PATH, broadcaster)
eval_db = EvalDatabase(EVAL_DB_PATH)
eval_service = EvalService(eval_db, run_manager)
chat_service = ChatService(run_manager)

# HF models to download into models/ on startup if not present
PRESET_MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}


def download_preset_models():
    """Download preset HF models into models/ if not already present."""
    for name, hf_id in PRESET_MODELS.items():
        model_dir = os.path.join(MODELS_DIR, name)
        if os.path.isdir(model_dir):
            continue
        print(f"Downloading {hf_id} -> {model_dir}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  Done: {name}")
        except Exception as e:
            print(f"  Warning: failed to download {hf_id}: {e}")
