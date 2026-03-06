"""PipelineRunner: multi-stage training orchestrator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.config.base import DataConfig, SchedulerConfig, StageConfig
from src.data.chat import ChatTemplate, build_chat_dataloaders
from src.data.datasets import HFChatDataset, HFStreamingDataset, TextFileDataset, load_dataset
from src.data.streaming import build_streaming_dataloaders
from src.data.text import build_dataloaders
from src.data.tokenizer import HFTokenizer
from src.training.optimizer import build_scheduler, set_optimizer_lr
from src.training.pipeline.callbacks import SanityCheckCallback, StageMetadataCallback
from src.training.pipeline.resize import _resize_embeddings

if TYPE_CHECKING:
    from src.config.base import ExperimentConfig
    from src.logging.logger import Logger
    from src.training.trainer import Trainer


class FixedBatchLoader:
    """Yields from a fixed list of batches, for sanity-check overfitting."""

    def __init__(self, batches: list):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PipelineRunner:
    """Orchestrates multi-stage training through the existing Trainer.

    Between stages:
    - Model weights and optimizer state (momentum) carry forward
    - Dataloaders are rebuilt (new seq_len, batch_size)
    - Scheduler is rebuilt (new LR curve, warmup, step budget)
    - Step counter is globally continuous
    """

    def __init__(
        self,
        config: ExperimentConfig,
        trainer: Trainer,
        dataset,
        tokenizer,
    ):
        self.config = config
        self.trainer = trainer
        self.dataset = dataset
        self.tokenizer = tokenizer

    def run(self):
        """Execute all stages sequentially."""
        stages = self.config.stages
        if not stages:
            self.trainer.train()
            return

        global_step = 0
        self._sft_prepared = False

        for i, stage in enumerate(stages):
            if self.trainer.should_stop:
                break

            # Prepare for SFT on first SFT stage
            if stage.stage_type == "sft" and not self._sft_prepared:
                self._prepare_for_sft()

            self._log_stage_transition(i, stage, len(stages))

            # Rebuild dataloaders for this stage
            seq_len = stage.seq_len or self.config.data.max_seq_len
            batch_size = stage.batch_size or self.config.data.batch_size
            train_loader, val_loader = self._build_stage_dataloaders(
                stage, seq_len, batch_size, stage.overfit_batches
            )

            # Rebuild scheduler for this stage
            if stage.lr is not None:
                set_optimizer_lr(self.trainer.optimizer, stage.lr)

            scheduler_config = SchedulerConfig(
                warmup_steps=stage.warmup_steps,
                min_lr_ratio=stage.min_lr_ratio
                if stage.min_lr_ratio is not None
                else self.config.scheduler.min_lr_ratio,
            )
            sched_steps = stage.max_steps
            if sched_steps == 0 and stage.max_flops is not None:
                # Estimate steps from FLOPs budget for scheduler curve
                fpt = self.trainer._compute_flops_per_token()
                grad_acc = self.config.training.gradient_accumulation_steps
                tps = seq_len * batch_size * grad_acc
                sched_steps = int(stage.max_flops / max(fpt * tps, 1)) + 1
            scheduler = build_scheduler(
                scheduler_config, self.trainer.optimizer, sched_steps
            )

            # Patch trainer for this stage
            self.trainer.train_loader = train_loader
            self.trainer.val_loader = val_loader
            self.trainer._cached_val_batches = None  # invalidate eval cache
            self.trainer.scheduler = scheduler
            self.trainer._train_iter = None  # force rebuild of infinite iterator
            self.trainer.start_step = global_step
            self.trainer.should_stop = False

            # Patch config budget
            self.trainer.config.training.max_steps = global_step + sched_steps
            if stage.max_flops is not None:
                self.trainer.config.training.max_flops = stage.max_flops
                self.trainer._max_flops = stage.max_flops
                # Reset flops counter for this stage's budget
                self.trainer.flops_total = 0
            else:
                self.trainer.config.training.max_flops = None
                self.trainer._max_flops = None
            if not stage.eval_enabled:
                self.trainer.config.training.eval_interval = sched_steps + 1
            elif stage.eval_interval is not None:
                self.trainer.config.training.eval_interval = stage.eval_interval
            if stage.log_interval is not None:
                self.trainer.config.training.log_interval = stage.log_interval
            if stage.save_interval is not None:
                self.trainer.config.training.save_interval = stage.save_interval

            # Attach stage-specific callbacks
            dataset_label = stage.dataset_name or self.config.data.dataset_name
            stage_cbs = [StageMetadataCallback(i, stage.name, len(stages), dataset_label)]
            if stage.overfit_batches > 0 and stage.loss_threshold is not None:
                stage_cbs.append(SanityCheckCallback(stage.loss_threshold))

            for cb in stage_cbs:
                self.trainer.callbacks.append(cb)

            # Run this stage
            self.trainer.train()

            # Remove stage-specific callbacks
            for cb in stage_cbs:
                self.trainer.callbacks.remove(cb)

            global_step += sched_steps

    def _prepare_for_sft(self):
        """Add chat special tokens and resize model embeddings for SFT."""
        if not isinstance(self.tokenizer, HFTokenizer):
            raise ValueError("SFT requires an HFTokenizer (e.g. 'gpt2'), not a char tokenizer")

        num_added = self.tokenizer.add_chat_tokens()
        if num_added > 0:
            new_vocab_size = self.tokenizer.vocab_size
            _resize_embeddings(self.trainer.model, new_vocab_size)
            print(f"SFT prep: added {num_added} chat tokens, vocab_size={new_vocab_size}")

        self._chat_template = ChatTemplate(self.tokenizer)
        self._sft_prepared = True

    def _build_stage_dataloaders(self, stage, seq_len, batch_size, overfit_batches):
        """Rebuild dataloaders with stage-specific seq_len and optional dataset override."""
        dataset = self.dataset
        if stage.dataset_name:
            dataset = load_dataset(stage.dataset_name)

        data_config = replace(
            self.config.data,
            max_seq_len=seq_len,
            batch_size=batch_size,
        )

        if isinstance(dataset, HFChatDataset):
            train_loader, val_loader, _ = build_chat_dataloaders(
                data_config, dataset, self.tokenizer, self._chat_template
            )
        elif isinstance(dataset, TextFileDataset):
            train_loader, val_loader, _ = build_dataloaders(
                data_config, dataset.text, self.tokenizer
            )
        elif isinstance(dataset, HFStreamingDataset):
            train_loader, val_loader, _ = build_streaming_dataloaders(
                data_config, dataset, self.tokenizer
            )
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        if overfit_batches > 0:
            batches = []
            for j, batch in enumerate(train_loader):
                if j >= overfit_batches:
                    break
                batches.append(batch)
            train_loader = FixedBatchLoader(batches)

        return train_loader, val_loader

    def _log_stage_transition(self, index, stage, total):
        """Log stage transition to console and broadcast."""
        dataset_label = stage.dataset_name or self.config.data.dataset_name
        stage_type = stage.stage_type
        if self.trainer.logger:
            self.trainer.logger.broadcast_stage(
                index, stage.name, total,
                dataset=dataset_label, stage_type=stage_type,
            )
        print(f"=== Stage {index + 1}/{total}: {stage.name} (type: {stage_type}, dataset: {dataset_label}) ===")
