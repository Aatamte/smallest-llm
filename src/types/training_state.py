"""TrainingStateSnapshot — full training state for REST API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.types.data_point import DataPoint
from src.types.generation import GenerationRecord
from src.types.layer_stat import LayerStatRecord


@dataclass
class TrainingStateSnapshot:
    experiment_name: str
    status: str
    text_state: str
    max_steps: int
    start_time: str
    series: dict[str, list[DataPoint]] = field(default_factory=dict)
    layer_stats: list[LayerStatRecord] = field(default_factory=list)
    generations: list[GenerationRecord] = field(default_factory=list)

    def to_wire(self) -> dict:
        series_wire = {k: [p.to_wire() for p in v] for k, v in self.series.items()}

        def _last(key: str) -> float:
            pts = self.series.get(key, [])
            return pts[-1].value if pts else 0

        best_val = 0.0
        val_pts = self.series.get("valLoss", [])
        if val_pts:
            best_val = min(p.value for p in val_pts)

        return {
            "experimentName": self.experiment_name,
            "status": self.status,
            "textState": self.text_state,
            "maxSteps": self.max_steps,
            "startTime": self.start_time,
            "series": series_wire,
            "currentStep": self.series.get("trainLoss", [DataPoint(0, 0)])[-1].step,
            "currentTrainLoss": _last("trainLoss"),
            "currentValLoss": _last("valLoss"),
            "bestValLoss": best_val,
            "currentLR": _last("lr"),
            "currentGradNorm": _last("gradNorm"),
            "tokensSeen": int(_last("tokensSeen")),
            "tokensPerSec": _last("tokensPerSec"),
            "bpc": _last("bpc"),
            "layerStats": [l.to_wire() for l in self.layer_stats],
            "generations": [g.to_wire() for g in self.generations],
        }

    @staticmethod
    def build_series(
        steps_dict: dict[int, dict[str, float]],
    ) -> dict[str, list[DataPoint]]:
        key_map = {
            "train/loss": "trainLoss",
            "val/loss": "valLoss",
            "train/lr": "lr",
            "train/grad_norm": "gradNorm",
            "train/tokens_seen": "tokensSeen",
            "train/step_time": "stepTime",
        }

        series: dict[str, list[DataPoint]] = {}

        for step in sorted(steps_dict.keys()):
            m = steps_dict[step]

            for internal_key, wire_key in key_map.items():
                if internal_key in m:
                    series.setdefault(wire_key, []).append(
                        DataPoint(step=step, value=m[internal_key])
                    )

            step_time = m.get("train/step_time", 0)
            tokens_seen = m.get("train/tokens_seen", 0)
            if step_time > 0 and tokens_seen > 0:
                series.setdefault("tokensPerSec", []).append(
                    DataPoint(step=step, value=round(tokens_seen / step_time))
                )

            train_loss = m.get("train/loss")
            if train_loss:
                series.setdefault("bpc", []).append(
                    DataPoint(step=step, value=round(train_loss / math.log(2), 4))
                )

        return series
