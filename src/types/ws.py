"""WebSocket message types shared between server and logger."""

from typing import Literal, TypedDict, Union


# Message type tags
WSMessageType = Literal["step", "layers", "generation", "status", "log"]

LogLevel = Literal["info", "warn", "error"]


class StepMetrics(TypedDict, total=False):
    step: int
    trainLoss: float
    lr: float
    gradNorm: float
    tokensSeen: int
    tokensPerSec: float
    stepTime: float
    valLoss: float
    bpc: float


class LayerStat(TypedDict):
    name: str
    gradNorm: float
    gradMean: float
    weightNorm: float


class GenerationData(TypedDict):
    step: int
    prompt: str
    output: str


class LogData(TypedDict):
    level: LogLevel
    message: str
