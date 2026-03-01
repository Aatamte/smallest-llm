"""Training state types returned by REST API."""

from typing import List, TypedDict

from src.types.ws import LayerStat, StepMetrics, GenerationData


class TrainingState(TypedDict, total=False):
    experimentName: str
    status: str
    maxSteps: int
    startTime: str
    steps: List[StepMetrics]
    currentStep: int
    currentTrainLoss: float
    currentValLoss: float
    bestValLoss: float
    currentLR: float
    currentGradNorm: float
    tokensSeen: int
    tokensPerSec: float
    bpc: float
    layerStats: List[LayerStat]
    generations: List[GenerationData]
