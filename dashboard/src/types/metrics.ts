export interface StepMetrics {
  step: number;
  trainLoss: number;
  valLoss?: number;
  lr: number;
  gradNorm: number;
  tokensSeen: number;
  tokensPerSec: number;
  stepTime: number;
  updateParamRatio?: number;
  bpc?: number;
}

export interface LayerStat {
  name: string;
  gradNorm: number;
  weightNorm: number;
  updateRatio: number;
}

export interface ActivationStat {
  name: string;
  mean: number;
  std: number;
  max: number;
  min: number;
  pctZero: number;
}

export interface Generation {
  step: number;
  prompt: string;
  output: string;
}

export type TrainingStatus = "training" | "paused" | "complete" | "idle";

export type LogLevel = "info" | "warn" | "error";

export interface LogEntry {
  timestamp: number;
  level: LogLevel;
  source: string;
  message: string;
}

export interface TrainingState {
  experimentName: string;
  status: TrainingStatus;
  maxSteps: number;
  startTime: number;

  // Time series
  steps: StepMetrics[];

  // Latest snapshot
  currentStep: number;
  currentTrainLoss: number;
  currentValLoss: number;
  bestValLoss: number;
  currentLR: number;
  currentGradNorm: number;
  tokensSeen: number;
  tokensPerSec: number;
  bpc: number;

  // Per-layer
  layerStats: LayerStat[];

  // Generations
  generations: Generation[];
}
