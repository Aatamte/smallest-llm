import type { StepMetrics, LayerStat, Generation } from "../types/metrics";

const LAYER_NAMES = [
  "embed",
  "block.0.attn",
  "block.0.ffn",
  "block.1.attn",
  "block.1.ffn",
  "block.2.attn",
  "block.2.ffn",
  "ln_f",
  "head",
];

const SAMPLE_PROMPTS = [
  "ROMEO:",
  "To be or not",
  "The king shall",
];

const SAMPLE_OUTPUTS = [
  " What light through yonder window breaks? It is the east, and Juliet is the sun.",
  " to be, that is the question. Whether 'tis nobler in the mind to suffer",
  " ride forth at dawn, with banners held aloft and trumpets sounding clear",
  " know that all men are mortal, and yet we persist in our follies",
  " command his armies westward, through the valley of shadows and mist",
  " What dreams may come when we have shuffled off this mortal coil",
];

let step = 0;
let baseLoss = 4.2;
let tokensSeen = 0;

export function generateStepMetrics(): StepMetrics {
  step += 1;
  tokensSeen += 64 * 256; // batch_size * seq_len

  // Simulate loss curve with noise
  const progress = step / 10000;
  baseLoss = 4.2 * Math.exp(-3 * progress) + 0.8 + (Math.random() - 0.5) * 0.05;
  const valLoss = step % 50 === 0 ? baseLoss + 0.1 + Math.random() * 0.05 : undefined;

  // LR: warmup then cosine decay
  const warmupSteps = 100;
  let lr: number;
  if (step < warmupSteps) {
    lr = 3e-4 * (step / warmupSteps);
  } else {
    const decay = 0.5 * (1 + Math.cos(Math.PI * (step - warmupSteps) / (10000 - warmupSteps)));
    lr = 3e-4 * (0.1 + 0.9 * decay);
  }

  const gradNorm = 0.3 + Math.random() * 0.4 + (step < 200 ? 1.0 : 0);
  const tokensPerSec = 10000 + Math.random() * 4000;
  const bpc = baseLoss / Math.log(2);

  return {
    step,
    trainLoss: baseLoss,
    valLoss,
    lr,
    gradNorm,
    tokensSeen,
    tokensPerSec,
    stepTime: (64 * 256) / tokensPerSec,
    updateParamRatio: 0.001 + Math.random() * 0.0005,
    bpc,
  };
}

export function generateLayerStats(): LayerStat[] {
  return LAYER_NAMES.map((name) => ({
    name,
    gradNorm: 0.01 + Math.random() * 0.5,
    weightNorm: 1.0 + Math.random() * 2.0,
    updateRatio: 0.0005 + Math.random() * 0.002,
  }));
}

export function generateGeneration(): Generation {
  const prompt = SAMPLE_PROMPTS[Math.floor(Math.random() * SAMPLE_PROMPTS.length)];
  const output = SAMPLE_OUTPUTS[Math.floor(Math.random() * SAMPLE_OUTPUTS.length)];
  return { step, prompt, output: prompt + output };
}

export function resetMockState() {
  step = 0;
  baseLoss = 4.2;
  tokensSeen = 0;
}
