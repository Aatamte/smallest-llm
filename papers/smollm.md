# SmolLM: Small Language Models (HuggingFace, July 2024)

Source: https://huggingface.co/blog/smollm

## Model Sizes
- 135M, 360M, 1.7B parameters

## Architecture

135M and 360M inspired by MobileLLM: prioritize **depth over width**, use GQA, embedding tying, 2048 context.

| Component | 135M | 360M | 1.7B |
|-----------|------|------|------|
| Hidden Size | 576 | 1024 | 2048 |
| Num Layers | 30 | 30 | 24 |
| Num Heads | 9 | 16 | 32 |
| Head Dim | 64 | 64 | 64 |
| MLP Ratio | 2.67 | 2.67 | 2.67 |

Tokenizer: vocab size 49,152, trained on SmolLM-Corpus.

## Training Data: SmolLM-Corpus (252B tokens)

1. **Cosmopedia v2** (28B tokens) — 39M synthetic docs from Mixtral-8x7B-Instruct. 40% middle school, 30% college, 30% mixed/stories/code. Uses 34K topics from BISAC book classification.
2. **FineWeb-Edu deduplicated** (220B tokens) — educational web pages filtered by quality classifier trained on Llama3-70B-Instruct annotations.
3. **Stack-Edu-Python** (4B tokens) — educational Python code scored >=4 by Python-edu-scorer. Converges 3x faster than unfiltered.

## Training

- LR scheduler: trapezoidal with 20% cooldown
- 135M/360M: 600B tokens
- 1.7B: 1T tokens
- Performance keeps improving beyond Chinchilla optimal point
- 600B tokens sufficient for 135M/360M (diminishing returns after 400B)

## Key Findings

- **Depth > width** for small models (MobileLLM finding confirmed)
- **Data quality is everything** — curated educational data + synthetic textbooks
- Middle school textbooks give best overall performance
- Stories help common sense benchmarks
- No improvement from instruct datasets during cooldown or upsampling curated subsets
- GQA works well at small scale

## Evaluation

- 135M outperforms MobileLLM-125M (trained on 1T tokens vs 600B)
- 360M outperforms all <500M models including Qwen2-500M
- 1.7B best in class <2B, beats Phi-1.5, MobileLLM-1.5B, Qwen2-1.5B
- 1.7B gets 24 pass@1 on HumanEval

## Memory Footprint

| Model | FP32 | int8 |
|-------|------|------|
| 135M | ~540MB | ~140MB |
| 360M | ~1.4GB | ~360MB |
| 1.7B | ~6.8GB | ~1.7GB |

## Instruction Tuning

- SFT: 1 epoch on WebInstructSub + StarCoder2-Self-OSS-Instruct, lr=3e-4
- DPO: 1 epoch on HelpSteer (135M/1.7B) or argilla/dpo-mix-7k (360M)
