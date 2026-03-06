# SmolLM3: smol, multilingual, long-context reasoner

Published July 8, 2025

## Model Summary

SmolLM3 is a 3B model trained on 11T tokens that achieves state-of-the-art performance at the 3B scale and remains competitive with 4B models. Key features include:

- **3B model** trained on 11T tokens, SoTA at the 3B scale and competitive with 4B models
- **Instruct model** with **dual mode reasoning,** supporting `think`/`no_think` modes
- **Multilingual support** for 6 languages: English, French, Spanish, German, Italian, and Portuguese
- **Long context** up to 128k with NoPE and using YaRN

**Available Models:**
- Base model: https://hf.co/HuggingFaceTB/SmolLM3-3B-Base
- Instruct and reasoning model: https://hf.co/HuggingFaceTB/SmolLM3-3B

---

## Pretraining

### Architecture and Training Details

SmolLM3 follows a transformer decoder architecture with tied embeddings, building on Llama architecture with key optimizations:

**Key Architectural Modifications:**

1. **Grouped Query Attention (GQA):** Replaced multi-head attention with 4 groups, matching multi-head attention performance while reducing KV cache size during inference.

2. **NoPE:** Implemented selective removal of rotary position embeddings from every 4th layer, improving long context performance without affecting short context capabilities.

3. **Intra-Document Masking:** Applied attention masking to ensure tokens from different documents in the same training sequence don't attend to each other, following Llama 3's approach.

4. **Training Stability:** Removed weight decay from embedding layers (following OLMo 2) to improve training stability.

**Training Configuration:**
- Global batch size: 2.36M tokens
- Sequence length: 4096
- Learning rate: 2e-4
- Optimizer: AdamW (beta1: 0.9, beta2: 0.95)
- Weight decay: 0.1
- Gradient clipping: 1
- Scheduler: WSD (Warmup-Stable-Decay) with 2000 warmup steps and linear decay to 0 in final 10%
- Training framework: nanotron
- Hardware: 384 H100 GPUs for 24 days

### Data Mixture and Training Stages

SmolLM3 employs a three-stage training strategy on 11.2T tokens:

**Stage 1: Stable phase (0T → 8T tokens)**
- Web: 85% (12% multilingual) - FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ
- Code: 12% - The Stack v2, StarCoder2, Jupyter, Kaggle, GitHub issues, StackExchange
- Math: 3% - FineMath3+, InfiWebMath3+

**Stage 2: Stable phase (8T → 10T tokens)**
- Web: 75% (12% Multilingual)
- Code: 15% - Adding Stack-Edu
- Math: 10% - FineMath4+, InfiWebMath4+, MegaMath

**Stage 3: Decay Phase (10T → 11.1T tokens)**
- Web: 63% (12% Multilingual)
- Code: 24% - high-quality code data
- Math: 13% - including instruction and reasoning datasets (OpenMathReasoning)

---

## Mid-training

### Long Context Extension

After main pretraining, SmolLM3 was trained on an additional 100B tokens to extend context length:

1. **Stage 1 (50B tokens):** 4k → 32k context with RoPE theta increased to 1.5M
2. **Stage 2 (50B tokens):** 32k → 64k context with RoPE theta increased to 5M

Both stages upsampled math, code, and reasoning data. Using NoPE and training on the decay mixture with longer sequences and increased RoPE theta values proved sufficient for competitive long context performance.

**Long Context Inference:** Using YARN to extrapolate beyond training length, the model can handle up to 128k context (2x extension beyond 64k training length).

### Reasoning Mid-training

Mid-training dataset contained 35B tokens sourced from:
- Open Thought's OpenThoughts3-1.2M
- NVIDIA's Llama-Nemotron-Post-Training-Dataset-v1.1 with reasoning traces

Training details:
- Chat template: ChatML
- Packing: Wrapped packing to avoid excessive structure
- Epochs: 4 (~140B tokens)

---

## Post-training

### Building the Chat Template

SmolLM3's chat template enables seamless switching between reasoning and non-reasoning modes:

- Users activate reasoning or non-reasoning modes via `/think` and `/no_think` flags in the system prompt
- In non-reasoning mode, the model's response is pre-filled with empty think blocks for direct answers
- Supports tool calling with two distinct sections: XML Tools and Python Tools
- Includes default system message with metadata (date, knowledge cutoff, reasoning mode)
- Use `/system_override` flag to exclude metadata section

### Supervised Finetuning

Following reasoning mid-training, SFT incorporates capabilities across both modes for:
- Math
- Code
- General reasoning
- Instruction following
- Multilinguality
- Tool calling

**Dataset Synthesis:** Generated synthetic data by prompting Qwen3-32B in reasoning mode with prompts from existing non-reasoning datasets to address scarcity in certain domains.

**Final SFT Dataset:** 1.8B tokens
- Non-reasoning mode: 1B tokens (12 datasets)
- Reasoning mode: 0.8B tokens (10 datasets with reasoning traces)
- Training: 4 epochs (~8B tokens)
- Packing: BFD (best-fit decreasing) with loss masked on user turns

### Off-policy Model Alignment with Anchored Preference Optimization (APO)

APO training used:
- **Non-reasoning mode:** Tulu3 preference dataset
- **Reasoning mode:** Synthetic preference pairs generated from Qwen3-32B (chosen) and Qwen3-0.6B (rejected)

**APO vs DPO:** APO provides a more stable optimization objective by using an anchored reference model, showing improved downstream performance across mathematics, science, instruction following, coding, chat, and multilingual tasks.

**Performance Note:** APO training caused degradation on long context benchmarks (RULER) due to focus on reasoning and limited training data (24k tokens). Resolved through model merging.

### Model Merging

Model merging strategy using MergeKit:
1. Create a model "soup" from APO checkpoints
2. Combine with mid-training checkpoint (strong long-context performance)
3. Linear merge with weights: 0.9 (APO soup) + 0.1 (mid-training checkpoint)

Result: Recovered base model's RULER score on contexts up to 128k tokens while maintaining performance across all tasks.

---

## Evaluation

### Base Model

SmolLM3 consistently outperforms other 3B models and achieves competitive performance with larger 4B models (Qwen3-4B, Gemma3-4B).

**Benchmark Performance:**
- First or second place on knowledge/reasoning benchmarks (HellaSwag, ARC, BoolQ)
- Competitive math and coding performance within 3B class
- Strong long-context performance on Ruler 64k
- Strong multilingual performance across 5 major European languages

**Evaluated Benchmarks:**
HellaSwag, ARC, Winogrande, CommonsenseQA, MMLU-CF, MMLU Pro CF, PIQA, OpenBookQA, GSM8K, MATH, HumanEval+, MBPP+

### Dual Instruct / Reasoning Model

**Non-reasoning Mode:**
- Outperforms other 3B non-reasoning models (Llama3.2 3B Instruct, Qwen2.5 3B Instruct)
- Sits at efficiency sweet spot between reasoning models
- Significantly outperforms Qwen3 1.7B while approaching 4B model performance

**Reasoning Mode (Extended Thinking):**
- Substantial improvements over non-reasoning mode
- Notable gains: AIME 2025 (36.7% vs 9.3%), LiveCodeBench (30.0% vs 15.2%), GPQA Diamond (41.7% vs 35.7%)
- Qwen3 4B generally achieves highest scores
- Competitive performance within 3B parameter class, excelling in mathematical reasoning

---

## How to Run Locally

### Installation

```bash
pip install -U transformers
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda" # for GPU usage or "cpu" for CPU usage

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)

# prepare the model input
prompt = "Give me a brief explanation of gravity in simple terms."
messages_think = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

**Recommended sampling parameters:** `temperature=0.6` and `top_p=0.95`

### Enabling and Disabling Extended Thinking Mode

Extended thinking is enabled by default. To disable:

```python
prompt = "Give me a brief explanation of gravity in simple terms."
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

To enable (default): use `/think` or omit the flag.

### Agentic Usage

SmolLM3 supports tool calling with `xml_tools` (standard tool-calling) or `python_tools` (Python-like function calls):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM3-3B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tools = [
    {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city to get the weather for"}}}}
]

messages = [
    {
        "role": "user",
        "content": "Hello! How is the weather today in Copenhagen?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    enable_thinking=False, # True works as well, your choice!
    xml_tools=tools,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

---

## Resources

- **Models collection with quantized checkpoints:** https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23
- **SmolLM GitHub repo:** https://github.com/huggingface/smollm
- **HuggingFace organization:** https://huggingface.co/HuggingFaceTB
- **Training configs and datasets:** https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs
- **SFT datasets:** https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- **Training logs:** https://wandb.ai/huggingface/SmolLM3-training-logs

---

## Citation

```bibtex
@misc{bakouch2025smollm3,
  title={{SmolLM3: smol, multilingual, long-context reasoner}},
  author={Bakouch, Elie and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and Tunstall, Lewis and Patiño, Carlos Miguel and Beeching, Edward and Roucher, Aymeric and Reedi, Aksel Joonas and Gallouédec, Quentin and Rasul, Kashif and Habib, Nathan and Fourrier, Clémentine and Kydlicek, Hynek and Penedo, Guilherme and Larcher, Hugo and Morlon, Mathieu and Srivastav, Vaibhav and Lochner, Joshua and Nguyen, Xuan-Son and Raffel, Colin and von Werra, Leandro and Wolf, Thomas},
  year={2025},
  howpublished={\url{https://huggingface.co/blog/smollm3}}
}
```
