# Goal: Train a Real Language Model on a MacBook Pro

## The Challenge

Train a language model on consumer hardware (Apple Silicon MacBook Pro) that demonstrates genuine language understanding — not just low loss numbers, but measurable capabilities. This may require frontier-level innovations in architecture, training efficiency, and evaluation.

## Constraints

- **Hardware**: Single MacBook Pro with Apple Silicon (MPS backend)
- **Time**: Training runs should complete in minutes to hours, not days
- **Parameters**: ~300K to ~10M (what fits and trains fast on MPS)
- **Data**: Public datasets (TinyStories, MiniPile, OpenWebText, Shakespeare)

## What Success Looks Like

### Tier 1: Statistical Coherence (baseline)
- Low perplexity on held-out text
- Token distribution matches natural language
- *This is what standard training already achieves.*

### Tier 2: Local Structure
- Completes common phrases correctly ("once upon a ___" → "time")
- Respects grammar within clauses
- Handles punctuation and capitalization
- No degenerate repetition loops

### Tier 3: Medium-Range Coherence
- Maintains a topic for a full sentence
- Subject-verb agreement across clauses
- Pronoun reference resolution within a paragraph
- Generates text that a human can read without cringing

### Tier 4: State Tracking (SSM advantage)
- Counting: "aaa" → knows there are 3 a's
- Recall: remembers a name introduced earlier in context
- Pattern completion: "ABCABC___" → "ABC"
- Parity/nesting tracking
- *This is where Mamba/SSM should beat transformers at the same parameter count.*

### Tier 5: Actual Utility
- Generates a coherent 3-5 sentence story
- Follows a simple instruction (after SFT)
- Few-shot pattern matching
- *Ambitious for this scale, but worth measuring.*

## Current Training Innovations

### Proven (from ablation experiments)
- **Muon optimizer**: 0.5 lower val loss than AdamW, clear winner
- **Muon + Echo**: marginal gain over Muon alone

### Under Investigation
- **State Anchoring**: forces SSM to retain information (addresses measured retention problem)
- **Gradient Sharpening**: keeps top-K% of gradient components, zeros noise
- **Neuroplasticity**: progressive model growing (needs longer runs to evaluate)

### Shelved (didn't help in ablations)
- Echo Loss alone: neutral effect on val loss
- Phantom Batches: neutral (correlated pseudo-gradients ≠ real data diversity)
- Hydra Training: tiny quality gain but 2.3x wall-clock cost
- Plastic Weights (Hebbian fast weights / EMA context modulation): neutral — the SSM state already acts as a learned EMA, making external context modulation redundant
- Multi-Scale SSM Init (forced fast/slow head timescale separation): hurt — constrained init prevents the model from learning its own timescales; worse val loss at all step counts
- Hybrid SSM-Attention (replace 1 SSM layer with local attention): early T3 coherence boost (+13%) fades by 500 steps; losing an SSM layer costs capacity that outweighs the attention benefit at this scale

## Evaluation Strategy

### What We Measure

1. **Perplexity** — held-out test set, standard metric (already implemented)

2. **Capability Probes** — synthetic tests with known answers:
   - Phrase completion (fill-in-the-blank on common phrases)
   - Grammar acceptance (BLiMP-style, already partially implemented)
   - State tracking tasks (counting, recall, pattern completion)
   - Repetition detection (does the model get stuck in loops?)

3. **Generation Quality** — score actual model outputs:
   - Coherence: does generated text make sense for N tokens?
   - Diversity: unique n-gram ratio (avoid repetitive generation)
   - Structural: does it produce sentence boundaries, paragraphs?

4. **Scaling Curves** — how capability changes with:
   - Training steps (learning efficiency)
   - Model size (parameter efficiency)
   - Data seen (sample efficiency)

### What We Compare Against

- Same architecture, different training techniques (ablation)
- Same parameters, different architectures (Mamba-3 vs Transformer)
- Published results for models at similar scale (if available)

## Architecture

- **Primary model**: Mamba-3 (SSM with trapezoidal discretization, SSD, RoPE)
- **Optimized variant**: Improved Mamba-3 (fused SSD, gradient checkpointing)
- **Baseline**: Transformer (for comparison)
- **Training**: Muon optimizer, cosine LR schedule, curriculum learning

## Open Questions

- Can state anchoring actually improve downstream capability, or just the retention probe metric?
- What's the minimum model size to achieve Tier 3 coherence?
- Is Mamba-3's state tracking advantage real at small scale, or does it need more parameters?
- Can we get Tier 4 capabilities at all with <1M parameters?
- What training data mix matters most at this scale?
