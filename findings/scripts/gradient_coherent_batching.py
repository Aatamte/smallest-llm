"""Test whether gradient-coherent batching breaks the 1/sqrt(n) efficiency bound.

Hypothesis: if we group sequences by similarity (so their token gradients
point in more aligned directions), gradient efficiency should exceed the
1/sqrt(n) prediction for random vectors.

Approach:
  1. Collect N sequences from TinyStories
  2. Compute a "gradient fingerprint" for each sequence (gradient direction
     from a single forward-backward pass)
  3. Cluster sequences by cosine similarity of their gradient fingerprints
  4. Measure gradient efficiency for:
     a) Random batches (baseline, should match 1/sqrt(n))
     b) Coherent batches (sequences from same cluster)
     c) Adversarial batches (sequences from maximally different clusters)
  5. Compare against the 1/sqrt(n) bound

Usage: uv run python findings/scripts/gradient_coherent_batching.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.models import build_model

SEED = 42
V = 256  # byte vocab
L = 64   # seq len
N_SEQS = 64  # sequences to fingerprint
B = 8    # batch size for efficiency measurement
SAMPLE_TOKENS = 16  # tokens to sample for per-token gradient norms


def load_real_sequences(n: int, seq_len: int) -> torch.Tensor:
    """Load n real sequences from TinyStories as byte-encoded tensors."""
    from src.data.byte_tokenizer import ByteTokenizer
    from datasets import load_dataset
    from src.data.datasets import HF_CACHE_DIR

    tok = ByteTokenizer()
    ds = load_dataset("roneneldan/TinyStories", split="train",
                      cache_dir=str(HF_CACHE_DIR))

    sequences = []
    token_buf: list[int] = []
    for example in ds:
        text = example["text"]
        tokens = tok.encode(text)
        token_buf.extend(tokens)
        while len(token_buf) >= seq_len + 1 and len(sequences) < n:
            chunk = token_buf[:seq_len + 1]
            token_buf = token_buf[seq_len + 1:]
            sequences.append(torch.tensor(chunk, dtype=torch.long))
        if len(sequences) >= n:
            break

    return torch.stack(sequences)  # (n, seq_len+1)


def compute_gradient_fingerprints(
    model: torch.nn.Module, sequences: torch.Tensor,
) -> torch.Tensor:
    """Compute gradient direction for each sequence individually.

    Returns (n_seqs, n_params) tensor of normalized gradient vectors.
    """
    fingerprints = []
    for i in range(len(sequences)):
        model.zero_grad()
        input_ids = sequences[i, :-1].unsqueeze(0)  # (1, L)
        labels = sequences[i, 1:].unsqueeze(0)       # (1, L)

        out = model(input_ids, labels=labels)
        out.loss.backward()

        grad = torch.cat([
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ])
        # Normalize to unit vector (direction only)
        grad = grad / (grad.norm() + 1e-10)
        fingerprints.append(grad)

    return torch.stack(fingerprints)


def cluster_by_gradient_similarity(
    fingerprints: torch.Tensor, n_clusters: int = 4,
) -> list[list[int]]:
    """Greedily cluster sequences by gradient cosine similarity."""
    n = len(fingerprints)
    # Cosine similarity matrix
    sim = fingerprints @ fingerprints.T  # (n, n), already normalized

    assigned = [False] * n
    clusters: list[list[int]] = []

    for _ in range(n_clusters):
        # Find the unassigned sequence most dissimilar to existing cluster centers
        if not clusters:
            # First cluster: pick the sequence with highest average dissimilarity
            unassigned = [i for i in range(n) if not assigned[i]]
            avg_sim = sim[unassigned][:, unassigned].mean(dim=1)
            seed_local = avg_sim.argmin().item()
            seed_idx = unassigned[seed_local]
        else:
            # Pick unassigned sequence most dissimilar to all existing seeds
            seeds = [c[0] for c in clusters]
            unassigned = [i for i in range(n) if not assigned[i]]
            if not unassigned:
                break
            max_sim_to_seeds = sim[unassigned][:, seeds].max(dim=1).values
            seed_local = max_sim_to_seeds.argmin().item()
            seed_idx = unassigned[seed_local]

        # Greedily add most similar unassigned sequences
        cluster = [seed_idx]
        assigned[seed_idx] = True

        target_size = n // n_clusters
        while len(cluster) < target_size:
            unassigned = [i for i in range(n) if not assigned[i]]
            if not unassigned:
                break
            # Similarity to cluster center
            sims = sim[seed_idx, unassigned]
            best_local = sims.argmax().item()
            best_idx = unassigned[best_local]
            cluster.append(best_idx)
            assigned[best_idx] = True

        clusters.append(cluster)

    # Assign any remaining
    remaining = [i for i in range(n) if not assigned[i]]
    for i, idx in enumerate(remaining):
        clusters[i % len(clusters)].append(idx)

    return clusters


def measure_batch_efficiency(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    batch_indices: list[int],
) -> dict:
    """Measure gradient efficiency for a specific batch of sequences."""
    batch_seqs = sequences[batch_indices].contiguous()  # (B, L+1)
    input_ids = batch_seqs[:, :-1].contiguous()  # (B, L)
    labels = batch_seqs[:, 1:].contiguous()       # (B, L)

    out = model(input_ids, labels=labels)
    per_token_loss = F.cross_entropy(
        out.logits.reshape(-1, V), labels.reshape(-1), reduction="none"
    )
    n_tokens = len(batch_indices) * L

    # Full averaged gradient
    model.zero_grad()
    per_token_loss.mean().backward(retain_graph=True)
    full_grad = torch.cat([
        p.grad.flatten() for p in model.parameters() if p.grad is not None
    ])
    full_norm = full_grad.norm().item()

    # Per-token gradient norms (sampled)
    sample = min(SAMPLE_TOKENS, n_tokens)
    indices = torch.randperm(n_tokens)[:sample]
    norms = []
    for idx in indices:
        model.zero_grad()
        per_token_loss[idx].backward(retain_graph=True)
        g = torch.cat([
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ])
        norms.append(g.norm().item())

    avg_norm = sum(norms) / len(norms)
    efficiency = full_norm / (avg_norm + 1e-10)

    return {
        "full_norm": full_norm,
        "avg_per_token_norm": avg_norm,
        "efficiency": efficiency,
        "theory": 1.0 / math.sqrt(n_tokens),
        "waste_pct": (1 - efficiency) * 100,
    }


def measure_sequence_gradient_cosines(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    batch_indices: list[int],
) -> float:
    """Measure average pairwise gradient cosine similarity between sequences in a batch."""
    grads = []
    for idx in batch_indices:
        model.zero_grad()
        input_ids = sequences[idx, :-1].unsqueeze(0)
        labels = sequences[idx, 1:].unsqueeze(0)
        out = model(input_ids, labels=labels)
        out.loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g / (g.norm() + 1e-10))

    grads_t = torch.stack(grads)
    sim = grads_t @ grads_t.T
    # Average off-diagonal
    n = len(batch_indices)
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


def run():
    torch.manual_seed(SEED)

    print("Loading real sequences from TinyStories...")
    sequences = load_real_sequences(N_SEQS, L)
    print(f"Loaded {len(sequences)} sequences of length {L+1}")

    # Use transformer for cleaner gradients
    model = build_model("transformer", vocab_size=V, max_seq_len=L,
                        extra_args={"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.0})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: transformer, {n_params:,} params")

    # ── 1. Compute gradient fingerprints ──
    print("\nComputing gradient fingerprints for each sequence...")
    fingerprints = compute_gradient_fingerprints(model, sequences)

    # Show pairwise similarity distribution
    sim_matrix = fingerprints @ fingerprints.T
    mask = ~torch.eye(N_SEQS, dtype=torch.bool)
    off_diag = sim_matrix[mask]
    print(f"Pairwise gradient cosine similarity:")
    print(f"  Mean: {off_diag.mean():.4f}")
    print(f"  Std:  {off_diag.std():.4f}")
    print(f"  Min:  {off_diag.min():.4f}")
    print(f"  Max:  {off_diag.max():.4f}")

    # ── 2. Cluster sequences ──
    n_clusters = N_SEQS // B
    print(f"\nClustering {N_SEQS} sequences into {n_clusters} groups of ~{B}...")
    clusters = cluster_by_gradient_similarity(fingerprints, n_clusters=n_clusters)

    # Show within-cluster vs between-cluster similarity
    within_sims = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                within_sims.append(sim_matrix[cluster[i], cluster[j]].item())
    between_sims = []
    for ci in range(len(clusters)):
        for cj in range(ci + 1, len(clusters)):
            for i in clusters[ci]:
                for j in clusters[cj]:
                    between_sims.append(sim_matrix[i, j].item())

    print(f"Within-cluster cosine:  {sum(within_sims)/len(within_sims):.4f}")
    print(f"Between-cluster cosine: {sum(between_sims)/len(between_sims):.4f}")

    # ── 3. Measure gradient efficiency ──
    print(f"\n{'='*60}")
    print("GRADIENT EFFICIENCY: COHERENT vs RANDOM BATCHING")
    print(f"{'='*60}")
    print(f"Batch size: {B}, Seq len: {L}, Tokens per batch: {B*L}")
    print(f"1/sqrt(n) bound: {1.0/math.sqrt(B*L):.4f}")

    # Random batches
    rng = torch.Generator().manual_seed(SEED)
    random_results = []
    for trial in range(n_clusters):
        indices = torch.randperm(N_SEQS, generator=rng)[:B].tolist()
        r = measure_batch_efficiency(model, sequences, indices)
        random_results.append(r)

    # Coherent batches (from clusters)
    coherent_results = []
    for cluster in clusters:
        indices = cluster[:B]
        if len(indices) < B:
            continue
        r = measure_batch_efficiency(model, sequences, indices)
        coherent_results.append(r)

    # Adversarial batches (one from each cluster)
    adversarial_results = []
    for trial in range(min(4, n_clusters)):
        indices = []
        for ci in range(min(B, len(clusters))):
            cluster = clusters[(ci + trial) % len(clusters)]
            indices.append(cluster[trial % len(cluster)])
        if len(indices) < B:
            continue
        r = measure_batch_efficiency(model, sequences, indices[:B])
        adversarial_results.append(r)

    def summarize(results: list[dict], label: str):
        effs = [r["efficiency"] for r in results]
        wastes = [r["waste_pct"] for r in results]
        avg_eff = sum(effs) / len(effs)
        avg_waste = sum(wastes) / len(wastes)
        print(f"\n{label}:")
        print(f"  Avg efficiency:  {avg_eff:.4f}")
        print(f"  Avg waste:       {avg_waste:.1f}%")
        print(f"  1/sqrt(n) bound: {1.0/math.sqrt(B*L):.4f}")
        print(f"  Ratio vs bound:  {avg_eff / (1.0/math.sqrt(B*L)):.2f}x")
        return avg_eff

    theory = 1.0 / math.sqrt(B * L)
    eff_random = summarize(random_results, "RANDOM batches (baseline)")
    eff_coherent = summarize(coherent_results, "COHERENT batches (clustered by gradient)")
    if adversarial_results:
        eff_adversarial = summarize(adversarial_results, "ADVERSARIAL batches (max diversity)")

    # ── 4. Also measure within-batch gradient cosine ──
    print(f"\n{'='*60}")
    print("WITHIN-BATCH GRADIENT COSINE (sequence-level)")
    print(f"{'='*60}")

    # Random
    rng2 = torch.Generator().manual_seed(SEED)
    random_cosines = []
    for _ in range(4):
        indices = torch.randperm(N_SEQS, generator=rng2)[:B].tolist()
        cos = measure_sequence_gradient_cosines(model, sequences, indices)
        random_cosines.append(cos)

    # Coherent
    coherent_cosines = []
    for cluster in clusters[:4]:
        if len(cluster) >= B:
            cos = measure_sequence_gradient_cosines(model, sequences, cluster[:B])
            coherent_cosines.append(cos)

    print(f"Random batch avg cosine:   {sum(random_cosines)/len(random_cosines):.4f}")
    if coherent_cosines:
        print(f"Coherent batch avg cosine: {sum(coherent_cosines)/len(coherent_cosines):.4f}")

    # ── 5. Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Theory (random vectors): efficiency = 1/sqrt({B*L}) = {theory:.4f}")
    print(f"Random batching:         efficiency = {eff_random:.4f} ({eff_random/theory:.2f}x theory)")
    print(f"Coherent batching:       efficiency = {eff_coherent:.4f} ({eff_coherent/theory:.2f}x theory)")
    if adversarial_results:
        print(f"Adversarial batching:    efficiency = {eff_adversarial:.4f} ({eff_adversarial/theory:.2f}x theory)")

    improvement = (eff_coherent - eff_random) / eff_random * 100
    print(f"\nCoherent vs Random: {improvement:+.1f}% efficiency change")

    if eff_coherent > theory * 1.5:
        print("RESULT: Coherent batching BREAKS the 1/sqrt(n) bound!")
    elif eff_coherent > eff_random * 1.1:
        print("RESULT: Coherent batching improves efficiency but doesn't break the bound.")
    else:
        print("RESULT: Coherent batching does NOT meaningfully improve efficiency.")


if __name__ == "__main__":
    run()
