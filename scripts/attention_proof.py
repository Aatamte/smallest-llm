"""Attention in ~20 lines of math. No PyTorch, no classes, no frameworks.

This is the entire mechanism that powers GPT, Claude, etc.
Everything else (layers, norms, MLPs) is just wrapping this.
"""

import numpy as np

def attention(X, Wq, Wk, Wv):
    """
    X:  (T, d) — sequence of T token vectors
    Wq: (d, d) — query projection ("what am I looking for?")
    Wk: (d, d) — key projection ("what do I contain?")
    Wv: (d, d) — value projection ("what do I provide?")
    """
    Q = X @ Wq                          # (T, d) — queries
    K = X @ Wk                          # (T, d) — keys
    V = X @ Wv                          # (T, d) — values

    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)       # (T, T) — similarity matrix

    # Causal mask: token i can only attend to tokens <= i
    mask = np.triu(np.ones_like(scores), k=1) * -1e9
    scores = scores + mask

    # Softmax: convert scores to probabilities
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exp / exp.sum(axis=-1, keepdims=True)  # (T, T)

    return weights @ V                  # (T, d) — output


# ── Demo ──────────────────────────────────────────────────────────────

np.random.seed(42)

T, d = 6, 8  # 6 tokens, 8 dimensions
X = np.random.randn(T, d)
Wq = np.random.randn(d, d) * 0.02
Wk = np.random.randn(d, d) * 0.02
Wv = np.random.randn(d, d) * 0.02

Y = attention(X, Wq, Wk, Wv)

print("Input shape: ", X.shape)
print("Output shape:", Y.shape)
print()

# The attention weights show WHO each token looks at
scores = (X @ Wq) @ (X @ Wk).T / np.sqrt(d)
mask = np.triu(np.ones_like(scores), k=1) * -1e9
scores = scores + mask
exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights = exp / exp.sum(axis=-1, keepdims=True)

print("Attention weights (each row = where that token looks):")
print("       ", "  ".join(f"t{j}" for j in range(T)))
for i in range(T):
    row = "  ".join(f"{w:.2f}" for w in weights[i])
    print(f"  t{i} -> {row}")

print()
print("Key observations:")
print(f"  - Token 0 can only see itself: weight = {weights[0,0]:.2f}")
print(f"  - Token 5 distributes across all 6 tokens")
print(f"  - Each row sums to 1.0 (it's a probability distribution)")
print(f"  - The upper triangle is 0 (causal: can't see the future)")
