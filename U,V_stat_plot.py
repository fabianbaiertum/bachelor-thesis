import numpy as np
import matplotlib.pyplot as plt

def weight_matrices(n: int):
    # U (ordered): i != j
    W_U_ord = np.ones((n, n)) - np.eye(n)
    W_U_ord /= (n * (n - 1))

    # U (unordered): i < j (upper triangle)
    W_U_unord = np.triu(np.ones((n, n)), k=1)
    W_U_unord /= (n * (n - 1) / 2.0)

    # V: all pairs including diagonal
    W_V = np.ones((n, n)) / (n * n)

    return W_U_ord, W_U_unord, W_V

def plot_mask(M, title):
    plt.imshow(M, origin='lower', interpolation='nearest')
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(fraction=0.046, pad=0.04)
    # tick labels 1..n for clarity
    n = M.shape[0]
    plt.xticks(range(n), range(1, n+1))
    plt.yticks(range(n), range(1, n+1))
    plt.grid(False)

def annotated_weights(W, title, decimals=3):
    plt.imshow(W, origin='lower', interpolation='nearest')
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(fraction=0.046, pad=0.04)
    n = W.shape[0]
    plt.xticks(range(n), range(1, n+1))
    plt.yticks(range(n), range(1, n+1))
    # annotate a few cells (not all) to keep readable
    for i in range(n):
        for j in range(n):
            # show only some annotations to avoid clutter
            if (i == j) or (i in (0, n-1) and j in (0, n-1)) or (i in (0, n-1) and j == (n//2)) or (j in (0, n-1) and i == (n//2)):
                plt.text(j, i, f"{W[i,j]:.{decimals}f}", ha='center', va='center', fontsize=8)

def visualize_u_v(n=10):
    W_U_ord, W_U_unord, W_V = weight_matrices(n)

    # 1) Inclusion masks (True/False) — what pairs are used?
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plot_mask(W_U_ord > 0, r"U-stat (ordered): include $i\neq j$ (no diagonal)")
    plt.subplot(1, 3, 2)
    plot_mask(W_U_unord > 0, r"U-stat (unordered): include $i<j$ (each pair once)")
    plt.subplot(1, 3, 3)
    plot_mask(W_V > 0, r"V-stat: include all $(i,j)$ (diagonal included)")
    plt.tight_layout()
    plt.show()

    # 2) Exact weight values
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    annotated_weights(W_U_ord, r"Weights: U (ordered) $=1/[n(n-1)]$ off-diagonal")
    plt.subplot(1, 3, 2)
    annotated_weights(W_U_unord, r"Weights: U (unordered) $=1/\binom{n}{2}$ on upper triangle")
    plt.subplot(1, 3, 3)
    annotated_weights(W_V, r"Weights: V $=1/n^2$ everywhere (incl. diagonal)")
    plt.tight_layout()
    plt.show()

    # 3) Difference heatmap: V - U (ordered)
    diff = W_V - W_U_ord
    plt.figure(figsize=(5.5, 5))
    plt.imshow(diff, origin='lower', interpolation='nearest')
    plt.title(r"Difference: $W_V - W_U$ (ordered)")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(fraction=0.046, pad=0.04)
    n = diff.shape[0]
    plt.xticks(range(n), range(1, n+1))
    plt.yticks(range(n), range(1, n+1))
    # annotate a few representative cells
    for i, j in [(0,0), (0,1), (1,0), (n-1, n-1), (n//2, n//2)]:
        plt.text(j, i, f"{diff[i,j]:.3f}", ha='center', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()

    # Sanity checks
    print("Sum of U ordered weights:", W_U_ord.sum())       # -> 1.0
    print("Sum of U unordered weights:", W_U_unord.sum())   # -> 1.0
    print("Sum of V weights:", W_V.sum())                   # -> 1.0
    print("Diagonal weight (V):", W_V[0,0], "  Diagonal weight (U ordered):", 0.0)
    print("Off-diagonal weight (U ordered):", W_U_ord[0,1], "  Off-diagonal weight (V):", W_V[0,1])

# Run
visualize_u_v(n=10)
