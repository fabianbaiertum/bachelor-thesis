from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Paths --------------------
# A: Tesla, B: Intel, C: Booking.com
path_stock_a = "C:/Users/Fabia/OneDrive/Desktop/TSLA_2015-01-01_2015-03-31_10/output-2015/0/0/3/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv"
path_stock_b = "C:/Users/Fabia/OneDrive/Desktop/INTC_2015-01-01_2015-03-31_10/output-2015/0/0/1/INTC_2015-01-02_34200000_57600000_orderbook_10.csv"
path_stock_c = "C:/Users/Fabia/OneDrive/Desktop/PCLN_2015-01-01_2015-03-31_10/output-2015/0/0/4/PCLN_2015-01-02_34200000_57600000_orderbook_10.csv"

# -------------------- LOBSTER column names --------------------
# Order book with 10 levels: AskPrice1, AskSize1, BidPrice1, BidSize1,etc.
names = []
for i in range(1, 11):
    names += [f"AskPrice{i}", f"AskSize{i}", f"BidPrice{i}", f"BidSize{i}"]

# -------------------- Loaders --------------------
def load_microprice(path, n_steps=5000):   #microprice is better for short-term strategies,
    #  e.g. market making as it can be used to model market microstructure
    """Microprice at top of book for first n_steps."""
    df = pd.read_csv(path, header=None, names=names, nrows=n_steps)
    a1  = df["AskPrice1"].astype(float)
    qa1 = df["AskSize1"].astype(float)
    b1  = df["BidPrice1"].astype(float)
    qb1 = df["BidSize1"].astype(float)
    micro = (a1 * qb1 + b1 * qa1) / (qa1 + qb1)
    return micro

def load_midprice(path, n_steps=5000):
    """Mid price (best bid/ask) for first n_steps."""
    df = pd.read_csv(path, header=None, names=names, nrows=n_steps)
    a1  = df["AskPrice1"].astype(float)
    b1  = df["BidPrice1"].astype(float)
    return (a1 + b1) / 2.0

# -------------------- Data: microprice --------------------
mp_tsla = load_microprice(path_stock_a, n_steps=5000)
mp_intc = load_microprice(path_stock_b, n_steps=5000)
mp_pcln = load_microprice(path_stock_c, n_steps=5000)

min_len = min(len(mp_tsla), len(mp_intc), len(mp_pcln))
mp_tsla = mp_tsla.iloc[:min_len].reset_index(drop=True)
mp_intc = mp_intc.iloc[:min_len].reset_index(drop=True)
mp_pcln = mp_pcln.iloc[:min_len].reset_index(drop=True)

# Plot microprice time series
plt.figure(figsize=(12, 6))
plt.plot(mp_tsla, label="TSLA microprice")
plt.plot(mp_intc, label="INTC microprice")
plt.plot(mp_pcln, label="PCLN microprice")
plt.title("Top-of-Book Microprice (LOBSTER)")
plt.xlabel("Book update index")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# Returns from microprice
ret_tsla = np.log(mp_tsla / mp_tsla.shift(1)).dropna()
ret_intc = np.log(mp_intc / mp_intc.shift(1)).dropna()
ret_pcln = np.log(mp_pcln / mp_pcln.shift(1)).dropna()

# Distributions (KDE) zoomed
plt.figure(figsize=(12, 6))
sns.kdeplot(ret_tsla, bw_adjust=0.5, label="TSLA", fill=True, alpha=0.4)
sns.kdeplot(ret_intc, bw_adjust=0.5, label="INTC", fill=True, alpha=0.4)
sns.kdeplot(ret_pcln, bw_adjust=0.5, label="PCLN", fill=True, alpha=0.4)
plt.title("Distribution of Microprice Log Returns (Zoomed)")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.xlim(-0.0005, 0.0005)
plt.legend()
plt.tight_layout()
plt.show()

print("Tesla micro returns:\n", ret_tsla.describe(), "\n")
print("Intel micro returns:\n", ret_intc.describe(), "\n")
print("Booking.com micro returns:\n", ret_pcln.describe(), "\n")

# -------------------- NORMAL MMD (unbiased) --------------------
def mmd_rbf_unbiased(x, y, gamma=None, sample_for_gamma=2000, random_state=42):
    """
    Unbiased MMD with RBF kernel using full kernel matrices.
    x, y: 1D arrays
    gamma: if None, uses median heuristic on a (possibly) subsampled pool to pick bandwidth
    sample_for_gamma: max pool size for gamma selection (keeps memory reasonable)
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    m, n = x.shape[0], y.shape[0]

    if gamma is None:
        rng = np.random.default_rng(random_state)
        z = np.vstack([x, y]).ravel()  # 1D values
        if len(z) > sample_for_gamma:
            z = rng.choice(z, size=sample_for_gamma, replace=False)
        z = z.reshape(-1, 1)
        # pairwise squared distances on the (small) sample
        D = pairwise_distances(z, metric="euclidean")
        d2 = (D ** 2)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else np.median(d2)
        if not np.isfinite(med) or med <= 0:
            med = np.mean(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / (2.0 * med)

    # Full kernel matrices
    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    # Unbiased estimator: remove diagonals in Kxx, Kyy
    mmd2 = ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2.0 * Kxy.mean())
    mmd2 = max(mmd2, 0.0)  # numerical guard
    return float(np.sqrt(mmd2))

# Pairwise MMDs (microprice returns)
x_tsla = ret_tsla.values
x_intc = ret_intc.values
x_pcln = ret_pcln.values

mmd_tsla_intc = mmd_rbf_unbiased(x_tsla, x_intc)
mmd_tsla_pcln = mmd_rbf_unbiased(x_tsla, x_pcln)
mmd_intc_pcln = mmd_rbf_unbiased(x_intc, x_pcln)

print(f"MMD(TSLA, INTC) [micro returns]: {mmd_tsla_intc:.6g}")
print(f"MMD(TSLA, PCLN) [micro returns]: {mmd_tsla_pcln:.6g}")
print(f"MMD(INTC, PCLN) [micro returns]: {mmd_intc_pcln:.6g}")

# -------------------- Mid-price workflow --------------------
mid_tsla = load_midprice(path_stock_a, n_steps=5000)
mid_intc = load_midprice(path_stock_b, n_steps=5000)
mid_pcln = load_midprice(path_stock_c, n_steps=5000)

L = min(len(mid_tsla), len(mid_intc), len(mid_pcln))
mid_tsla = mid_tsla.iloc[:L].reset_index(drop=True)
mid_intc = mid_intc.iloc[:L].reset_index(drop=True)
mid_pcln = mid_pcln.iloc[:L].reset_index(drop=True)

ret_mid_tsla = np.log(mid_tsla / mid_tsla.shift(1)).dropna()
ret_mid_intc = np.log(mid_intc / mid_intc.shift(1)).dropna()
ret_mid_pcln = np.log(mid_pcln / mid_pcln.shift(1)).dropna()

plt.figure(figsize=(12, 6))
sns.kdeplot(ret_mid_tsla, bw_adjust=0.5, label="TSLA (mid)", fill=True, alpha=0.4)
sns.kdeplot(ret_mid_intc, bw_adjust=0.5, label="INTC (mid)", fill=True, alpha=0.4)
sns.kdeplot(ret_mid_pcln, bw_adjust=0.5, label="PCLN (mid)", fill=True, alpha=0.4)
plt.title("Distribution of Mid-Price Log Returns (Zoomed)")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.xlim(-0.0005, 0.0005)
plt.legend()
plt.tight_layout()
plt.show()

print("TSLA mid returns:\n", ret_mid_tsla.describe(), "\n")
print("INTC mid returns:\n", ret_mid_intc.describe(), "\n")
print("PCLN mid returns:\n", ret_mid_pcln.describe(), "\n")

# Pairwise MMDs (mid-price returns) with standard MMD
mmd_tsla_intc_mid = mmd_rbf_unbiased(ret_mid_tsla.values, ret_mid_intc.values)
mmd_tsla_pcln_mid = mmd_rbf_unbiased(ret_mid_tsla.values, ret_mid_pcln.values)
mmd_intc_pcln_mid = mmd_rbf_unbiased(ret_mid_intc.values, ret_mid_pcln.values)

print(f"MMD(TSLA_mid, INTC_mid): {mmd_tsla_intc_mid:.6g}")
print(f"MMD(TSLA_mid, PCLN_mid): {mmd_tsla_pcln_mid:.6g}")
print(f"MMD(INTC_mid, PCLN_mid): {mmd_intc_pcln_mid:.6g}")

# Optional: matrix view
labels = ["TSLA (mid)", "INTC (mid)", "PCLN (mid)"]
mmd_matrix_mid = pd.DataFrame(
    [
        [0.0,                    mmd_tsla_intc_mid, mmd_tsla_pcln_mid],
        [mmd_tsla_intc_mid,      0.0,               mmd_intc_pcln_mid],
        [mmd_tsla_pcln_mid,      mmd_intc_pcln_mid, 0.0]
    ],
    index=labels, columns=labels
)
print("\nPairwise MMD (mid-price returns):")
print(mmd_matrix_mid)
