import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Paths --------------------
path_stock_a = "C:/Users/Fabia/OneDrive/Desktop/TSLA_2015-01-01_2015-03-31_10/output-2015/0/0/3/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv"
path_stock_b = "C:/Users/Fabia/OneDrive/Desktop/INTC_2015-01-01_2015-03-31_10/output-2015/0/0/1/INTC_2015-01-02_34200000_57600000_orderbook_10.csv"
path_stock_c = "C:/Users/Fabia/OneDrive/Desktop/PCLN_2015-01-01_2015-03-31_10/output-2015/0/0/4/PCLN_2015-01-02_34200000_57600000_orderbook_10.csv"

# -------------------- LOBSTER column names --------------------
names = []
for i in range(1, 11):
    names += [f"AskPrice{i}", f"AskSize{i}", f"BidPrice{i}", f"BidSize{i}"]

# <<< NEW: tiny helper to infer tick size from the file
def infer_tick_size_from_lobster(path, nrows=20000):
    df = pd.read_csv(path, header=None, names=names, nrows=nrows)
    prices = []
    for i in range(1, 11):
        prices.append(df[f"AskPrice{i}"].values)
        prices.append(df[f"BidPrice{i}"].values)
    prices = np.unique(np.concatenate(prices))
    diffs = np.diff(prices)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.min(diffs))

def load_midprice(path, n_steps=5000):
    df = pd.read_csv(path, header=None, names=names, nrows=n_steps)
    a1  = df["AskPrice1"].astype(float)
    b1  = df["BidPrice1"].astype(float)
    return (a1 + b1) / 2.0

# ---------- multi-level microprice ----------
def microprice_calculation(bid_prices, bid_volumes, ask_prices, ask_volumes,
                           alpha_decay, tick_size=0.1, max_levels=5,
                           clamp_inside=True, max_distance_ticks=None, min_weight=None):

    bid_prices = np.asarray(bid_prices, dtype=float)
    bid_volumes = np.asarray(bid_volumes, dtype=float)
    ask_prices = np.asarray(ask_prices, dtype=float)
    ask_volumes = np.asarray(ask_volumes, dtype=float)

    # Ensure sorted best→worse
    bid_sort = np.argsort(bid_prices)[::-1]
    ask_sort = np.argsort(ask_prices)
    bid_prices, bid_volumes = bid_prices[bid_sort], bid_volumes[bid_sort]
    ask_prices, ask_volumes = ask_prices[ask_sort], ask_volumes[ask_sort]

    # Limit depth
    if max_levels is not None and max_levels > 0:
        bid_prices, bid_volumes = bid_prices[:max_levels], bid_volumes[:max_levels]
        ask_prices, ask_volumes = ask_prices[:max_levels], ask_volumes[:max_levels]

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]

    # Distances in ticks (robust to scaling)
    if tick_size and tick_size > 0:
        bid_distances = np.maximum(np.round((best_bid - bid_prices) / tick_size), 0)
        ask_distances = np.maximum(np.round((ask_prices - best_ask) / tick_size), 0)
    else:
        bid_distances = np.arange(len(bid_prices))   # <<< NEW fallback to level distance
        ask_distances = np.arange(len(ask_prices))

    # Optional distance gate
    if max_distance_ticks is not None:
        b_keep = bid_distances <= max_distance_ticks
        a_keep = ask_distances <= max_distance_ticks
        bid_prices, bid_volumes, bid_distances = bid_prices[b_keep], bid_volumes[b_keep], bid_distances[b_keep]
        ask_prices, ask_volumes, ask_distances = ask_prices[a_keep], ask_volumes[a_keep], ask_distances[a_keep]

    # Exponential decay weights
    bid_weights = np.exp(-alpha_decay * bid_distances)
    ask_weights = np.exp(-alpha_decay * ask_distances)

    # Optional weight gate
    if min_weight is not None:
        bid_keep = bid_weights >= min_weight
        ask_keep = ask_weights >= min_weight
        bid_prices, bid_volumes, bid_weights = bid_prices[bid_keep], bid_volumes[bid_keep], bid_weights[bid_keep]
        ask_prices, ask_volumes, ask_weights = ask_prices[ask_keep], ask_volumes[ask_keep], ask_weights[ask_keep]

    # Effective volumes
    eff_bid_vol = np.sum(bid_weights * bid_volumes)
    eff_ask_vol = np.sum(ask_weights * ask_volumes)

    # Guards
    if eff_bid_vol <= 0 or eff_ask_vol <= 0:
        if len(bid_prices) > 0 and len(ask_prices) > 0 and bid_volumes.sum() + ask_volumes.sum() > 0:
            mp_tob = (ask_prices[0] * bid_volumes[0] + bid_prices[0] * ask_volumes[0]) / (bid_volumes[0] + ask_volumes[0])
            return float(mp_tob)
        return float((best_bid + best_ask) / 2.0)

    # Effective prices
    eff_bid_price = np.sum(bid_weights * bid_prices * bid_volumes) / eff_bid_vol
    eff_ask_price = np.sum(ask_weights * ask_prices * ask_volumes) / eff_ask_vol

    # Multi-level microprice
    fair_value = (eff_bid_price * eff_ask_vol + eff_ask_price * eff_bid_vol) / (eff_bid_vol + eff_ask_vol)
    if clamp_inside:
        fair_value = min(max(fair_value, best_bid), best_ask)

    return float(fair_value)

# ---------- Loader using multi-level microprice ----------
def load_microprice_multilevel(path, n_steps=5000,
                               alpha_decay=0.3,      # <<< CHANGED: milder default
                               max_levels=5,
                               tick_size=None,       # <<< CHANGED: allow None => infer
                               clamp_inside=True, max_distance_ticks=None, min_weight=None):
    df = pd.read_csv(path, header=None, names=names, nrows=n_steps)

    # Infer tick size if not provided
    if tick_size is None:                           # <<< NEW
        tick_size = infer_tick_size_from_lobster(path, nrows=min(20000, n_steps))

    ask_price_cols = [f"AskPrice{i}" for i in range(1, 11)]
    ask_size_cols  = [f"AskSize{i}"  for i in range(1, 11)]
    bid_price_cols = [f"BidPrice{i}" for i in range(1, 11)]
    bid_size_cols  = [f"BidSize{i}"  for i in range(1, 11)]

    mp = []
    for _, row in df.iterrows():
        ask_prices = row[ask_price_cols].to_numpy(dtype=float, copy=False)
        ask_sizes  = row[ask_size_cols ].to_numpy(dtype=float, copy=False)
        bid_prices = row[bid_price_cols].to_numpy(dtype=float, copy=False)
        bid_sizes  = row[bid_size_cols ].to_numpy(dtype=float, copy=False)

        fair = microprice_calculation(
            bid_prices=bid_prices, bid_volumes=bid_sizes,
            ask_prices=ask_prices, ask_volumes=ask_sizes,
            alpha_decay=alpha_decay, tick_size=tick_size,
            max_levels=max_levels, clamp_inside=clamp_inside,
            max_distance_ticks=max_distance_ticks, min_weight=min_weight
        )
        mp.append(fair)

    return pd.Series(mp, name="microprice")

# ---------- Use multi-level microprice ----------
ALPHA   = 0.3          # <<< CHANGED: gentler decay
LEVELS  = 5
TICK    = None         # <<< CHANGED: None => infer automatically
CLAMP   = True
MAXDIST = 5            # <<< NEW: consider up to 5 ticks from best
MINW    = None

mp_tsla = load_microprice_multilevel(path_stock_a, n_steps=5000,
                                     alpha_decay=ALPHA, max_levels=LEVELS,
                                     tick_size=TICK, clamp_inside=CLAMP,
                                     max_distance_ticks=MAXDIST, min_weight=MINW)
mp_intc = load_microprice_multilevel(path_stock_b, n_steps=5000,
                                     alpha_decay=ALPHA, max_levels=LEVELS,
                                     tick_size=TICK, clamp_inside=CLAMP,
                                     max_distance_ticks=MAXDIST, min_weight=MINW)
mp_pcln = load_microprice_multilevel(path_stock_c, n_steps=5000,
                                     alpha_decay=ALPHA, max_levels=LEVELS,
                                     tick_size=TICK, clamp_inside=CLAMP,
                                     max_distance_ticks=MAXDIST, min_weight=MINW)

# Align lengths
min_len = min(len(mp_tsla), len(mp_intc), len(mp_pcln))
mp_tsla = mp_tsla.iloc[:min_len].reset_index(drop=True)
mp_intc = mp_intc.iloc[:min_len].reset_index(drop=True)
mp_pcln = mp_pcln.iloc[:min_len].reset_index(drop=True)

# <<< NEW: quick sanity check vs TOB
def top_of_book_microprice_series(path, n_steps=5000):
    df = pd.read_csv(path, header=None, names=names, nrows=n_steps)
    a1 = df["AskPrice1"].astype(float).to_numpy()
    b1 = df["BidPrice1"].astype(float).to_numpy()
    qa1 = df["AskSize1"].astype(float).to_numpy()
    qb1 = df["BidSize1"].astype(float).to_numpy()
    return pd.Series((a1*qb1 + b1*qa1) / (qa1 + qb1), name="mp_tob")

mp_tob_tsla = top_of_book_microprice_series(path_stock_a, n_steps=min_len)
corr = np.corrcoef(mp_tsla.values, mp_tob_tsla.values)[0,1]
same = np.mean(np.isclose(mp_tsla.values, mp_tob_tsla.values, rtol=0, atol=1e-9))
print(f"[Sanity] TSLA corr(multi, TOB)={corr:.6f}, exactly_equal_fraction={same:.4f}")

# Plot: index to 100 (label fixed)
SCALE = 10000.0
def plot_three_views(name, series, scale=SCALE):
    s = series.astype(float) / scale
    s_norm = 100 * s / s.iloc[0]
    plt.figure(figsize=(12, 3))
    plt.plot(s_norm)
    plt.title(f"{name}: Multi-level Microprice (indexed to 100 at t0)")  # <<< CHANGED
    plt.ylabel("Index (t0 = 100)")
    plt.xlabel("Book update index")
    plt.tight_layout()
    plt.show()

plot_three_views("TSLA", mp_tsla)
plot_three_views("INTC", mp_intc)
plot_three_views("PCLN", mp_pcln)

# Raw series plot (label fixed)
plt.figure(figsize=(12, 6))
plt.plot(mp_tsla, label="TSLA multi-level microprice")
plt.plot(mp_intc, label="INTC multi-level microprice")
plt.plot(mp_pcln, label="PCLN multi-level microprice")
plt.title("Multi-level Microprice (LOBSTER)")  # <<< CHANGED
plt.xlabel("Book update index")
plt.ylabel("Price (ticks)")
plt.legend()
plt.tight_layout()
plt.show()

# Returns
ret_tsla = np.log(mp_tsla / mp_tsla.shift(1)).dropna()
ret_intc = np.log(mp_intc / mp_intc.shift(1)).dropna()
ret_pcln = np.log(mp_pcln / mp_pcln.shift(1)).dropna()

# KDE of returns
plt.figure(figsize=(12, 6))
sns.kdeplot(ret_tsla, bw_adjust=0.5, label="TSLA", fill=True, alpha=0.4)
sns.kdeplot(ret_intc, bw_adjust=0.5, label="INTC", fill=True, alpha=0.4)
sns.kdeplot(ret_pcln, bw_adjust=0.5, label="PCLN", fill=True, alpha=0.4)
plt.title("Distribution of Multi-level Microprice Log Returns (Zoomed)")  # <<< CHANGED
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.xlim(-0.0005, 0.0005)
plt.legend()
plt.tight_layout()
plt.show()

print("Tesla micro returns:\n", ret_tsla.describe(), "\n")
print("Intel micro returns:\n", ret_intc.describe(), "\n")
print("Booking.com micro returns:\n", ret_pcln.describe(), "\n")

# -------------------- Unbiased MMD --------------------
def mmd_rbf_unbiased(x, y, gamma=None, sample_for_gamma=2000, random_state=42):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    m, n = x.shape[0], y.shape[0]

    if gamma is None:
        rng = np.random.default_rng(random_state)
        z = np.vstack([x, y]).ravel()
        if len(z) > sample_for_gamma:
            z = rng.choice(z, size=sample_for_gamma, replace=False)
        z = z.reshape(-1, 1)
        D = pairwise_distances(z, metric="euclidean")
        d2 = (D ** 2)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else np.median(d2)
        if not np.isfinite(med) or med <= 0:
            med = np.mean(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / (2.0 * med)

    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    mmd2 = ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2.0 * Kxy.mean())
    mmd2 = max(mmd2, 0.0)
    return float(np.sqrt(mmd2))

# Pairwise MMDs (multi-level microprice returns)
x_tsla = ret_tsla.values
x_intc = ret_intc.values
x_pcln = ret_pcln.values

mmd_tsla_intc = mmd_rbf_unbiased(x_tsla, x_intc)
mmd_tsla_pcln = mmd_rbf_unbiased(x_tsla, x_pcln)
mmd_intc_pcln = mmd_rbf_unbiased(x_intc, x_pcln)

print(f"MMD(TSLA, INTC) [multi-level micro returns]: {mmd_tsla_intc:.6g}")
print(f"MMD(TSLA, PCLN) [multi-level micro returns]: {mmd_tsla_pcln:.6g}")
print(f"MMD(INTC, PCLN) [multi-level micro returns]: {mmd_intc_pcln:.6g}")
