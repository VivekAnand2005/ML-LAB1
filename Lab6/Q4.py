# === A4: Root Node with Equal-Width & Equal-Frequency Binning (standalone) ===
import pandas as pd
import numpy as np
from math import log2

# Load dataset
df = pd.read_csv("features.csv")   # adjust path if needed
y = df["genre"].astype(str)
X = df.drop(columns=["file_path", "genre"])

# --- Helper functions ---
def entropy(values: pd.Series) -> float:
    counts = values.value_counts().values
    probs = counts / counts.sum()
    return -np.sum([p * log2(p) for p in probs if p > 0])

def equal_width_binning(series, n_bins=4):
    labels = [f"bin_{i+1}" for i in range(n_bins)]
    return pd.cut(series, bins=n_bins, labels=labels, include_lowest=True).astype(str)

def equal_frequency_binning(series, n_bins=4):
    labels = [f"bin_{i+1}" for i in range(n_bins)]
    return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop").astype(str)

def information_gain(y, x_cat):
    base_entropy = entropy(y)
    ig = base_entropy
    for v in x_cat.unique():
        subset = y[x_cat == v]
        weight = len(subset) / len(y)
        ig -= weight * entropy(subset)
    return ig

def choose_root_node(X, y, binning="width", n_bins=4):
    ig_scores = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if binning == "width":
                x = equal_width_binning(X[col], n_bins=n_bins)
            else:
                x = equal_frequency_binning(X[col], n_bins=n_bins)
        else:
            x = X[col].astype(str)
        ig_scores[col] = information_gain(y, x)
    return max(ig_scores, key=ig_scores.get), ig_scores

# --- Run A4 ---
best_width, scores_width = choose_root_node(X, y, binning="width")
best_freq, scores_freq = choose_root_node(X, y, binning="freq")

print("Best root feature (equal-width binning):", best_width)
print("Best root feature (equal-frequency binning):", best_freq)

# Show top 5 features for each
print("\nTop 5 features (Equal-Width):")
for feat, val in sorted(scores_width.items(), key=lambda kv: kv[1], reverse=True)[:5]:
    print(f"{feat:20s} -> IG = {val:.6f}")

print("\nTop 5 features (Equal-Frequency):")
for feat, val in sorted(scores_freq.items(), key=lambda kv: kv[1], reverse=True)[:5]:
    print(f"{feat:20s} -> IG = {val:.6f}")
