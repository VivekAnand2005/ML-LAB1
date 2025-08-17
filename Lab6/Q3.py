# === A3: Root Node Selection using Information Gain (standalone) ===
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
                x = pd.qcut(X[col], q=n_bins, labels=[f"bin_{i}" for i in range(n_bins)], duplicates="drop")
        else:
            x = X[col].astype(str)
        ig_scores[col] = information_gain(y, x)
    return max(ig_scores, key=ig_scores.get), ig_scores

# --- Run A3 ---
best_feature, scores = choose_root_node(X, y, binning="width")
print("Best root feature (equal-width binning):", best_feature)

# Show top 5 features
sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
print("\nTop 5 features by Information Gain:")
for feat, val in sorted_scores:
    print(f"{feat:20s} -> IG = {val:.6f}")
