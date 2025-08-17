# === A5: Build a Custom Decision Tree (ID3-like) ===
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

def choose_root_node(X, y, n_bins=4):
    ig_scores = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            x = equal_width_binning(X[col], n_bins=n_bins)
        else:
            x = X[col].astype(str)
        ig_scores[col] = information_gain(y, x)
    return max(ig_scores, key=ig_scores.get), ig_scores

# --- Decision Tree Class (ID3-like) ---
class Node:
    def __init__(self, feature=None, children=None, prediction=None):
        self.feature = feature
        self.children = children or {}
        self.prediction = prediction

def build_tree(X, y, depth=0, max_depth=3):
    # If all labels are same → leaf
    if len(set(y)) == 1:
        return Node(prediction=y.iloc[0])
    
    # If depth limit reached or no features left → majority class
    if depth == max_depth or X.empty:
        return Node(prediction=y.mode()[0])
    
    # Choose best feature
    best_feature, _ = choose_root_node(X, y)
    root = Node(feature=best_feature)
    
    # Bin values of the chosen feature
    x_binned = equal_width_binning(X[best_feature])
    
    for val in x_binned.unique():
        mask = x_binned == val
        if mask.sum() == 0:
            continue
        child = build_tree(X[mask].drop(columns=[best_feature]), y[mask], depth+1, max_depth)
        root.children[val] = child
    
    return root

# --- Print the tree ---
def print_tree(node, depth=0):
    prefix = "  " * depth
    if node.prediction is not None:
        print(f"{prefix}Predict → {node.prediction}")
    else:
        print(f"{prefix}[Split on: {node.feature}]")
        for val, child in node.children.items():
            print(f"{prefix} └── if {node.feature} = {val}:")
            print_tree(child, depth+1)

# --- Run A5 ---
tree = build_tree(X, y, max_depth=3)
print("=== A5: Custom Decision Tree (ID3-like) ===")
print_tree(tree)
