import pandas as pd
import numpy as np
from math import log2

# Load dataset
df = pd.read_csv("features.csv")
y = df["genre"]  # target column

# Equal-width binning function
def equal_width_binning(series, n_bins=4):
    labels = [f"bin_{i+1}" for i in range(n_bins)]
    return pd.cut(series, bins=n_bins, labels=labels, include_lowest=True).astype(str)

# Entropy function
def entropy(values: pd.Series) -> float:
    counts = values.value_counts().values
    probs = counts / counts.sum()
    return -np.sum([p * log2(p) for p in probs if p > 0])

# A1 result
print("Entropy of genre:", entropy(y))

# Example binning demo on tempo
print(equal_width_binning(df["tempo"], n_bins=4).head())
