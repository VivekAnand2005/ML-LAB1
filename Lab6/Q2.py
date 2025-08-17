# === A2: Gini index (standalone) ===
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("features.csv")  # change path if needed

# Target column
y = df["genre"].astype(str)

# Gini index function
def gini_index(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False).values.astype(float)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))

# Result
print("Gini impurity (genre):", gini_index(y))
