# === A7: Decision Boundary with 2 Features ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load dataset
df = pd.read_csv("features.csv")   # adjust path if needed
y = df["genre"].astype(str)

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Select 2 numeric features for visualization
feat1, feat2 = "spectral_centroid", "tempo"
X2 = df[[feat1, feat2]]

# For clarity, take only top 2 most common classes
class_counts = Counter(y_enc)
top2 = [c for c, _ in class_counts.most_common(2)]
mask = np.isin(y_enc, top2)
X2, y2 = X2[mask], y_enc[mask]

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X2, y2)

# Create meshgrid for decision boundary
x_min, x_max = X2[feat1].min() - 1, X2[feat1].max() + 1
y_min, y_max = X2[feat2].min() - 1, X2[feat2].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict for grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2[feat1], X2[feat2], c=y2, edgecolors="k", cmap="coolwarm")
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("Decision Boundary (A7)")
plt.show()

print("Classes shown:", [le.classes_[c] for c in top2])
