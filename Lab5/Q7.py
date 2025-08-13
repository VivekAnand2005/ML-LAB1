import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("features.csv")

# Prepare features (remove target + non-numeric columns)
X = df.drop(columns=['spectral_centroid', 'genre', 'file_path'])

# Store inertia values (distortions)
distortions = []

# Range of k values to test
k_range = range(2, 20)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Distortion)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()
