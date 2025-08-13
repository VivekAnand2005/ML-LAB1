import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("features.csv")

# Prepare features (remove target + non-numeric columns)
X = df.drop(columns=['spectral_centroid', 'genre', 'file_path'])

# Range of k values
k_values = range(2, 11)

# Lists to store scores
sil_scores = []
ch_scores = []
db_scores = []

# Compute metrics for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    sil_scores.append(silhouette_score(X, kmeans.labels_))
    ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X, kmeans.labels_))

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, sil_scores, marker='o', color='b')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.grid(True)
plt.show()

# Plot Calinski–Harabasz Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, ch_scores, marker='s', color='g')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Calinski–Harabasz Score")
plt.title("Calinski–Harabasz Score vs k")
plt.grid(True)
plt.show()

# Plot Davies–Bouldin Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, db_scores, marker='^', color='r')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Davies–Bouldin Score")
plt.title("Davies–Bouldin Score vs k")
plt.grid(True)
plt.show()
