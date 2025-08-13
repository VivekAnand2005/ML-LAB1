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

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    sil_scores.append(silhouette_score(X, kmeans.labels_))
    ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X, kmeans.labels_))

# Plot the scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, sil_scores, marker='o', label='Silhouette Score')
plt.plot(k_values, ch_scores, marker='s', label='Calinski-Harabasz Score')
plt.plot(k_values, db_scores, marker='^', label='Davies-Bouldin Score')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")
plt.title("KMeans Clustering Metrics vs k")
plt.legend()
plt.grid(True)
plt.show()
