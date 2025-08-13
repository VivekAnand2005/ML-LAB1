import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_csv("features.csv")

# Prepare features for clustering
X = df.drop(columns=['spectral_centroid', 'genre', 'file_path'])

# Train-test split (optional, here using only train set for scoring)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Fit KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_train)

# Calculate metrics
sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_score = davies_bouldin_score(X_train, kmeans.labels_)

# Output
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski–Harabasz Score: {ch_score:.4f}")
print(f"Davies–Bouldin Score: {db_score:.4f}")
