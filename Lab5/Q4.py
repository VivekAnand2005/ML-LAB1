import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("features.csv")

# Prepare data: remove target and non-numeric columns
X = df.drop(columns=['spectral_centroid', 'genre', 'file_path'])

# Train-test split (clustering usually uses all data, but we'll follow lab style)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_train)

# Output results
print("Cluster Labels:", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
