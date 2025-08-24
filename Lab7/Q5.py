"""
A5 – Clustering (Agglomerative + DBSCAN)
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN

RANDOM_STATE = 42
DATA_PATH = Path("/mnt/data/features.csv")

def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["file_path", "genre"])
    return X, df["genre"]

def main():
    print("=== A5 Clustering ===")
    X, y_true = load_data(DATA_PATH)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Agglomerative clustering
    agg = AgglomerativeClustering(n_clusters=10)  # you can adjust cluster count
    agg_labels = agg.fit_predict(X_scaled)

    print("\nAgglomerative Clustering:")
    print(f"Number of clusters: {len(set(agg_labels))}")
    print(pd.Series(agg_labels).value_counts())

    # DBSCAN clustering
    db = DBSCAN(eps=3, min_samples=5)  # tune eps/min_samples if needed
    db_labels = db.fit_predict(X_scaled)

    print("\nDBSCAN Clustering:")
    print(f"Number of clusters (excluding noise): {len(set(db_labels)) - (1 if -1 in db_labels else 0)}")
    print(pd.Series(db_labels).value_counts())

    # Save results
    df_results = pd.DataFrame({
        "true_genre": y_true,
        "agg_cluster": agg_labels,
        "dbscan_cluster": db_labels
    })

    out_path = Path("/mnt/data/a5_clustering_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved clustering assignments to {out_path}")

if __name__ == "__main__":
    main()
