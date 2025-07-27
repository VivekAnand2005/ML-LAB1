import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

metrics = ["euclidean", "manhattan", "chebyshev"]

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f"Metric: {metric}, Accuracy: {accuracy:.4f}")
