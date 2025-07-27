import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Load dataset
df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Custom kNN implementation
def custom_knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = [np.linalg.norm(test_point - x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        predictions.append(Counter(k_labels).most_common(1)[0][0])
    return np.array(predictions)

# Predictions using custom kNN
y_pred_custom = custom_knn_predict(X_train, y_train, X_test, k=3)
custom_accuracy = np.mean(y_pred_custom == y_test)

# sklearn kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
sklearn_accuracy = knn.score(X_test, y_test)

print(f"Custom kNN Accuracy: {custom_accuracy:.4f}")
print(f"sklearn kNN Accuracy: {sklearn_accuracy:.4f}")
