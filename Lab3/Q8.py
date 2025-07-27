import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k_values = range(1, 12)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.title('kNN Accuracy vs k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()
