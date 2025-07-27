import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Binarize labels for multi-class AUROC
y_test_bin = label_binarize(y_test, classes=["Low", "Medium", "High"])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_score = knn.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
for i, label in enumerate(["Low", "Medium", "High"]):
    plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC for kNN Classifier')
plt.legend(loc="lower right")
plt.show()
