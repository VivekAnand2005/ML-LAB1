`import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values

# Convert continuous labels into 3 categories
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

unique_classes = np.unique(y)
class1, class2 = unique_classes[0], unique_classes[1]

X1 = X[y == class1]
X2 = X[y == class2]

centroid1 = np.mean(X1, axis=0)
centroid2 = np.mean(X2, axis=0)

spread1 = np.std(X1, axis=0)
spread2 = np.std(X2, axis=0)

interclass_distance = np.linalg.norm(centroid1 - centroid2)

print(f"Class {class1} Centroid:\n", centroid1)
print(f"Class {class1} Spread:\n", spread1)
print(f"\nClass {class2} Centroid:\n", centroid2)
print(f"Class {class2} Spread:\n", spread2)
print(f"\nInter-class Distance: {interclass_distance}")
