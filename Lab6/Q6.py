# === A6: Visualize a Decision Tree with sklearn ===
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("features.csv")   # adjust path if needed
y = df["genre"].astype(str)
X = df.drop(columns=["file_path", "genre"])

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Use only numeric features for sklearn tree
X_num = X.select_dtypes(include=["number"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_num, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
print("Test Accuracy:", clf.score(X_test, y_test))

# Plot the tree
plt.figure(figsize=(18, 10))
plot_tree(
    clf,
    feature_names=X_num.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization (A6)")
plt.show()
