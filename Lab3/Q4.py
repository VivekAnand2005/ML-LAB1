import pandas as pd
import numpy as np                    # ✅ Added this import
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values
y_continuous = df.iloc[:, -1].values

# Convert continuous labels into 3 categories
y = pd.qcut(y_continuous, q=3, labels=["Low", "Medium", "High"])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Train Set Size:", X_train.shape)
print("Test Set Size:", X_test.shape)
