import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("features.csv")

# Feature (using only 'tempo') and Target ('spectral_centroid')
X = df[['tempo']]
y = df['spectral_centroid']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Output
print("Model Coefficient:", reg.coef_[0])
print("Model Intercept:", reg.intercept_)
print("First 5 predictions on training data:", y_train_pred[:5])
