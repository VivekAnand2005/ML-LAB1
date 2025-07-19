import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df_purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Create a new column 'Label' where:
df_purchase['Label'] = df_purchase['Payment (Rs)'].apply(lambda x: 1 if x > 200 else 0)

# Select the features and target label
X = df_purchase[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]  # Feature matrix
y = df_purchase['Label']  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Classification Report for RICH vs POOR Classification:")
print(classification_report(y_test, y_pred, target_names=["POOR", "RICH"]))
