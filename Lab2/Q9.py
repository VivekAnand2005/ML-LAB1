import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

numeric_cols = df_thyroid.select_dtypes(include=[np.number]).columns
df_numeric = df_thyroid[numeric_cols]

# Min-Max Scaling
scaler = MinMaxScaler()
df_scaled = df_numeric.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_numeric[numeric_cols])

df_thyroid[numeric_cols] = df_scaled[numeric_cols]

print("Sample normalized values (0 to 1 range):")
print(df_thyroid[numeric_cols].head())
