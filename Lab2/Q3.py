import pandas as pd

df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Data types
data_types = df_thyroid.dtypes

# Missing values
missing_vals = df_thyroid.isnull().sum()

# Numeric columns only
df_num = df_thyroid.select_dtypes(include='number')

# Numeric stats
numeric_stats = df_num.describe()

# IQR for numeric columns
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1

# Detect outliers (count how many values are below Q1 - 1.5*IQR or above Q3 + 1.5*IQR)
outliers = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).sum()

print("Data types:\n", data_types)
print("\nMissing values:\n", missing_vals)
print("\nNumeric stats:\n", numeric_stats)
print("\nOutliers count:\n", outliers)
