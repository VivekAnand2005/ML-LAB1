import pandas as pd
import numpy as np

df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Handle missing values appropriately
for column in df_thyroid.columns:
    if df_thyroid[column].isnull().sum() > 0:
        if df_thyroid[column].dtype == 'object':
            mode_val = df_thyroid[column].mode()[0]
            df_thyroid[column].fillna(mode_val, inplace=True)
        else:
            Q1 = df_thyroid[column].quantile(0.25)
            Q3 = df_thyroid[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            has_outliers = ((df_thyroid[column] < lower_bound) | (df_thyroid[column] > upper_bound)).any()
            
            if has_outliers:
                median_val = df_thyroid[column].median()
                df_thyroid[column].fillna(median_val, inplace=True)
            else:
                mean_val = df_thyroid[column].mean()
                df_thyroid[column].fillna(mean_val, inplace=True)

missing_after = df_thyroid.isnull().sum().sum()
print("Total missing values after imputation:", missing_after)
