import pandas as pd
import numpy as np

df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

print("Previewing column value types:")
for col in df_thyroid.columns:
    print(f"{col}: {df_thyroid[col].dropna().unique()[:5]}")

df_binary_like = df_thyroid.copy()
df_binary_like.replace({
    't': 1, 'f': 0,
    'T': 1, 'F': 0,
    'yes': 1, 'no': 0,
    'TRUE': 1, 'FALSE': 0,
    'True': 1, 'False': 0,
    'Y': 1, 'N': 0
}, inplace=True)

binary_cols = []
for col in df_binary_like.columns:
    unique_vals = df_binary_like[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}):
        binary_cols.append(col)

df_binary = df_binary_like[binary_cols]

print(f"\nSelected binary columns ({len(binary_cols)}): {binary_cols}")

if len(binary_cols) == 0:
    print("No binary columns found after conversion.")
elif len(df_binary) < 2:
    print("Dataset has fewer than 2 rows.")
else:
    vec1 = df_binary.iloc[0].values
    vec2 = df_binary.iloc[1].values

    def jc_smc(v1, v2):
        f11 = np.sum((v1 == 1) & (v2 == 1))
        f00 = np.sum((v1 == 0) & (v2 == 0))
        f10 = np.sum((v1 == 1) & (v2 == 0))
        f01 = np.sum((v1 == 0) & (v2 == 1))

        denom_jc = f11 + f10 + f01
        jc = f11 / denom_jc if denom_jc > 0 else 0

        denom_smc = f11 + f00 + f10 + f01
        smc = (f11 + f00) / denom_smc if denom_smc > 0 else 0

        return jc, smc

    jc_value, smc_value = jc_smc(vec1, vec2)

    print("\n First row vector:", vec1)
    print("Second row vector:", vec2)
    print("\n Jaccard Coefficient (JC):", jc_value)
    print("Simple Matching Coefficient (SMC):", smc_value)
