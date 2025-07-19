import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

df_binary = df_thyroid.select_dtypes(include='int64')
df_binary = df_binary.loc[:, df_binary.nunique() <= 2]
binary_20 = df_binary.iloc[:20]

df_numeric = df_thyroid.select_dtypes(include=[np.number]).iloc[:20]

# JC and SMC function
def jc_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    
    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
    smc = (f11 + f00) / (f11 + f00 + f10 + f01) if (f11 + f00 + f10 + f01) > 0 else 0
    
    return jc, smc

n = 20
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cos_matrix = cosine_similarity(df_numeric.values)

for i in range(n):
    for j in range(n):
        jc_matrix[i, j], smc_matrix[i, j] = jc_smc(binary_20.iloc[i], binary_20.iloc[j])

plt.figure(figsize=(10, 8))
sns.heatmap(jc_matrix, annot=False, cmap='YlGnBu')
plt.title("Jaccard Coefficient Heatmap (First 20 Rows)")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(smc_matrix, annot=False, cmap='OrRd')
plt.title("Simple Matching Coefficient Heatmap (First 20 Rows)")
plt.show()

# Cosine Similarity Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cos_matrix, annot=False, cmap='PuBu')
plt.title("Cosine Similarity Heatmap (First 20 Rows)")
plt.show()
