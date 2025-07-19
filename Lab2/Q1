import pandas as pd
import numpy as np

df_purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
A = df_purchase[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df_purchase[['Payment (Rs)']].values

dimensionality = A.shape[1]  # 3 products
num_vectors = A.shape[0]     # 10 customers
rank = np.linalg.matrix_rank(A)

A_pinv = np.linalg.pinv(A)
X = A_pinv @ C  # Cost per item

print("Dimensionality:", dimensionality)
print("Number of vectors:", num_vectors)
print("Rank of matrix A:", rank)
print("Estimated Costs (Candy, Mango, Milk):", X.flatten())
