import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values

feature_index = 0  # choose first feature
feature_data = X[:, feature_index]

plt.hist(feature_data, bins=10, color='skyblue', edgecolor='black')
plt.title(f'Feature {feature_index} Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

print("Mean:", np.mean(feature_data))
print("Variance:", np.var(feature_data))
