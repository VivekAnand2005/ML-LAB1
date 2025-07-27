import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('features.csv')
df = df.select_dtypes(include=[np.number])

X = df.iloc[:, :-1].values

vec1 = X[0]
vec2 = X[1]

distances = []
for r in range(1, 11):
    dist = np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)
    distances.append(dist)

plt.plot(range(1, 11), distances, marker='o')
plt.title('Minkowski Distance (r=1 to 10)')
plt.xlabel('r')
plt.ylabel('Distance')
plt.show()

print("Distances:", distances)
