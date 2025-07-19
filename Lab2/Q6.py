import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load thyroid dataset
df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Select only numeric columns for cosine similarity
df_numeric = df_thyroid.select_dtypes(include=[np.number])

# Extract the first 2 rows as vectors
vec1 = df_numeric.iloc[0].values.reshape(1, -1)
vec2 = df_numeric.iloc[1].values.reshape(1, -1)

# Compute cosine similarity
cos_sim = cosine_similarity(vec1, vec2)[0][0]

# Step 5: Output
print("Cosine Similarity between first two observations:", cos_sim)
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load thyroid dataset
df_thyroid = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Select only numeric columns for cosine similarity
df_numeric = df_thyroid.select_dtypes(include=[np.number])

# Extract the first 2 rows as vectors
vec1 = df_numeric.iloc[0].values.reshape(1, -1)
vec2 = df_numeric.iloc[1].values.reshape(1, -1)

# Compute cosine similarity
cos_sim = cosine_similarity(vec1, vec2)[0][0]

# Step 5: Output
print("Cosine Similarity between first two observations:", cos_sim)
