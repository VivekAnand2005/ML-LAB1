import numpy as np
import matplotlib.pyplot as plt

# Create normal distribution data
data = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std=1

# Plot normal distribution curve
count, bins, ignored = plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue')
plt.plot(bins, 1/(1 * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - 0)**2 / (2 * 1**2)), linewidth=2, color='red')
plt.title("Normal Distribution vs Histogram")
plt.show()
