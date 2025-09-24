import numpy as np

# -----------------------------
# A1: Basic Perceptron Modules
# -----------------------------

# Summation unit
def summation(inputs, weights, bias):
    """Compute weighted sum of inputs + bias"""
    return np.dot(inputs, weights) + bias

# Activation functions
def step_activation(x):
    return np.where(x >= 0, 1, 0)

def bipolar_step_activation(x):
    return np.where(x >= 0, 1, -1)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

def leaky_relu_activation(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Comparator (Error calculation)
def compute_error(y_true, y_pred):
    """Return difference between true and predicted"""
    return y_true - y_pred
