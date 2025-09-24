import numpy as np
import matplotlib.pyplot as plt

def step_activation(x):
    return 1 if x >= 0 else 0

def train_perceptron(X, y, w_init, bias_init, lr=0.05, max_epochs=1000, tol=0.002):
    weights = np.array(w_init, dtype=float)
    bias = bias_init
    history = []
    for epoch in range(1, max_epochs+1):
        outputs = []
        for xi, yi in zip(X, y):
            net = np.dot(xi, weights) + bias
            out = step_activation(net)
            delta = yi - out
            weights += lr * delta * xi
            bias += lr * delta
            outputs.append(out)
        err = np.sum((y - outputs) ** 2) / 2.0
        history.append(err)
        if err <= tol:
            break
    return weights, bias, history, epoch

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    w_init = [0.2, -0.75]
    bias_init = 10.0
    weights, bias, history, epochs = train_perceptron(X, y, w_init, bias_init)
    print("Final Weights:", weights)
    print("Final Bias:", bias)
    print("Epochs:", epochs)
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("AND Gate Training")
    plt.show()
