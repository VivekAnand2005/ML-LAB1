import numpy as np
import matplotlib.pyplot as plt

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 if (1 / (1 + np.exp(-x))) >= 0.5 else 0

def relu(x):
    return 1 if x > 0 else 0

def train_perceptron(X, y, activation, w_init, bias_init, lr=0.05, max_epochs=1000, tol=0.002):
    weights = np.array(w_init, dtype=float)
    bias = bias_init
    history = []

    for epoch in range(1, max_epochs+1):
        outputs = []
        for xi, yi in zip(X, y):
            net = np.dot(xi, weights) + bias
            if activation == "bipolar":
                out = bipolar_step(net)
            elif activation == "sigmoid":
                out = sigmoid(net)
            elif activation == "relu":
                out = relu(net)
            else:
                raise ValueError("Invalid activation")
            delta = yi - out
            weights += lr * delta * xi
            bias += lr * delta
            outputs.append(out)
        err = np.sum((y - outputs)**2)/2.0
        history.append(err)
        if err <= tol:
            break
    return weights, bias, history, epoch

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    activations = ["bipolar", "sigmoid", "relu"]
    for act in activations:
        w, b, hist, ep = train_perceptron(X, y, act, [0.2,-0.75], 10.0)
        print(f"{act} → Weights:{w}, Bias:{b}, Epochs:{ep}")
        plt.plot(hist, label=act)
    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("A3: AND Gate with Different Activations")
    plt.legend()
    plt.show()
