import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def train_logistic(X, y, lr=0.5, max_epochs=10000, tol=0.002):
    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    b = 0.0
    history = []
    for epoch in range(1, max_epochs+1):
        z = X.dot(W) + b
        y_hat = sigmoid(z)
        loss = np.mean((y - y_hat)**2) / 2.0
        history.append(loss)

        grad = (y_hat - y) * sigmoid_derivative(y_hat)
        W -= lr * (X.T.dot(grad) / n_samples)
        b -= lr * np.mean(grad)

        if loss <= tol:
            break
    return W, b, history, epoch

if __name__ == "__main__":
    data = np.array([
        [20,6,2,386,1],
        [16,3,6,289,1],
        [27,6,2,393,1],
        [19,1,2,110,0],
        [24,4,2,280,1],
        [22,1,5,167,0],
        [15,4,2,271,1],
        [18,4,2,274,1],
        [21,1,4,148,0],
        [16,2,4,198,0]
    ])
    X = data[:,:4]
    y = data[:,4]

    # normalize
    X = (X - X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))

    W,b,hist,ep = train_logistic(X,y)
    print("Weights:",W,"Bias:",b,"Epochs:",ep)

    preds = (sigmoid(X.dot(W)+b) >= 0.5).astype(int)
    print("Predictions:",preds)
    print("Actual:",y)

    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("A6: Customer Data Training (Sigmoid)")
    plt.show()
