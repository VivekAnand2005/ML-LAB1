import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(y): return y*(1-y)

def train_mlp(X, y, hidden=2, lr=0.05, max_epochs=1000, tol=0.002):
    n_samples, n_features = X.shape
    np.random.seed(1)
    W1 = np.random.randn(n_features, hidden)
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1)
    b2 = np.zeros((1,1))
    history = []

    for epoch in range(1, max_epochs+1):
        z1 = X.dot(W1)+b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2)+b2
        a2 = sigmoid(z2)

        loss = np.mean((y - a2)**2)/2.0
        history.append(loss)
        if loss <= tol:
            break

        dz2 = (a2 - y) * sigmoid_deriv(a2)
        dW2 = a1.T.dot(dz2)/n_samples
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dz1 = dz2.dot(W2.T) * sigmoid_deriv(a1)
        dW1 = X.T.dot(dz1)/n_samples
        db1 = np.mean(dz1, axis=0, keepdims=True)

        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2

    return W1,b1,W2,b2,history,epoch

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    W1,b1,W2,b2,hist,ep = train_mlp(X,y)
    print("Epochs:",ep,"Final Loss:",hist[-1])

    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("A9: XOR Gate with Backpropagation MLP")
    plt.show()
