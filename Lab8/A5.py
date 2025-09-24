import numpy as np
import matplotlib.pyplot as plt

def step(x): return 1 if x >= 0 else 0

def train_perceptron(X, y, w_init, b_init, lr=0.05, max_epochs=1000):
    w = np.array(w_init, dtype=float)
    b = b_init
    history = []
    for epoch in range(1,max_epochs+1):
        outputs = []
        for xi, yi in zip(X, y):
            out = step(np.dot(xi,w)+b)
            delta = yi - out
            w += lr * delta * xi
            b += lr * delta
            outputs.append(out)
        err = np.sum((y - outputs)**2)/2.0
        history.append(err)
    return w,b,history,epoch

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    w,b,hist,epochs = train_perceptron(X,y,[0.2,-0.75],10.0)
    print("Final Weights:",w,"Bias:",b)
    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("A5: XOR with Single-layer Perceptron (fails to converge)")
    plt.show()
