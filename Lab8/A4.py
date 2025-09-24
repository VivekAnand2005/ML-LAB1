import numpy as np
import matplotlib.pyplot as plt

def step(x): return 1 if x >= 0 else 0

def perceptron(X, y, w_init, b_init, lr, max_epochs=1000, tol=0.002):
    w = np.array(w_init, dtype=float)
    b = b_init
    for epoch in range(1, max_epochs+1):
        outputs = []
        for xi, yi in zip(X, y):
            out = step(np.dot(xi, w)+b)
            delta = yi - out
            w += lr * delta * xi
            b += lr * delta
            outputs.append(out)
        err = np.sum((y - outputs)**2)/2.0
        if err <= tol:
            return epoch
    return max_epochs

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    w_init = [0.2,-0.75]
    b_init = 10.0
    rates = [0.1*i for i in range(1,11)]
    iters = []
    for lr in rates:
        ep = perceptron(X,y,w_init,b_init,lr)
        iters.append(ep)
        print(f"LR={lr} → Epochs={ep}")
    plt.plot(rates, iters, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.title("A4: Learning Rate Effect (AND Gate)")
    plt.show()
