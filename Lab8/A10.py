import numpy as np

def step(x): return 1 if x>=0 else 0

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_logic = np.array([0,0,0,1])   # AND
    y = np.array([[1,0] if val==0 else [0,1] for val in y_logic])

    W = np.zeros((2,2))   # two outputs
    b = np.zeros(2)
    lr=0.1

    for epoch in range(50):
        for xi, yi in zip(X,y):
            net = xi.dot(W)+b
            out = np.array([step(n) for n in net])
            delta = yi - out
            W += lr*np.outer(xi,delta)
            b += lr*delta

    print("Final Weights:\n",W)
    print("Final Bias:",b)
    for xi in X:
        net = xi.dot(W)+b
        out = [step(n) for n in net]
        print(xi,"→",out)
