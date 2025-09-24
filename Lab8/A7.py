import numpy as np

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
    X_bias = np.hstack([X, np.ones((X.shape[0],1))])

    # pseudo-inverse solution
    W = np.linalg.pinv(X_bias).dot(y)
    preds = (X_bias.dot(W) >= 0.5).astype(int)

    print("Pseudo-inverse Weights:",W)
    print("Predictions:",preds)
    print("Actual:",y)
