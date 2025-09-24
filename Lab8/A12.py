import numpy as np
from sklearn.neural_network import MLPClassifier

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

    clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, activation='logistic', random_state=1)
    clf.fit(X,y)
    preds = clf.predict(X)
    print("Predictions:",preds)
    print("Actual:",y)
