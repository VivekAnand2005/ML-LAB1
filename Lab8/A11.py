import numpy as np
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    y_xor = np.array([0,1,1,0])

    clf_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, activation='logistic', random_state=1)
    clf_and.fit(X,y_and)
    print("MLP AND predictions:",clf_and.predict(X))

    clf_xor = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, activation='logistic', random_state=1)
    clf_xor.fit(X,y_xor)
    print("MLP XOR predictions:",clf_xor.predict(X))
