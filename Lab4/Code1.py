import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_percentage_error,
    r2_score
)


def load_data(path, bins=3):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number])  # keep numeric columns
    X = df.iloc[:, :-1].values
    y_continuous = df.iloc[:, -1].values

    # Convert continuous target into discrete classes
    y = pd.qcut(y_continuous, q=bins, labels=False)

    return train_test_split(X, y, test_size=0.3, random_state=42), df


def A1(X_train, X_test, y_train, y_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("Train Classification Report:\n", classification_report(y_train, y_pred_train))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("Test Classification Report:\n", classification_report(y_test, y_pred_test))


def A2(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"R² Score: {r2:.2f}")


def A3():
    np.random.seed(0)
    X = np.random.uniform(1, 10, (20, 2))
    y = np.array([0]*10 + [1]*10)
    colors = np.array(['blue', 'red'])
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor='k')
    plt.title("A3: Training Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    return X, y


def A4(X_train, y_train, k=3):
    x_range = np.arange(0, 10.1, 0.1)
    X_test = np.array([[x, y] for x in x_range for y in x_range])
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    colors = np.array(['blue', 'red'])
    plt.figure(figsize=(6, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_pred], s=1)
    plt.title(f"A4: Classified Test Data (k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def A5(X_train, y_train, k_values=[1, 3, 5, 7]):
    for k in k_values:
        A4(X_train, y_train, k)

def A6(df, feature_indices=[0, 1], k_values=[1, 3, 5]):
    # Use numeric columns
    df = df.select_dtypes(include=[np.number])
    X = df.iloc[:, feature_indices].values
    y_continuous = df.iloc[:, -1].values
    y = pd.qcut(y_continuous, q=3, labels=False)

    colors = np.array(['blue', 'red', 'green'])

    # Plot original data
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=colors[y], edgecolor='k')
    plt.title("A6: Project Training Data (2 Features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

    # Ensure meshgrid is valid
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    # Ensure valid range for arange
    if x_max - x_min < 1.0:
        x_max += 1
        x_min -= 1
    if y_max - y_min < 1.0:
        y_max += 1
        y_min -= 1

    # Mesh grid with reduced resolution to save memory
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    X_test_grid = np.c_[xx.ravel(), yy.ravel()]

    if len(X_test_grid) == 0:
        print("⚠️ Skipping A6: Meshgrid resulted in 0 test points.")
        return

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        y_pred_grid = knn.predict(X_test_grid)

        plt.figure(figsize=(6, 6))
        plt.scatter(X_test_grid[:, 0], X_test_grid[:, 1], c=colors[y_pred_grid], s=1)
        plt.title(f"A6: Project Class Boundaries (k={k})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()


    


def A7(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best k value:", grid.best_params_['n_neighbors'])
    print("Best cross-validation score:", grid.best_score_)

# A1
print("---------- A1 ----------")
A1(X_train, X_test, y_train, y_test)

# A2 (Regression example — optional, for Lab02)
# A2(np.array([10, 15, 20]), np.array([11, 14, 21]))

# A3
print("---------- A3 ----------")
X_simple, y_simple = A3()

# A4
print("---------- A4 ----------")
A4(X_simple, y_simple, k=3)

# A5
print("---------- A5 ----------")
A5(X_simple, y_simple, k_values=[1, 3, 5, 7])

# A6
print("---------- A6 ----------")
A6(df, feature_indices=[0, 1], k_values=[1, 3, 5])

# A7
print("---------- A7 ----------")
A7(X_train, y_train)
