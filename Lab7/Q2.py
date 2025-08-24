"""
A2 – RandomizedSearchCV Hyperparameter Tuning for Perceptron
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

RANDOM_STATE = 42
DATA_PATH = Path("/mnt/data/features.csv")

def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["genre", "file_path"])
    y = df["genre"].astype(str)
    return X, y

def main():
    print("=== A2 RandomizedSearchCV for Perceptron ===")
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", Perceptron(random_state=RANDOM_STATE))
    ])

    # Hyperparameter space
    param_dist = {
        "clf__penalty": ["l2", "l1", "elasticnet"],
        "clf__alpha": np.logspace(-5, -1, 10),
        "clf__max_iter": [500, 1000, 1500, 2000],
        "clf__eta0": [0.001, 0.01, 0.1, 1],
        "clf__fit_intercept": [True, False]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, 
        n_iter=20, scoring="accuracy", 
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=2
    )

    search.fit(X_train, y_train)

    print(f"Best CV Score: {search.best_score_:.4f}")
    print("Best Params:", search.best_params_)

    # Evaluate on test set
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print("\nConfusion Matrix:")
    print(cm_df)

    # Save model
    from joblib import dump
    out_dir = Path("/mnt/data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "a2_perceptron_randomsearch.joblib"
    dump(search.best_estimator_, model_path)
    print(f"\nSaved best tuned model to: {model_path}")

if __name__ == "__main__":
    main()
