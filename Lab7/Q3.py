"""
A3 – Classification with Multiple Classifiers
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from joblib import dump

RANDOM_STATE = 42
DATA_PATH = Path("/mnt/data/features.csv")

def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["genre", "file_path"])
    y = df["genre"].astype(str)
    return X, y

def main():
    print("=== A3 Multiple Classifiers ===")
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    classifiers = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[name] = acc

        # Save model
        out_dir = Path("/mnt/data/models")
        out_dir.mkdir(parents=True, exist_ok=True)
        dump(pipe, out_dir / f"a3_{name.lower()}.joblib")

    print("\n=== Summary of Accuracies ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    main()
