"""
A1 – Baseline Perceptron on features.csv
- Loads /mnt/data/features.csv
- Predicts 'genre' using numeric features (drops 'file_path')
- Train/test split with stratification
- Standardize features
- Train Perceptron
- Report accuracy, classification report, confusion matrix
- 5-fold cross-validation on the training set
- Saves model and scaler
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
DATA_PATH = Path("/mnt/data/features.csv")

def load_data(path: Path):
    df = pd.read_csv(path)
    # Drop non-numeric and non-target columns we don't want as features
    X = df.drop(columns=["genre", "file_path"])
    y = df["genre"].astype(str)
    return X, y

def build_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", Perceptron(random_state=RANDOM_STATE, max_iter=1000, tol=1e-3))
    ])
    return pipe

def main():
    print("=== A1 Baseline Perceptron ===")
    X, y = load_data(DATA_PATH)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV accuracy (5-fold): mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
    print("Fold scores:", ", ".join(f"{s:.4f}" for s in cv_scores))

    # Fit on full training
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print("\nConfusion matrix:")
    print(cm_df)

    # Save model artifacts
    from joblib import dump
    out_dir = Path("/mnt/data/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "a1_perceptron.joblib"
    dump(pipe, model_path)
    print(f"\nSaved trained pipeline to: {model_path}")

if __name__ == "__main__":
    main()

