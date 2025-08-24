"""
A4 – Regression Models on features.csv
Predicts 'tempo' from audio features
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from joblib import dump

RANDOM_STATE = 42
DATA_PATH = Path("/mnt/data/features.csv")

def load_data(path: Path, target_col="tempo"):
    df = pd.read_csv(path)
    X = df.drop(columns=["file_path", "genre", target_col])
    y = df[target_col]
    return X, y

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse

def main():
    print("=== A4 Regression Models ===")
    X, y = load_data(DATA_PATH, target_col="tempo")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=RANDOM_STATE),
        "XGBRegressor": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)
    }

    results = {}

    for name, reg in regressors.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", reg)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2, rmse = evaluate(y_test, y_pred)
        print(f"{name} → R²={r2:.4f}, RMSE={rmse:.4f}")

        results[name] = (r2, rmse)

        # Save model
        out_dir = Path("/mnt/data/models")
        out_dir.mkdir(parents=True, exist_ok=True)
        dump(pipe, out_dir / f"a4_{name.lower()}.joblib")

    print("\n=== Summary of Regression Results ===")
    for name, (r2, rmse) in results.items():
        print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
