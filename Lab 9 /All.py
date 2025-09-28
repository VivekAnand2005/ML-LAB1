# lab09_all_codes.py
# Single-file, modular implementation for:
# A1. Stacking classifier/regressor
# A2. Pipeline with preprocessing + model
# A3. LIME explanation for pipeline outputs
# Author: generated for user's Lab09 assignment
# Requirements: scikit-learn, pandas, numpy, lime

import os
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# classifiers/regressors and stacking
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# LIME
from lime.lime_tabular import LimeTabularExplainer

# -------------------------
# 1) Data loading function
# -------------------------
def load_csv_data(path: str, target_column: str, drop_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV and return X (DataFrame) and y (Series).
    Args:
        path: path to CSV file
        target_column: name of target column
        drop_columns: list of columns to drop from dataframe (optional)
    Returns:
        X, y
    """
    df = pd.read_csv(path)
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in data columns: {df.columns.tolist()}")
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])
    return X, y

# ------------------------------------------
# 2) Build preprocessing pipeline (ColumnTransformer)
# ------------------------------------------
def build_preprocessor(X: pd.DataFrame,
                       numeric_impute_strategy: str = "mean",
                       categorical_impute_strategy: str = "most_frequent",
                       numeric_scaler: Optional[Any] = StandardScaler()) -> ColumnTransformer:
    """
    Create a ColumnTransformer preprocessing for numeric and categorical features.
    Args:
        X: input dataframe (used to find numeric/categorical columns)
        numeric_impute_strategy: imputer strategy for numeric
        categorical_impute_strategy: imputer strategy for categorical
        numeric_scaler: scaler instance for numeric features (or None)
    Returns:
        ColumnTransformer
    """
    # Identify dtypes
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Numeric pipeline
    numeric_steps = []
    numeric_steps.append(('num_imputer', SimpleImputer(strategy=numeric_impute_strategy)))
    if numeric_scaler is not None:
        numeric_steps.append(('scaler', numeric_scaler))

    # Categorical pipeline
    categorical_steps = [
        ('cat_imputer', SimpleImputer(strategy=categorical_impute_strategy, fill_value="missing")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]

    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline(steps=numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline(steps=categorical_steps), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    return preprocessor

# ------------------------------------------
# 3) Build stacking model for classification
# ------------------------------------------
def build_stacking_classifier(preprocessor: ColumnTransformer,
                              base_estimators: Optional[List[Tuple[str, Any]]] = None,
                              final_estimator: Optional[Any] = None,
                              cv: int = 5) -> Pipeline:
    """
    Build a pipeline with preprocessing and a StackingClassifier.
    Args:
        preprocessor: ColumnTransformer
        base_estimators: list of (name, estimator) tuples
        final_estimator: meta-estimator for stacking
        cv: cross-validation folds used internally by stacking
    Returns:
        Pipeline object containing preprocessor and stacking classifier
    """
    if base_estimators is None:
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000))
        ]
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000)

    stacking_clf = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator, cv=cv, n_jobs=-1, passthrough=False)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stack', stacking_clf)
    ])
    return pipeline

# ------------------------------------------
# 4) Build stacking model for regression
# ------------------------------------------
def build_stacking_regressor(preprocessor: ColumnTransformer,
                             base_estimators: Optional[List[Tuple[str, Any]]] = None,
                             final_estimator: Optional[Any] = None,
                             cv: int = 5) -> Pipeline:
    """
    Build a pipeline with preprocessing and a StackingRegressor.
    Args:
        preprocessor: ColumnTransformer
        base_estimators: list of (name, estimator) tuples
        final_estimator: meta-estimator for stacking (regressor)
        cv: cross-validation folds
    Returns:
        Pipeline with preprocessing and stacking regressor
    """
    if base_estimators is None:
        base_estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('lr', LinearRegression())
        ]
    if final_estimator is None:
        final_estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)

    stacking_reg = StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, cv=cv, n_jobs=-1, passthrough=False)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stack', stacking_reg)
    ])
    return pipeline

# ------------------------------------------
# 5) Model training and evaluation utilities
# ------------------------------------------
def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Fit the pipeline on training data and return fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_classification(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a classification pipeline and return metrics.
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": conf_mat, "y_pred": y_pred}

def evaluate_regression(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a regression pipeline and return metrics.
    """
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "y_pred": y_pred}

# ------------------------------------------
# 6) LIME explanation helper
# ------------------------------------------
def explain_with_lime(pipeline: Pipeline,
                      X_train: pd.DataFrame,
                      X_explain: pd.DataFrame,
                      feature_names: Optional[List[str]] = None,
                      class_names: Optional[List[str]] = None,
                      task: str = "classification",
                      num_features: int = 10,
                      explanation_save_path: Optional[str] = "lime_explanation.html") -> Dict[str, Any]:
    """
    Generate LIME explanations for given instances.
    Args:
        pipeline: fitted pipeline including preprocessor and final estimator
        X_train: training features DataFrame (used by LimeTabularExplainer)
        X_explain: DataFrame of instances to explain (one or more rows)
        feature_names: optional list of feature names (defaults to X_train.columns)
        class_names: for classification, list of class names (strings)
        task: "classification" or "regression"
        num_features: how many features to show in explanation
        explanation_save_path: path to save HTML explanation
    Returns:
        dict containing explanation objects and path to saved HTML
    Notes:
        - LIME expects numpy arrays for training data; we pass the preprocessed data's pipeline 'preprocessor'
          by exposing the raw training data to the LimeTabularExplainer and then calling pipeline.predict / predict_proba.
    """
    # Prepare data for explainer (LIME requires raw training data values)
    X_train_np = X_train.to_numpy()
    feature_names_local = feature_names if feature_names is not None else X_train.columns.tolist()

    mode = 'classification' if task == 'classification' else 'regression'
    explainer = LimeTabularExplainer(X_train_np, feature_names=feature_names_local, class_names=class_names,
                                     mode=mode, discretize_continuous=True)

    explanations = []
    # Predict wrapper depending on pipeline API
    for idx in range(X_explain.shape[0]):
        instance = X_explain.iloc[idx].to_numpy()
        # LIME expects 2d array
        if task == "classification":
            # For classification LIME asks for predict_proba
            predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=X_train.columns))
            exp = explainer.explain_instance(instance, predict_fn, num_features=num_features)
        else:
            # Regression -> predict function
            predict_fn = lambda x: pipeline.predict(pd.DataFrame(x, columns=X_train.columns)).reshape(-1, 1)
            exp = explainer.explain_instance(instance, predict_fn, num_features=num_features)
        explanations.append(exp)

    # Save HTML for the first explanation (if available)
    if explanations:
        with open(explanation_save_path, "w", encoding="utf-8") as f:
            f.write(explanations[0].as_html())
    return {"explanations": explanations, "html_path": os.path.abspath(explanation_save_path)}

# ------------------------------------------
# 7) Utility: cross-validate stacking pipeline (optional)
# ------------------------------------------
def cross_validate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: Optional[str] = None) -> np.ndarray:
    """
    Run cross-validation on pipeline and return scores array.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores

# ------------------------------------------
# Main: example usage (prints allowed here)
# ------------------------------------------
if __name__ == "__main__":
    # USER ACTION: Edit these variables to match your dataset
    dataset_path = "your_dataset.csv"       # <- replace with your dataset file path
    target_column = "target"               # <- replace with your dataset's target column name
    drop_columns = None                    # <- optionally drop ID or irrelevant cols e.g. ["Row ID"]
    task = "classification"                # "classification" or "regression"

    # Load data
    X, y = load_csv_data(dataset_path, target_column, drop_columns=drop_columns)

    # Example: for classification ensure y is categorical / labels
    if task == "classification":
        # Optional: encode classes if needed (sklearn handles string labels)
        pass

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task=="classification" else None)

    # Build preprocessor using training data to detect types
    preprocessor = build_preprocessor(X_train)

    # Build pipeline according to task
    if task == "classification":
        pipeline = build_stacking_classifier(preprocessor=preprocessor)
    else:
        pipeline = build_stacking_regressor(preprocessor=preprocessor)

    # Train
    pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate
    if task == "classification":
        results = evaluate_classification(pipeline, X_test, y_test)
        print("Classification Accuracy:", results["accuracy"])
        print("Classification Report:")
        print(pd.DataFrame(results["report"]).transpose())
        print("Confusion Matrix:")
        print(results["confusion_matrix"])
    else:
        results = evaluate_regression(pipeline, X_test, y_test)
        print("Regression R2:", results["r2"])
        print("Regression RMSE:", results["rmse"])
        print("Regression MSE:", results["mse"])
        print("Regression MAE:", results["mae"])

    # LIME explanation: explain first 3 rows of test set (or fewer)
    explain_instances = X_test.iloc[:3]
    if task == "classification":
        # If classes are numeric/strings, prepare class_names
        unique_classes = [str(c) for c in np.unique(y_train)]
        lime_out = explain_with_lime(pipeline, X_train, explain_instances, feature_names=X_train.columns.tolist(),
                                    class_names=unique_classes, task="classification", num_features=8,
                                    explanation_save_path="lime_classification_explanation.html")
    else:
        lime_out = explain_with_lime(pipeline, X_train, explain_instances, feature_names=X_train.columns.tolist(),
                                    class_names=None, task="regression", num_features=8,
                                    explanation_save_path="lime_regression_explanation.html")

    print("LIME explanation saved to:", lime_out["html_path"])
    print("Done.")
