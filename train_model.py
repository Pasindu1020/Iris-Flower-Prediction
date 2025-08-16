
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Optional KaggleHub import (only used if --data-source kaggle is chosen)
def try_import_kagglehub():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        return kagglehub, KaggleDatasetAdapter
    except Exception as e:
        raise RuntimeError(
            "KaggleHub is not installed or could not be imported. "
            "Install with: pip install kagglehub[pandas-datasets]"
        )

def load_data_from_kaggle(file_path: str | None = None) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    kagglehub, KaggleDatasetAdapter = try_import_kagglehub()
    if not file_path:
        # Common file name in uciml/iris
        file_path = "Iris.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/iris",
        file_path,
    )

    # Normalize column names (lowercase, replace spaces with underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Possible column aliases across different Iris variants
    col_aliases = {
        "sepal_length": ["sepal_length", "sepallengthcm", "sepal_length_cm", "sepal_length_(cm)"],
        "sepal_width": ["sepal_width", "sepalwidthcm", "sepal_width_cm", "sepal_width_(cm)"],
        "petal_length": ["petal_length", "petallengthcm", "petal_length_cm", "petal_length_(cm)"],
        "petal_width": ["petal_width", "petalwidthcm", "petal_width_cm", "petal_width_(cm)"],
        "species": ["species", "variety"]
    }

    def find_col(key):
        for candidate in col_aliases[key]:
            if candidate in df.columns:
                return candidate
        raise KeyError(f"Could not find a column for '{key}' in the Kaggle dataset. Columns: {df.columns.tolist()}")

    sl = find_col("sepal_length")
    sw = find_col("sepal_width")
    pl = find_col("petal_length")
    pw = find_col("petal_width")
    sp = find_col("species")

    X = df[[sl, sw, pl, pw]].astype(float)
    y = df[sp].astype(str)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = sorted(y.unique().tolist())  # alphabetical order

    # Encode targets to integers in the order of target_names
    label_to_idx = {label: i for i, label in enumerate(target_names)}
    y_idx = y.map(label_to_idx).to_numpy()

    return X, y_idx, feature_names, target_names

def load_data_from_sklearn() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    iris = load_iris(as_frame=True)
    X = iris.frame[iris.feature_names]
    y = iris.target.to_numpy()
    feature_names = iris.feature_names
    target_names = iris.target_names.tolist()
    return X, y, feature_names, target_names

def main():
    parser = argparse.ArgumentParser(description="Train an Iris classifier and save model artifacts.")
    parser.add_argument("--data-source", choices=["sklearn", "kaggle"], default="sklearn",
                        help="Where to load the dataset from.")
    parser.add_argument("--kaggle-file", type=str, default=None,
                        help="Filename inside the Kaggle dataset (e.g., Iris.csv).")
    parser.add_argument("--model-path", type=str, default="model.pkl", help="Where to save the trained model.")
    parser.add_argument("--meta-path", type=str, default="model_meta.json", help="Where to save metadata.")
    args = parser.parse_args()

    if args.data_source == "kaggle":
        X, y, feature_names, target_names = load_data_from_kaggle(args.kaggle_file)
    else:
        X, y, feature_names, target_names = load_data_from_sklearn()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Simple, strong baseline: Standardize + Logistic Regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Save model
    joblib.dump(pipe, args.model_path)

    # Save metadata (feature order, class names, metrics)
    metadata = {
        "problem_type": "classification",
        "model_type": "Pipeline(StandardScaler -> LogisticRegression)",
        "feature_names": feature_names,
        "target_names": target_names,
        "test_accuracy": acc,
        "classification_report": report,
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {args.model_path}")
    print(f"Saved metadata to {args.meta_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
