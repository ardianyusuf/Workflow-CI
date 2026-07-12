"""Train a RandomForest model on the preprocessed Titanic dataset.

Uses MLflow autolog for tracking and persists the trained model as model.pkl
at the repository root (Kriteria 3 Skilled: save artifact to repository).
"""
import os
import argparse

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="titanic_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    # Resolve paths relative to this file (robust regardless of cwd)
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, args.data_dir)

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="titanic_rf"):
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))

    # Persist model artifact to repository root (Skilled requirement)
    repo_root = os.path.dirname(here)
    model_path = os.path.join(repo_root, "model.pkl")
    joblib.dump(model, model_path)
    print(f"[OK] Model artifact saved to {model_path}")


if __name__ == "__main__":
    main()
