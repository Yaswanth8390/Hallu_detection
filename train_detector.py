"""Train a token-level Logistic Regression detector on generated feature rows."""

import argparse
import csv
import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import MODEL_NAME


def feature_vector(row: dict, expected_harp_size: int | None = None) -> np.ndarray:
    try:
        harp_features = np.asarray(json.loads(row["HARP_features"]), dtype=np.float64)
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("Each row must contain a JSON HARP_features vector") from error
    if harp_features.ndim != 1 or not np.isfinite(harp_features).all():
        raise ValueError("HARP_features must be a finite one-dimensional vector")
    if expected_harp_size is not None and harp_features.size != expected_harp_size:
        raise ValueError(
            f"Expected {expected_harp_size} HARP features, got {harp_features.size}"
        )

    values = []
    missing = []
    for column in ("counterfactual_score", "semantic_similarity"):
        raw_value = row.get(column, "")
        try:
            value = float(raw_value) if raw_value not in (None, "") else np.nan
        except ValueError as error:
            raise ValueError(f"Invalid numeric value for {column}: {raw_value!r}") from error
        if not np.isnan(value) and not np.isfinite(value):
            raise ValueError(f"{column} must be finite or empty")
        missing.append(float(np.isnan(value)))
        values.append(0.0 if np.isnan(value) else value)
    return np.concatenate((np.asarray(values + missing), harp_features))


def load_training_rows(path: str):
    with open(path, newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    required = {"example_index", "HARP_features", "counterfactual_score",
                "semantic_similarity", "label"}
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return rows


def train_detector(data_path: str, basis_path: str, model_path: str,
                   test_size: float = 0.2, random_state: int = 42):
    rows = load_training_rows(data_path)
    with open(basis_path, "rb") as basis_file:
        basis_record = torch.load(basis_file, map_location="cpu", weights_only=True)
    if basis_record["model_name"] != MODEL_NAME:
        raise ValueError(
            f"Dataset basis is for {basis_record['model_name']!r}, expected {MODEL_NAME!r}"
        )
    reasoning_basis = basis_record["basis"].numpy()
    X = np.vstack([
        feature_vector(row, expected_harp_size=reasoning_basis.shape[1])
        for row in rows
    ])
    try:
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    except ValueError as error:
        raise ValueError("Labels must be integers 0 (supported) or 1 (hallucinated)") from error
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Training requires both binary label classes, encoded as 0 and 1")

    groups = np.asarray([row["example_index"] for row in rows])
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct examples are needed for group-held-out validation")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, test_indices = next(splitter.split(X, y, groups))
    if len(np.unique(y[train_indices])) < 2:
        raise ValueError("The training split contains only one label class; add more examples")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
    )
    classifier.fit(X[train_indices], y[train_indices])
    test_probabilities = classifier.predict_proba(X[test_indices])[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(np.int64)
    print(f"Held-out examples: {len(np.unique(groups[test_indices]))}")
    print(f"Held-out token rows: {len(test_indices)}")
    print(f"Accuracy: {accuracy_score(y[test_indices], test_predictions):.3f}")
    print(f"F1: {f1_score(y[test_indices], test_predictions, zero_division=0):.3f}")
    if len(np.unique(y[test_indices])) == 2:
        print(f"ROC-AUC: {roc_auc_score(y[test_indices], test_probabilities):.3f}")
    else:
        print("ROC-AUC: unavailable (held-out split has one label class)")

    classifier.fit(X, y)
    joblib.dump({
        "classifier": classifier,
        "model_name": MODEL_NAME,
        "reasoning_basis": reasoning_basis,
        "harp_feature_size": reasoning_basis.shape[1],
        "decision_threshold": 0.5,
    }, model_path)
    print(f"Saved token-level detector to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results_tokens.csv")
    parser.add_argument("--harp-basis", type=str, default="harp_basis.pt")
    parser.add_argument("--out", type=str, default="token_detector.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    train_detector(args.data, args.harp_basis, args.out, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
