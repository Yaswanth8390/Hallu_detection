"""Train a linear hidden-state probe to predict semantic entropy."""

import argparse
import csv
import json

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_utils import MODEL_NAME


def read_sep_dataset(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    with open(path, newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"No SEP dataset rows found in {path}")
    required = {"hidden_features", "semantic_entropy", "model_name", "hidden_layer"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"SEP dataset is missing columns: {sorted(missing)}")

    model_names = {row["model_name"] for row in rows}
    layers = {int(row["hidden_layer"]) for row in rows}
    if model_names != {MODEL_NAME}:
        raise ValueError(f"SEP features must all be from {MODEL_NAME!r}, got {model_names}")
    if len(layers) != 1:
        raise ValueError("SEP dataset mixes hidden-state layers")

    features = []
    targets = []
    expected_size = None
    for row in rows:
        try:
            vector = np.asarray(json.loads(row["hidden_features"]), dtype=np.float64)
            target = float(row["semantic_entropy"])
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("Invalid hidden_features JSON or semantic_entropy value") from error
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("hidden_features must be a non-empty finite vector")
        if not np.isfinite(target) or target < 0:
            raise ValueError("semantic_entropy must be finite and non-negative")
        if expected_size is None:
            expected_size = vector.size
        elif vector.size != expected_size:
            raise ValueError("SEP dataset has inconsistent hidden feature dimensions")
        features.append(vector)
        targets.append(target)
    return np.vstack(features), np.asarray(targets), next(iter(layers))


def train_probe(data_path: str, model_path: str, test_size: float = 0.2,
                random_state: int = 42, alpha: float = 1.0,
                alert_quantile: float = 0.75):
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 < alert_quantile < 1.0:
        raise ValueError("alert_quantile must be between 0 and 1")
    features, targets, layer = read_sep_dataset(data_path)
    if len(targets) < 5:
        raise ValueError("At least five examples are required for train/validation evaluation")

    indices = np.arange(len(targets))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    validation_scaler = StandardScaler().fit(features[train_indices])
    validation_probe = Ridge(alpha=alpha).fit(
        validation_scaler.transform(features[train_indices]),
        targets[train_indices],
    )
    validation_predictions = validation_probe.predict(
        validation_scaler.transform(features[test_indices])
    )
    print(f"Held-out examples: {len(test_indices)}")
    print(f"Semantic-entropy MAE: {mean_absolute_error(targets[test_indices], validation_predictions):.4f} nats")
    print(f"Semantic-entropy RMSE: {mean_squared_error(targets[test_indices], validation_predictions) ** 0.5:.4f} nats")

    scaler = StandardScaler().fit(features)
    probe = Ridge(alpha=alpha).fit(scaler.transform(features), targets)
    detector = {
        "scaler": scaler,
        "probe": probe,
        "model_name": MODEL_NAME,
        "hidden_layer": layer,
        "hidden_size": features.shape[1],
        "training_examples": len(targets),
        "target": "sampled_response_semantic_entropy_nats",
        "nli_supervision": True,
        "alert_quantile": alert_quantile,
        "alert_threshold": float(np.quantile(targets, alert_quantile)),
    }
    joblib.dump(detector, model_path)
    print(
        f"Saved SEP probe to {model_path} "
        f"(default alert threshold {detector['alert_threshold']:.4f} nats)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="sep_dataset.csv")
    parser.add_argument("--out", default="sep_probe.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alert-quantile", type=float, default=0.75,
                        help="training-target quantile used as default uncertainty threshold")
    args = parser.parse_args()
    try:
        train_probe(args.data, args.out, args.test_size, args.random_state,
                    args.alpha, args.alert_quantile)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
