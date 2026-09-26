"""Smoke test the answer-level max-pooled logistic training objective."""

import csv
import json
import joblib
import numpy as np
import tempfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch

from model_utils import MODEL_NAME
from prepare_annotations import export_annotation_template
from train_detector import (
    answer_groups,
    fit_max_pool_logistic_regression,
    train_detector,
    token_probabilities,
)


features = np.asarray([
    [-2.0, 0.0], [-1.0, 0.2],  # supported answer
    [0.0, 2.0], [0.2, 3.0],    # hallucinated answer
    [-1.5, 0.1],               # supported answer
    [0.1, 2.5], [0.2, 2.0],   # hallucinated answer
], dtype=np.float64)
labels = np.asarray([0, 0, 1, 1, 0, 1, 1])
groups, answer_labels = answer_groups(
    ["a", "a", "b", "b", "c", "d", "d"], labels
)

assert len(groups) == 4
assert answer_labels.tolist() == [0, 1, 0, 1]
scaler = StandardScaler().fit(features)
scaled_features = scaler.transform(features)
weights, bias = fit_max_pool_logistic_regression(
    scaled_features, answer_labels, groups, regularization=0.1, max_iter=100
)

assert weights.shape == (2,)
assert np.isfinite(weights).all()
assert np.isfinite(bias)

detector = {
    "scaler": scaler,
    "weights": weights,
    "bias": bias,
}
probabilities = token_probabilities(detector, features)
assert probabilities.shape == (len(features),)
assert np.isfinite(probabilities).all()
assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
assert token_probabilities(detector, np.asarray([])).size == 0
answer_scores = [probabilities[group].max() for group in groups]
assert answer_scores[1] > answer_scores[0]
assert answer_scores[3] > answer_scores[2]

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    data_path = directory / "tokens.csv"
    labels_path = directory / "human_labels.csv"
    basis_path = directory / "basis.pt"
    detector_path = directory / "detector.joblib"
    rows = []
    for answer_index in range(12):
        label = answer_index % 2
        for token_index in range(2):
            rows.append({
                "example_index": answer_index,
                "question": f"Question {answer_index}?",
                "generated_text": f"Generated response {answer_index}.",
                "token": f"token-{token_index}",
                "counterfactual_score": 1.0 if label else -1.0,
                "semantic_similarity": 0.2 if label else 0.8,
                "HARP_features": json.dumps([1.0 if label else -1.0, token_index * 0.1]),
                "label": label,
            })
    with data_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert export_annotation_template(str(data_path), str(labels_path)) == 12
    with labels_path.open(newline="", encoding="utf-8") as input_file:
        annotation_rows = list(csv.DictReader(input_file))
    for row in annotation_rows:
        row["human_label"] = (
            "hallucinated" if int(row["example_index"]) % 2 else "supported"
        )
    with labels_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(annotation_rows[0]))
        writer.writeheader()
        writer.writerows(annotation_rows)
    torch.save({
        "model_name": MODEL_NAME,
        "basis": torch.eye(2, dtype=torch.float64),
    }, basis_path)
    train_detector(
        str(data_path), str(labels_path), str(basis_path), str(detector_path),
        test_size=0.25, random_state=42, regularization=0.1, max_iter=100,
    )
    assert detector_path.is_file()
    saved_detector = joblib.load(detector_path)
    assert saved_detector["label_source"] == "human_answer_level"

print("Answer-level max-pooled logistic training smoke test passed.")
