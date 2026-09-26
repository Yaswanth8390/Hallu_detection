"""Train token scores with answer-level max-pooling logistic regression."""

import argparse
import csv
import json

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
                "semantic_similarity"}
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return rows


def load_human_labels(path: str):
    labels = {}
    with open(path, newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {"example_index", "question", "generated_text", "human_label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Human-label file is missing columns: {sorted(missing)}")
        for row in reader:
            example_id = str(row["example_index"])
            if example_id in labels:
                raise ValueError(f"Duplicate human label for example {example_id!r}")
            raw_label = row["human_label"].strip().casefold()
            if raw_label not in {"supported", "hallucinated", "uncertain", ""}:
                raise ValueError(
                    f"Invalid human_label {row['human_label']!r} for example "
                    f"{example_id!r}; use supported, hallucinated, or uncertain"
                )
            labels[example_id] = {
                "question": row["question"],
                "generated_text": row["generated_text"],
                "label": {
                    "supported": 0,
                    "hallucinated": 1,
                }.get(raw_label),
            }
    return labels


def answer_groups(group_ids, labels):
    """Return row-index groups and one consistent weak label per answer."""
    grouped_rows = {}
    grouped_labels = {}
    for index, (group_id, label) in enumerate(zip(group_ids, labels)):
        group_key = str(group_id)
        if group_key in grouped_labels and grouped_labels[group_key] != label:
            raise ValueError(f"Answer {group_key!r} has inconsistent token labels")
        grouped_labels[group_key] = int(label)
        grouped_rows.setdefault(group_key, []).append(index)
    return (
        list(grouped_rows.values()),
        np.asarray([grouped_labels[key] for key in grouped_rows], dtype=np.int64),
    )


def fit_max_pool_logistic_regression(X: np.ndarray, y: np.ndarray,
                                     groups: list[np.ndarray],
                                     regularization: float = 1.0,
                                     max_iter: int = 300) -> tuple[np.ndarray, float]:
    """Fit token logits while optimizing BCE on each answer's maximum logit.

    This is a linear logistic detector with HARP's multiple-instance objective:
    each answer score is the maximum of its content-token scores. Answer labels
    supervise the pooled score; they are not treated as token annotations.
    """
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both binary answer-label classes")
    if len(groups) != len(y):
        raise ValueError("There must be exactly one token-index group per answer label")

    features = torch.as_tensor(X, dtype=torch.float64)
    labels = torch.as_tensor(y, dtype=torch.float64)
    group_indices = [
        torch.as_tensor(group, dtype=torch.long) for group in groups
    ]
    class_counts = torch.bincount(labels.to(torch.long), minlength=2).to(torch.float64)
    class_weights = len(y) / (2.0 * class_counts)
    sample_weights = class_weights[labels.to(torch.long)]

    weights = torch.nn.Parameter(torch.zeros(X.shape[1], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [weights, bias], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        token_logits = features @ weights + bias
        answer_logits = torch.stack([
            token_logits[index].max() for index in group_indices
        ])
        losses = F.binary_cross_entropy_with_logits(
            answer_logits, labels, reduction="none"
        )
        loss = (losses * sample_weights).mean()
        loss = loss + 0.5 * regularization * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach().numpy(), float(bias.detach().item())


def token_probabilities(detector: dict, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float64)
    standardized = detector["scaler"].transform(X)
    logits = standardized @ detector["weights"] + detector["bias"]
    logits = np.clip(logits, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-logits))


def train_detector(data_path: str, labels_path: str, basis_path: str, model_path: str,
                   test_size: float = 0.2, random_state: int = 42,
                   regularization: float = 1.0, max_iter: int = 300):
    rows = load_training_rows(data_path)
    human_labels = load_human_labels(labels_path)
    feature_answers = {}
    for row in rows:
        example_id = str(row["example_index"])
        answer = (row.get("question", ""), row.get("generated_text", ""))
        if example_id in feature_answers and feature_answers[example_id] != answer:
            raise ValueError(f"Feature rows disagree on answer {example_id!r}")
        feature_answers[example_id] = answer
    unknown_labels = set(human_labels) - set(feature_answers)
    if unknown_labels:
        raise ValueError(
            "Human-label file contains example IDs absent from feature data: "
            f"{sorted(unknown_labels)[:5]}"
        )
    for example_id, human_row in human_labels.items():
        if (human_row["question"], human_row["generated_text"]) != feature_answers[example_id]:
            raise ValueError(
                f"Question/answer text mismatch for human-labeled example {example_id!r}"
            )

    groups_all, _ = answer_groups(
        [row["example_index"] for row in rows],
        [0] * len(rows),
    )
    selected_groups = []
    selected_labels = []
    for row_group, example_id in zip(groups_all, feature_answers):
        label = human_labels.get(str(example_id), {}).get("label")
        if label is None:
            continue
        selected_groups.append(row_group)
        selected_labels.append(label)
    if not selected_groups:
        raise ValueError("No supported/hallucinated human labels match the feature data")

    with open(basis_path, "rb") as basis_file:
        basis_record = torch.load(basis_file, map_location="cpu", weights_only=True)
    if basis_record["model_name"] != MODEL_NAME:
        raise ValueError(
            f"Dataset basis is for {basis_record['model_name']!r}, expected {MODEL_NAME!r}"
        )
    reasoning_basis = basis_record["basis"].numpy()
    all_X = np.vstack([
        feature_vector(row, expected_harp_size=reasoning_basis.shape[1])
        for row in rows
    ])
    selected_row_indices = np.concatenate(selected_groups)
    X = all_X[selected_row_indices]
    rebased_groups = []
    row_offset = 0
    for group in selected_groups:
        rebased_groups.append(np.arange(row_offset, row_offset + len(group)))
        row_offset += len(group)
    answer_labels = np.asarray(selected_labels, dtype=np.int64)
    groups = rebased_groups
    if len(np.unique(answer_labels)) != 2:
        raise ValueError(
            "Human training labels must include both 'supported' and 'hallucinated' answers"
        )
    if min(np.bincount(answer_labels, minlength=2)) < 2:
        raise ValueError("At least two distinct answers per label class are needed for validation")

    train_answers, test_answers = train_test_split(
        np.arange(len(groups)),
        test_size=test_size,
        random_state=random_state,
        stratify=answer_labels,
    )
    train_rows = np.concatenate([groups[index] for index in train_answers])
    test_rows = np.concatenate([groups[index] for index in test_answers])
    scaler = StandardScaler().fit(X[train_rows])
    scaled_X = scaler.transform(X)

    train_groups = [groups[index] for index in train_answers]
    train_labels = answer_labels[train_answers]
    train_X = scaled_X[train_rows]
    local_groups = []
    offset = 0
    for group in train_groups:
        local_groups.append(np.arange(offset, offset + len(group)))
        offset += len(group)
    weights, bias = fit_max_pool_logistic_regression(
        train_X, train_labels, local_groups, regularization, max_iter
    )

    heldout_token_probs = 1.0 / (
        1.0 + np.exp(-np.clip(scaled_X[test_rows] @ weights + bias, -700, 700))
    )
    heldout_group_probs = []
    for answer_index in test_answers:
        heldout_group_probs.append(float(heldout_token_probs[
            np.isin(test_rows, groups[answer_index])
        ].max()))
    test_labels = answer_labels[test_answers]
    test_predictions = (np.asarray(heldout_group_probs) >= 0.5).astype(np.int64)
    print(f"Held-out examples: {len(test_answers)}")
    print(f"Held-out token rows: {len(test_rows)}")
    print(f"Answer-level accuracy: {accuracy_score(test_labels, test_predictions):.3f}")
    print(f"Answer-level F1: {f1_score(test_labels, test_predictions, zero_division=0):.3f}")
    if len(np.unique(test_labels)) == 2:
        print(f"Answer-level ROC-AUC: {roc_auc_score(test_labels, heldout_group_probs):.3f}")
    else:
        print("Answer-level ROC-AUC: unavailable (held-out split has one label class)")

    full_scaler = StandardScaler().fit(X)
    full_X = full_scaler.transform(X)
    full_weights, full_bias = fit_max_pool_logistic_regression(
        full_X, answer_labels, groups, regularization, max_iter
    )
    detector = {
        "scaler": full_scaler,
        "weights": full_weights,
        "bias": full_bias,
        "model_name": MODEL_NAME,
        "reasoning_basis": reasoning_basis,
        "harp_feature_size": reasoning_basis.shape[1],
        "decision_threshold": 0.5,
        "training_objective": "answer_level_max_pool_binary_cross_entropy",
        "training_answers": len(groups),
        "label_source": "human_answer_level",
    }
    joblib.dump(detector, model_path)
    print(f"Human-labeled answers used: {len(groups)}")
    print(f"Unlabeled/uncertain answers excluded: {len(feature_answers) - len(groups)}")
    print(f"Saved max-pooled token detector to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results_tokens.csv")
    parser.add_argument("--labels", required=True,
                        help="human_labels.csv completed with answer-level labels")
    parser.add_argument("--harp-basis", type=str, default="harp_basis.pt")
    parser.add_argument("--out", type=str, default="token_detector.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="L2 penalty strength for the logistic detector")
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    if args.regularization <= 0.0:
        parser.error("--regularization must be positive")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    train_detector(
        args.data, args.labels, args.harp_basis, args.out, args.test_size,
        args.random_state, args.regularization, args.max_iter,
    )


if __name__ == "__main__":
    main()
