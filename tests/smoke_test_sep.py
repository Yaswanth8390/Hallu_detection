"""Smoke-test semantic-entropy targets, response representations, and SEP training."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from model_utils import MODEL_NAME
from semantic_entropy import (
    extract_response_representation,
    semantic_entropy,
    semantic_entropy_from_clusters,
)
from train_sep_probe import train_probe


responses = ["Paris", "The answer is Paris.", "Rome", "Rome"]
entailment = np.eye(4, dtype=bool)
entailment[0, 1] = entailment[1, 0] = True
entropy, cluster_count, cluster_ids = semantic_entropy(
    responses, [-1.0, -1.0, -1.0, -1.0], entailment
)
assert cluster_count == 2
assert cluster_ids == [0, 0, 1, 1]
assert np.isclose(entropy, np.log(2.0))
assert np.isclose(semantic_entropy_from_clusters([0.0, 0.0], [0, 0]), 0.0)

torch.manual_seed(0)
config = Qwen2Config(
    vocab_size=1000,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
model = AutoModelForCausalLM.from_config(config).eval()
representation = extract_response_representation(
    model, torch.tensor([10, 11]), torch.tensor([12, 13]), layer=-1
)
assert representation.shape == (32,)
assert np.isfinite(representation).all()

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    data_path = directory / "sep_dataset.csv"
    probe_path = directory / "sep_probe.joblib"
    rows = []
    rng = np.random.default_rng(0)
    for index in range(20):
        target = float(index % 5) / 2.0
        features = (np.full(8, target) + rng.normal(0, 0.01, 8)).tolist()
        rows.append({
            "hidden_features": json.dumps(features),
            "semantic_entropy": target,
            "model_name": MODEL_NAME,
            "hidden_layer": -1,
        })
    with data_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_probe(str(data_path), str(probe_path), test_size=0.25, random_state=0)
    assert probe_path.is_file()

print("Semantic Entropy Probe smoke test passed.")
