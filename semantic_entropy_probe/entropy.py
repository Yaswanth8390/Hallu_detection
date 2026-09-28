"""Semantic-entropy targets from sampled answers and bidirectional NLI."""

import math
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def cluster_responses(responses: list[str], entailment_matrix: np.ndarray) -> list[int]:
    """Cluster responses by mutual entailment, including exact text duplicates."""
    count = len(responses)
    if entailment_matrix.shape != (count, count):
        raise ValueError("The entailment matrix must have one row and column per response")

    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    normalized = [re.sub(r"\s+", " ", text).strip().casefold() for text in responses]
    for left in range(count):
        for right in range(left + 1, count):
            if normalized[left] == normalized[right] or (
                entailment_matrix[left, right] and entailment_matrix[right, left]
            ):
                union(left, right)

    cluster_ids = {}
    return [
        cluster_ids.setdefault(root, len(cluster_ids))
        for root in (find(index) for index in range(count))
    ]


def semantic_entropy_from_clusters(log_probabilities: list[float],
                                   cluster_ids: list[int]) -> float:
    """Estimate semantic entropy in nats from sampled sequence log-likelihoods."""
    if len(log_probabilities) != len(cluster_ids) or not log_probabilities:
        raise ValueError("At least one log-probability and matching cluster ID are required")
    values = np.asarray(log_probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Sequence log-probabilities must be finite")

    log_masses = []
    for cluster in dict.fromkeys(cluster_ids):
        cluster_values = values[np.asarray(cluster_ids) == cluster]
        maximum = float(cluster_values.max())
        log_masses.append(maximum + math.log(float(np.exp(cluster_values - maximum).sum())))

    log_masses = np.asarray(log_masses)
    maximum = float(log_masses.max())
    log_normalizer = maximum + math.log(float(np.exp(log_masses - maximum).sum()))
    probabilities = np.exp(log_masses - log_normalizer)
    return float(-(probabilities * np.log(probabilities)).sum())


def semantic_entropy(responses: list[str], log_probabilities: list[float],
                     entailment_matrix: np.ndarray) -> tuple[float, int, list[int]]:
    clusters = cluster_responses(responses, entailment_matrix)
    return (
        semantic_entropy_from_clusters(log_probabilities, clusters),
        len(set(clusters)),
        clusters,
    )


class NLIEntailment:
    """Batched NLI classifier used to identify mutually entailing answers."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        labels = {
            int(index): str(label).casefold()
            for index, label in self.model.config.id2label.items()
        }
        entailment_indices = [
            index for index, label in labels.items() if "entail" in label
        ]
        if len(entailment_indices) == 1:
            self.entailment_index = entailment_indices[0]
        elif len(labels) == 3 and set(labels) == {0, 1, 2}:
            self.entailment_index = 2
        else:
            raise ValueError(
                f"Cannot determine entailment label from NLI model labels: {labels}"
            )

    @torch.no_grad()
    def matrix(self, responses: list[str], threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("NLI entailment threshold must be between 0 and 1")
        count = len(responses)
        matrix = np.eye(count, dtype=bool)
        pairs = [
            (responses[left], responses[right], left, right)
            for left in range(count)
            for right in range(count)
            if left != right
        ]
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset + self.batch_size]
            encoded = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            probabilities = self.model(**encoded).logits.softmax(dim=-1)
            scores = probabilities[:, self.entailment_index].cpu().numpy()
            for (_, _, left, right), score in zip(batch, scores):
                matrix[left, right] = score >= threshold
        return matrix


def extract_response_representation(model, prompt_ids: torch.Tensor,
                                    generated_ids: torch.Tensor,
                                    layer: int = -1) -> np.ndarray:
    """Mean-pool causal hidden states that predict generated response tokens."""
    if generated_ids.numel() == 0:
        raise ValueError("Cannot extract a response representation for an empty answer")
    device = model.get_input_embeddings().weight.device
    prompt_ids = prompt_ids.reshape(-1)
    generated_ids = generated_ids.reshape(-1)
    full_ids = torch.cat((prompt_ids, generated_ids)).to(device).unsqueeze(0)
    base_model = (
        model.get_base_model()
        if hasattr(model, "get_base_model")
        else model.base_model
    )
    with torch.no_grad():
        outputs = base_model(
            input_ids=full_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("The model did not return hidden states")
        if not -len(hidden_states) <= layer < len(hidden_states):
            raise ValueError(
                f"Hidden-state layer {layer} is out of range for "
                f"{len(hidden_states)} available states"
            )
        hidden = hidden_states[layer][0].float()
        positions = torch.arange(
            prompt_ids.numel() - 1,
            prompt_ids.numel() + generated_ids.numel() - 1,
            device=hidden.device,
        )
        return hidden[positions].mean(dim=0).cpu().numpy().astype(np.float64)
