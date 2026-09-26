"""Reasoning-subspace projection features from the HARP formulation."""

import torch


def _unembedding_weight(model) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or output_embeddings.weight.ndim != 2:
        raise ValueError("The model must expose a two-dimensional output embedding weight")
    return output_embeddings.weight


@torch.no_grad()
def build_reasoning_basis(model, semantic_fraction: float = 0.95,
                          row_chunk_size: int = 4096) -> torch.Tensor:
    """Return V_R, the low-singular-value right-singular-vector basis.

    HARP sets k to 95% of hidden size and uses the remaining right-singular
    vectors of the unembedding matrix as the reasoning subspace.
    """
    if not 0.0 < semantic_fraction < 1.0:
        raise ValueError("semantic_fraction must be between 0 and 1")
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")

    weight = _unembedding_weight(model)
    hidden_size = weight.shape[1]
    semantic_rank = int(hidden_size * semantic_fraction)
    reasoning_size = hidden_size - semantic_rank
    if semantic_rank < 1 or reasoning_size < 1:
        raise ValueError("The semantic and reasoning subspaces must both be non-empty")

    gram = torch.zeros((hidden_size, hidden_size), dtype=torch.float64, device="cpu")
    for start in range(0, weight.shape[0], row_chunk_size):
        chunk = weight[start:start + row_chunk_size].to(
            device="cpu", dtype=torch.float64
        )
        gram.addmm_(chunk.transpose(0, 1), chunk)

    _, eigenvectors = torch.linalg.eigh(gram)
    return eigenvectors[:, :reasoning_size].contiguous()


@torch.no_grad()
def project_content_tokens(model, prompt_ids: torch.Tensor,
                           generated_ids: torch.Tensor,
                           word_groups: list[dict],
                           reasoning_basis: torch.Tensor) -> list[list[float]]:
    """Compute HARP's V_R^T h projections for each grouped output word.

    Each output token is associated with the causal hidden state that predicts
    it. Multi-subtoken words receive the mean of their individual HARP vectors.
    """
    device = model.get_input_embeddings().weight.device
    prompt_ids = prompt_ids.reshape(-1)
    generated_ids = generated_ids.reshape(-1)
    full_ids = torch.cat((prompt_ids, generated_ids)).to(device).unsqueeze(0)
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model.base_model
    hidden = base_model(
        input_ids=full_ids, use_cache=False, return_dict=True
    ).last_hidden_state[0].float()

    basis = reasoning_basis.to(device=hidden.device, dtype=hidden.dtype)
    if basis.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"HARP basis hidden dimension {basis.shape[0]} does not match "
            f"model hidden dimension {hidden.shape[-1]}"
        )

    features = []
    for group in word_groups:
        prediction_positions = [
            prompt_ids.numel() - 1 + index for index in group["token_indices"]
        ]
        if any(position < 0 or position >= hidden.shape[0]
               for position in prediction_positions):
            raise ValueError("Content-token indices do not align with generated token IDs")
        token_hidden = hidden[
            torch.tensor(prediction_positions, device=hidden.device)
        ]
        projected = token_hidden @ basis
        features.append(projected.mean(dim=0).cpu().tolist())
    return features
