"""Input-embedding directional sensitivity for generated semantic tokens."""

import torch
import torch.nn.functional as F

from model_utils import input_device


def compute_directional_sensitivity(model, full_ids: torch.Tensor,
                                    prompt_len: int,
                                    token_positions: list[int]) -> dict:
    """Score a semantic token represented by one or more generated sub-tokens.

    The scalar target is the sum of its constituent next-token logits. Its
    gradient is compared with the concatenated prompt-embedding direction.
    The signed cosine is returned unchanged; its sign is not a hallucination
    label.
    """
    if not token_positions:
        raise ValueError("token_positions must contain at least one generated position")
    if prompt_len <= 0 or any(position <= 0 for position in token_positions):
        raise ValueError("prompt_len and generated token positions must be positive")

    device = input_device(model)
    full_ids = full_ids.to(device).unsqueeze(0)
    model.zero_grad(set_to_none=True)

    embeddings = model.get_input_embeddings()(full_ids).detach().clone()
    embeddings.requires_grad_(True)
    output = model(inputs_embeds=embeddings, use_cache=False)

    input_positions = torch.tensor(token_positions, device=full_ids.device)
    logits_positions = input_positions.to(output.logits.device)
    target_ids = full_ids[0, input_positions].to(output.logits.device)
    target_logits = output.logits[0, logits_positions - 1, target_ids]
    grouped_logit = target_logits.sum()
    gradient = torch.autograd.grad(grouped_logit, embeddings)[0][0, :prompt_len]
    input_direction = embeddings.detach()[0, :prompt_len]

    alignment = F.cosine_similarity(
        gradient.reshape(1, -1), input_direction.reshape(1, -1), dim=1
    ).item()
    return {
        "target_logit": grouped_logit.detach().item(),
        "alignment_score": alignment,
    }