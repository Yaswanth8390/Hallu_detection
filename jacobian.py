"""Counterfactual conditional log-probabilities for generated tokens."""

import torch

from model_utils import input_device


def _conditional_token_logprobs(model, prompt_ids: torch.Tensor,
                               generated_ids: torch.Tensor) -> torch.Tensor:
    """Score fixed generated tokens conditioned on a prompt and their prefix."""
    device = input_device(model)
    prompt_ids = prompt_ids.to(device).reshape(-1)
    generated_ids = generated_ids.to(device).reshape(-1)
    sequence = torch.cat((prompt_ids, generated_ids)).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=sequence, use_cache=False).logits[0].float()
        positions = torch.arange(
            prompt_ids.numel() - 1,
            prompt_ids.numel() + generated_ids.numel() - 1,
            device=logits.device,
        )
        targets = generated_ids.to(logits.device)
        return logits[positions].log_softmax(dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1).cpu()


def compute_token_logprobs(model, prompt_ids: torch.Tensor,
                           generated_ids: torch.Tensor) -> torch.Tensor:
    """Return conditional log-probabilities for a fixed generated sequence."""
    return _conditional_token_logprobs(model, prompt_ids, generated_ids)


def compute_counterfactual_logprobs(model, original_prompt_ids: torch.Tensor,
                                    counterfactual_prompt_ids: torch.Tensor,
                                    generated_ids: torch.Tensor,
                                    original_token_logprobs: torch.Tensor | None = None) -> dict:
    """Score the same generated sequence under original and altered prompts.

    Because each output position is causally masked, each subtoken's score is
    conditioned only on the prompt and the unchanged generated prefix before
    that subtoken, not on later generated tokens.
    """
    original = original_token_logprobs
    if original is None:
        original = _conditional_token_logprobs(model, original_prompt_ids, generated_ids)
    counterfactual = _conditional_token_logprobs(
        model, counterfactual_prompt_ids, generated_ids
    )
    return {
        "original_token_logprobs": original,
        "counterfactual_token_logprobs": counterfactual,
    }