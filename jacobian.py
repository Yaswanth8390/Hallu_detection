"""Counterfactual conditional log-probabilities for generated tokens."""

import torch

from model_utils import input_device


def _conditional_token_distributions(model, prompt_ids: torch.Tensor,
                                     generated_ids: torch.Tensor) -> dict:
    """Single forward pass producing everything downstream needs per token:
    the realized log-probability, the full-distribution entropy, and the
    top1-vs-top2 log-probability margin. Computing these together means a
    caller who wants confidence as well as logprob does not pay for a second
    forward pass through the model.
    """
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
        log_probs = logits[positions].log_softmax(dim=-1)
        token_logprob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        top2 = log_probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        return {
            "logprob": token_logprob.cpu(),
            "entropy": entropy.cpu(),
            "margin": margin.cpu(),
        }


def _conditional_token_logprobs(model, prompt_ids: torch.Tensor,
                               generated_ids: torch.Tensor) -> torch.Tensor:
    """Score fixed generated tokens conditioned on a prompt and their prefix."""
    return _conditional_token_distributions(model, prompt_ids, generated_ids)["logprob"]


def compute_token_logprobs(model, prompt_ids: torch.Tensor,
                           generated_ids: torch.Tensor) -> torch.Tensor:
    """Return conditional log-probabilities for a fixed generated sequence."""
    return _conditional_token_logprobs(model, prompt_ids, generated_ids)


def compute_token_confidence(model, prompt_ids: torch.Tensor,
                             generated_ids: torch.Tensor) -> dict:
    """Per-generated-token log-probability, entropy, and top1/top2 margin
    (all in nats, from natural-log softmax) under a single prompt.

    This is meant to be called on the *original* prompt only, as a read of
    how confident the model was in each word independent of whether that
    word turned out to depend on the question. Low entropy / high margin
    means the model would likely have produced this word regardless of the
    prompt -- consistent with correct parametric knowledge, but not proof of
    it. High entropy / low margin combined with low input-dependence is the
    more concerning pattern: the model wasn't relying on the question *and*
    wasn't sure of itself either.
    """
    return _conditional_token_distributions(model, prompt_ids, generated_ids)


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