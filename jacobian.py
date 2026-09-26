"""
Layer-wise Jacobian (vector-Jacobian product) extraction.

We do NOT materialize the full d x d Jacobian matrix between consecutive layers
(too expensive: d ~ 3584 for Qwen2.5-7B, and we'd need it per token per layer).
Instead, for a chosen scalar (the logit of the actually-generated token at a
given position), we compute its gradient w.r.t. the hidden state at every
layer in a single backward pass. This gradient vector at layer l tells you
the direction in hidden-state space at layer l that would most increase the
target logit -- i.e. how much and in what direction each layer's
representation is "responsible for" the final prediction. Tracking the norm
and direction of this vector across layers is the "how the token's
representation evolves" signal from the problem statement.

This costs one forward + one backward pass per token position (not per
layer), because retain_grad() on every layer's hidden state lets a single
backward() populate all of their .grad tensors at once.
"""

from dataclasses import dataclass
from typing import List

import torch

from model_utils import input_device, output_device


@dataclass
class TokenJacobianTrace:
    position: int                     # index into full_ids of the predicted token
    token_id: int
    layer_grad_norms: List[float]     # len = num_layers + 1 (embeddings + each block)
    layer_grads: List[torch.Tensor]   # raw gradient vectors per layer, each (hidden_dim,)


def compute_layerwise_jacobian(model, full_ids: torch.Tensor, positions: List[int],
                                device: str = "cuda") -> List[TokenJacobianTrace]:
    """For each position p in `positions`, compute d(logit of full_ids[p]) / d(hidden_state[l])
    for every layer l, using the hidden state at position p-1 (which produced that logit).

    `positions` should be the indices of the generated tokens in `full_ids`
    (i.e. prompt_len, prompt_len+1, ...).

    `device` is accepted for backward compatibility but ignored for actual
    placement -- input_ids go on input_device(model) and the manual lm_head
    call happens on output_device(model), so this is correct whether the
    model lives on one GPU or is split across several via device_map="auto".
    """
    full_ids = full_ids.to(input_device(model)).unsqueeze(0)  # (1, seq_len)
    lm_head_device = output_device(model)
    traces = []

    # One forward+backward per target position. This is O(num_generated_tokens)
    # forward/backward passes, each over the *full* sequence -- fine for short
    # generations (a few dozen tokens) and a single QA example at a time.
    for p in positions:
        model.zero_grad(set_to_none=True)

        # Build inputs_embeds explicitly and force requires_grad on THAT
        # tensor, rather than passing full_ids (token ids) straight into the
        # model. This matters because model_utils.load_model freezes every
        # model parameter (to avoid allocating a full weight-sized gradient
        # buffer on every backward -- the actual fix for the OOM you hit).
        # With every weight frozen, model(full_ids, ...) would produce a
        # graph where nothing requires grad at all (int token ids aren't
        # differentiable, and a frozen embedding matrix wouldn't make its
        # output require grad either), and .backward() would fail outright.
        # Making inputs_embeds itself a requires_grad leaf sidesteps that:
        # every downstream hidden state requires grad because of THIS
        # tensor, regardless of which weights are frozen.
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(full_ids).detach().clone().requires_grad_(True)

        out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states  # tuple, len = num_layers+1
        for hs in hidden_states:
            hs.retain_grad()

        target_token_id = full_ids[0, p].item()
        # hidden_states[-1] is the final layer's output (pre-lm_head, post final norm
        # in most HF causal LMs the norm is applied inside the model before hidden_states
        # is returned as the last element -- check model.config for exact norm placement
        # if results look off).
        final_hidden = hidden_states[-1][0, p - 1, :].to(lm_head_device)  # (hidden_dim,)
        logits_at_pos = model.lm_head(final_hidden)         # (vocab,)
        target_logit = logits_at_pos[target_token_id]

        target_logit.backward()

        layer_grad_norms = []
        layer_grads = []
        for hs in hidden_states:
            g = hs.grad[0, p - 1, :].detach().cpu()
            layer_grad_norms.append(g.norm().item())
            layer_grads.append(g)

        traces.append(TokenJacobianTrace(
            position=p,
            token_id=target_token_id,
            layer_grad_norms=layer_grad_norms,
            layer_grads=layer_grads,
        ))

    return traces


def trajectory_features(trace: TokenJacobianTrace) -> dict:
    """Cheap scalar summary of a single token's layer-wise Jacobian trajectory,
    for use as classifier features. Extend this as you find signal.
    """
    norms = trace.layer_grad_norms
    n = len(norms)
    return {
        "grad_norm_final_layer": norms[-1],
        "grad_norm_mean": sum(norms) / n,
        "grad_norm_max": max(norms),
        "grad_norm_argmax_layer_frac": norms.index(max(norms)) / (n - 1),
        # Ratio of late-layer to early-layer sensitivity: hallucinated tokens
        # are hypothesized to show representation "drift" late in the network
        # rather than early grounding -- this is the feature to validate.
        "late_to_early_ratio": (sum(norms[n // 2:]) + 1e-8) / (sum(norms[: n // 2]) + 1e-8),
    }
