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
layer), because forward hooks on every decoder layer (+ the final norm)
capture each layer's live output tensor during the forward pass and
retain_grad() on it, so a single backward() populates all of their .grad
tensors at once.

We deliberately do NOT rely on `output_hidden_states=True`'s returned tuple
for this. In several transformers versions those tensors get detached (or
otherwise disconnected from the live graph) before being stored in the
output tuple, since normally nobody backprops through them -- that silently
produces `hs.grad is None` after backward() with no error at any earlier
step. Forward hooks capture the tensor *while it's still part of the active
graph*, which sidesteps that regardless of the transformers version's
internal behavior. It also fixes a related correctness bug: the returned
hidden_states tuple's last entry is usually the last decoder layer's output
*before* the final RMSNorm, so feeding it straight into lm_head (as an
earlier version of this file did) skips the norm and doesn't match the
model's real logits. Hooking the norm module directly gives the actually-
normed hidden state lm_head expects.
"""

from dataclasses import dataclass
from typing import List

import torch

from model_utils import input_device, output_device


@dataclass
class TokenJacobianTrace:
    position: int                     # index into full_ids of the predicted token
    token_id: int
    layer_grad_norms: List[float]     # embeddings + each decoder layer + final norm
    layer_grads: List[torch.Tensor]   # raw gradient vectors per layer, each (hidden_dim,)


def _get_decoder_layers(model):
    """Locate the ModuleList of transformer decoder layers and the final norm
    module. Works for the standard Llama/Qwen2-family layout
    (model.model.layers, model.model.norm); adjust here if you point this at
    a model with a different attribute layout.
    """
    base = model.model  # Qwen2Model / LlamaModel etc.
    return base.layers, base.norm


def compute_layerwise_jacobian(model, full_ids: torch.Tensor, positions: List[int],
                                device: str = "cuda") -> List[TokenJacobianTrace]:
    """For each position p in `positions`, compute d(logit of full_ids[p]) / d(hidden_state[l])
    for every layer l (embeddings, each decoder layer, and the final norm),
    using the hidden state at position p-1 (which produced that logit).

    `positions` should be the indices of the generated tokens in `full_ids`
    (i.e. prompt_len, prompt_len+1, ...).

    `device` is accepted for backward compatibility but ignored for actual
    placement -- input_ids go on input_device(model) and the manual lm_head
    call happens on output_device(model), so this is correct whether the
    model lives on one GPU or is split across several via device_map="auto".
    """
    full_ids = full_ids.to(input_device(model)).unsqueeze(0)  # (1, seq_len)
    lm_head_device = output_device(model)
    decoder_layers, final_norm = _get_decoder_layers(model)
    traces = []

    # One forward+backward per target position. This is O(num_generated_tokens)
    # forward/backward passes, each over the *full* sequence -- fine for short
    # generations (a few dozen tokens) and a single QA example at a time.
    for p in positions:
        model.zero_grad(set_to_none=True)

        captured = []  # filled in forward order: [layer_0_out, layer_1_out, ..., norm_out]

        def _make_hook():
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs.retain_grad()
                captured.append(hs)
                # Returning None leaves `output` unchanged -- we're only
                # observing here, not modifying the forward computation.
                return None
            return hook

        handles = [layer.register_forward_hook(_make_hook()) for layer in decoder_layers]
        handles.append(final_norm.register_forward_hook(_make_hook()))

        # Build inputs_embeds explicitly and force requires_grad on THAT
        # tensor, rather than passing full_ids (token ids) straight into the
        # model. This matters because model_utils.load_model freezes every
        # model parameter (to avoid allocating a full weight-sized gradient
        # buffer on every backward). With every weight frozen, model(full_ids,
        # ...) would produce a graph where nothing requires grad at all (int
        # token ids aren't differentiable, and a frozen embedding matrix
        # wouldn't make its output require grad either). Making inputs_embeds
        # itself a requires_grad leaf sidesteps that: every downstream hidden
        # state requires grad because of THIS tensor, regardless of which
        # weights are frozen.
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(full_ids).detach().clone().requires_grad_(True)
        inputs_embeds.retain_grad()

        model(inputs_embeds=inputs_embeds, use_cache=False)

        for h in handles:
            h.remove()

        assert len(captured) == len(decoder_layers) + 1, (
            f"expected {len(decoder_layers) + 1} captured tensors (one per decoder "
            f"layer + final norm), got {len(captured)} -- a hook didn't fire, which "
            "usually means the model's forward signature doesn't call these modules "
            "the way this function assumes; check _get_decoder_layers() against your "
            "model's actual module structure."
        )
        all_hidden = [inputs_embeds] + captured  # embeddings, layer_0..N-1 outputs, final-norm output

        target_token_id = full_ids[0, p].item()
        final_hidden = all_hidden[-1][0, p - 1, :].to(lm_head_device)  # post-norm, correct lm_head input
        logits_at_pos = model.lm_head(final_hidden)         # (vocab,)
        target_logit = logits_at_pos[target_token_id]

        target_logit.backward()

#        for i, hs in enumerate(all_hidden):
#            print(
#                f"layer {i}: "
#                f"requires_grad={hs.requires_grad}, "
#                f"is_leaf={hs.is_leaf}, "
#                f"grad_fn={type(hs.grad_fn).__name__ if hs.grad_fn else None}, "
#                f"grad_none={hs.grad is None}"
#            )

        layer_grad_norms = []
        layer_grads = []
        for hs in all_hidden:
            if hs.grad is None:
                raise RuntimeError(
                    "hs.grad is still None after backward() even with forward-hook "
                    "capture -- the autograd graph is broken somewhere upstream of "
                    "this specific layer. Print `hs.requires_grad` for each captured "
                    "tensor right after the forward pass to find where it flips to "
                    "False; that pinpoints which layer/hook/op is detaching."
                )
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
