"""
Grounding score: is the generated token's representation actually tied to a
specific input entity (e.g. does "Paris" trace back to "France" in the
prompt), or is it floating free of the input (a hallucination red flag)?

Two complementary signals, both reusing the Jacobian machinery:

1. Gradient attribution (cheap, correlational): backprop the target logit
   all the way to the *input embeddings* and see how much gradient mass
   lands on the entity's token positions vs. the rest of the prompt.

2. Causal ablation (more expensive, causal): replace the entity's token
   embeddings with a neutral baseline, re-run the forward pass, and measure
   how much the target logit actually drops. A token that's "grounded" in
   the entity should drop a lot when the entity is ablated; a hallucinated
   token often won't, because it was never really relying on that entity.

Both need to know *which input positions* correspond to the entity you're
checking grounding against (e.g. "France" for the question "What is the
capital of France?"). `find_entity_span` does simple token-level string
matching for this -- swap in a proper NER/span-finder if entities aren't
literally substrings of the prompt.
"""

from typing import List, Optional, Tuple

import torch

from model_utils import input_device


def find_entity_span(tokenizer, prompt_ids: torch.Tensor, entity: str) -> Optional[Tuple[int, int]]:
    """Find the (start, end) token index range in prompt_ids whose decoded text
    contains `entity` (case-insensitive substring match). Returns None if not found.

    This is intentionally simple (token-window scan + decode + substring check)
    rather than exact offset-mapping, since Qwen's BPE tokenizer doesn't always
    align word boundaries to token boundaries. Good enough for named entities
    that are a few tokens long; verify manually for anything ambiguous.
    """
    ids = prompt_ids.tolist()
    entity_lower = entity.lower().strip()
    if not entity_lower:
        return None
    for window in range(1, 6):  # entities are rarely more than ~5 tokens
        for start in range(0, len(ids) - window + 1):
            span_text = tokenizer.decode(ids[start:start + window]).lower()
            if entity_lower in span_text:
                return (start, start + window)
    return None


def gradient_grounding_score(model, full_ids: torch.Tensor, position: int,
                              entity_span: Tuple[int, int], device: str = "cuda") -> dict:
    """Gradient-attribution grounding score for the token predicted at `position`,
    w.r.t. the entity token span in the prompt.

    Returns the fraction of total input-embedding gradient norm that falls on
    the entity span vs. everything else. High fraction => token's logit is
    gradient-sensitive to the entity; low fraction => logit barely depends on
    the entity at all (candidate hallucination signal).

    `device` is accepted for backward compatibility but ignored -- inputs
    always go on input_device(model), correct whether the model is on one
    GPU or split across several via device_map="auto".
    """
    full_ids = full_ids.to(input_device(model)).unsqueeze(0)
    model.zero_grad(set_to_none=True)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(full_ids).detach().clone().requires_grad_(True)

    out = model(inputs_embeds=inputs_embeds, use_cache=False)
    target_token_id = full_ids[0, position].item()
    logits_at_pos = out.logits[0, position - 1, :]
    target_logit = logits_at_pos[target_token_id]
    target_logit.backward()

    grad = inputs_embeds.grad[0]              # (seq_len, hidden_dim)
    per_token_norm = grad.norm(dim=-1)         # (seq_len,)

    start, end = entity_span
    entity_norm = per_token_norm[start:end].sum().item()
    total_norm = per_token_norm[: position].sum().item()  # only positions causally visible
    other_norm = max(total_norm - entity_norm, 1e-8)

    return {
        "entity_grad_norm": entity_norm,
        "total_grad_norm": total_norm,
        "grounding_fraction": entity_norm / (total_norm + 1e-8),
        "entity_to_other_ratio": entity_norm / other_norm,
    }


@torch.no_grad()
def ablation_grounding_score(model, tokenizer, full_ids: torch.Tensor, position: int,
                              entity_span: Tuple[int, int], device: str = "cuda",
                              baseline_token: str = " something") -> dict:
    """Causal grounding score: replace the entity span with a neutral baseline
    token (repeated to match span length), re-run the forward pass, and measure
    the drop in the target token's logit and its rank.

    Larger drop / bigger rank change => the original prediction was causally
    relying on that entity (well-grounded). Little to no change => the model
    predicted the token largely independent of the entity being present at
    all, which is the hallucination-flavored failure mode we're hunting for.

    `device` is accepted for backward compatibility but ignored -- inputs
    always go on input_device(model).
    """
    full_ids = full_ids.to(input_device(model)).unsqueeze(0).clone()
    start, end = entity_span
    target_token_id = full_ids[0, position].item()

    # Clean run.
    clean_out = model(full_ids, use_cache=False)
    clean_logits = clean_out.logits[0, position - 1, :]
    clean_logit = clean_logits[target_token_id].item()
    clean_rank = (clean_logits > clean_logits[target_token_id]).sum().item()

    # Ablated run: overwrite the entity span with a repeated neutral token.
    baseline_id = tokenizer(baseline_token, add_special_tokens=False).input_ids[0]
    ablated_ids = full_ids.clone()
    ablated_ids[0, start:end] = baseline_id

    ablated_out = model(ablated_ids, use_cache=False)
    ablated_logits = ablated_out.logits[0, position - 1, :]
    ablated_logit = ablated_logits[target_token_id].item()
    ablated_rank = (ablated_logits > ablated_logits[target_token_id]).sum().item()

    return {
        "clean_logit": clean_logit,
        "ablated_logit": ablated_logit,
        "logit_drop": clean_logit - ablated_logit,
        "clean_rank": clean_rank,
        "ablated_rank": ablated_rank,
        "rank_worsened_by": ablated_rank - clean_rank,
    }
