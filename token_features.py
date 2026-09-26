"""Shared semantic, counterfactual, and HARP feature extraction."""

from harp import project_content_tokens
from jacobian import compute_counterfactual_logprobs, compute_token_logprobs
from grounding import select_semantic_evidence_spans


def extract_token_features(model, tokenizer, generation, question: str,
                           word_groups: list[dict],
                           reasoning_basis) -> list[dict]:
    evidence_spans = select_semantic_evidence_spans(
        model, tokenizer, generation.prompt, generation.prompt_ids,
        generation.generated_ids, question, word_groups,
    )
    harp_features = project_content_tokens(
        model, generation.prompt_ids, generation.generated_ids,
        word_groups, reasoning_basis,
    )
    if len(evidence_spans) != len(word_groups) or len(harp_features) != len(word_groups):
        raise RuntimeError("Feature extractors returned inconsistent content-token counts")
    original_scores = compute_token_logprobs(
        model, generation.prompt_ids, generation.generated_ids
    )
    counterfactual_cache = {}
    rows = []

    for group, evidence, harp_vector in zip(
        word_groups, evidence_spans, harp_features
    ):
        token_indices = group["token_indices"]
        counterfactual_score = None
        evidence_text = ""
        semantic_similarity = None
        if evidence is not None:
            span_key = (evidence["start"], evidence["end"])
            evidence_text = evidence["text"]
            semantic_similarity = evidence["similarity"]
            if span_key not in counterfactual_cache:
                counterfactual_question = (
                    question[:evidence["start"]] + question[evidence["end"]:]
                )
                counterfactual_prompt = generation.prompt.replace(
                    question, counterfactual_question, 1
                )
                counterfactual_prompt_ids = tokenizer(
                    counterfactual_prompt, return_tensors="pt"
                ).input_ids[0]
                if counterfactual_prompt_ids.tolist() == generation.prompt_ids.tolist():
                    raise RuntimeError(
                        f"Removing evidence span {evidence_text!r} did not change "
                        "the prompt token IDs; refusing to report a no-op counterfactual."
                    )
                counterfactual_cache[span_key] = compute_counterfactual_logprobs(
                    model, generation.prompt_ids, counterfactual_prompt_ids,
                    generation.generated_ids,
                    original_token_logprobs=original_scores,
                )["counterfactual_token_logprobs"]
            counterfactual_scores = counterfactual_cache[span_key]
            original_logprob = sum(original_scores[index].item()
                                   for index in token_indices)
            counterfactual_logprob = sum(counterfactual_scores[index].item()
                                         for index in token_indices)
            counterfactual_score = original_logprob - counterfactual_logprob

        rows.append({
            "token": group["token"],
            "evidence_span": evidence_text,
            "counterfactual_score": counterfactual_score,
            "semantic_similarity": semantic_similarity,
            "HARP_features": harp_vector,
            "token_indices": token_indices,
            "subtoken_count": group["subtoken_count"],
            "char_start": group["char_start"],
            "char_end": group["char_end"],
        })
    return rows
