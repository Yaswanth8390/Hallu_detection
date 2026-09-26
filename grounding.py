"""Content-word grouping for token-level input-grounding analysis."""

import re

import torch
import torch.nn.functional as F

from model_utils import input_device


_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "as",
    "at", "by", "for", "from", "in", "into", "of", "on", "onto", "per",
    "to", "upon", "via", "with", "about", "above", "after", "against",
    "along", "among", "around", "before", "behind", "below", "beneath",
    "beside", "between", "beyond", "during", "inside", "near", "off", "out",
    "over", "through", "under", "until", "up", "without", "i", "me", "my",
    "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him",
    "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "theirs", "this", "that", "these", "those", "who", "whom", "whose",
    "which", "what", "where", "when", "why", "how", "am", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "doing", "have",
    "has", "had", "having", "can", "could", "may", "might", "must", "shall",
    "should", "will", "would", "also", "very", "just", "some", "any", "each",
    "every", "both", "either", "neither", "such", "own", "same", "more", "most",
    "other", "another", "few", "many", "much", "less", "least", "because",
    "although", "though", "while", "since", "unless", "nor", "yet", "there",
}

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def content_word_groups(tokenizer, generated_ids, generated_text: str,
                        prompt_len: int) -> list[dict]:
    """Return content words and their generated positions, grouped by offsets.

    Subword pieces overlapping the same complete lexical word are kept in one
    group. A clear error is raised if decoded text cannot be aligned back to
    the generated token IDs, rather than silently scoring the wrong positions.
    """
    raw_ids = [int(token_id) for token_id in generated_ids.tolist()]
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    ordinary = [(index, token_id) for index, token_id in enumerate(raw_ids)
                if token_id not in special_ids]
    encoded = tokenizer(
        generated_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    encoded_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if encoded_ids and isinstance(encoded_ids[0], list):
        encoded_ids = encoded_ids[0]
        offsets = offsets[0]
    if [int(token_id) for token_id in encoded_ids] != [token_id for _, token_id in ordinary]:
        raise ValueError(
            "Generated text does not round-trip to generated token IDs; "
            "cannot safely align semantic words with model positions."
        )

    groups = []
    for match in _WORD_PATTERN.finditer(generated_text):
        word = match.group()
        if word.casefold() in _FUNCTION_WORDS:
            continue
        piece_indices = [index for index, (start, end) in enumerate(offsets)
                         if end > match.start() and start < match.end()]
        if not piece_indices:
            continue
        groups.append({
            "token": word,
            "token_indices": [ordinary[index][0] for index in piece_indices],
            "subtoken_count": len(piece_indices),
        })
    return groups


def select_semantic_evidence_spans(model, tokenizer, prompt_text: str,
                                   prompt_ids: torch.Tensor,
                                   generated_ids: torch.Tensor,
                                   question: str, word_groups: list[dict],
                                   max_span_words: int = 5) -> list[dict | None]:
    """Select the most similar contiguous question span for each generated word.

    Both the generated word and candidate spans are mean-pooled final-layer
    contextual hidden states from the same causal forward pass. Candidate
    spans are contiguous n-grams of question words, capped at max_span_words.
    This selects representational similarity; it does not establish entailment.
    """
    question_start = prompt_text.find(question)
    if question_start < 0 or prompt_text.find(question, question_start + 1) >= 0:
        raise ValueError("Question must occur exactly once in the formatted prompt")
    if max_span_words < 1:
        raise ValueError("max_span_words must be at least 1")

    encoded_prompt = tokenizer(
        prompt_text, return_offsets_mapping=True
    )
    encoded_ids = encoded_prompt["input_ids"]
    offsets = encoded_prompt["offset_mapping"]
    if encoded_ids and isinstance(encoded_ids[0], list):
        encoded_ids = encoded_ids[0]
        offsets = offsets[0]
    expected_ids = [int(token_id) for token_id in prompt_ids.tolist()]
    if [int(token_id) for token_id in encoded_ids] != expected_ids:
        raise ValueError("Prompt tokenization with offsets does not match generation prompt IDs")

    lexical_words = list(_WORD_PATTERN.finditer(question))
    candidates = []
    for start_word in range(len(lexical_words)):
        for end_word in range(start_word + 1,
                              min(len(lexical_words), start_word + max_span_words) + 1):
            local_start = lexical_words[start_word].start()
            local_end = lexical_words[end_word - 1].end()
            char_start = question_start + local_start
            char_end = question_start + local_end
            prompt_positions = [
                position for position, (offset_start, offset_end) in enumerate(offsets)
                if offset_end > char_start and offset_start < char_end
            ]
            if prompt_positions:
                candidates.append({
                    "text": question[local_start:local_end],
                    "start": local_start,
                    "end": local_end,
                    "prompt_start": char_start,
                    "prompt_end": char_end,
                    "prompt_positions": prompt_positions,
                })
    if not candidates:
        return [None] * len(word_groups)

    device = input_device(model)
    full_ids = torch.cat((prompt_ids.reshape(-1), generated_ids.reshape(-1)))
    with torch.no_grad():
        base_model = (
            model.get_base_model() if hasattr(model, "get_base_model")
            else model.base_model
        )
        hidden = base_model(
            input_ids=full_ids.to(device).unsqueeze(0),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0]

    hidden = hidden.float()
    candidate_vectors = [
        hidden[torch.tensor(candidate["prompt_positions"], device=hidden.device)].mean(dim=0)
        for candidate in candidates
    ]
    results = []
    for group in word_groups:
        generated_positions = [prompt_ids.numel() + index
                               for index in group["token_indices"]]
        target_vector = hidden[
            torch.tensor(generated_positions, device=hidden.device)
        ].mean(dim=0)
        similarities = F.cosine_similarity(
            torch.stack(candidate_vectors), target_vector.unsqueeze(0), dim=-1
        )
        best_index = int(similarities.argmax().item())
        result = dict(candidates[best_index])
        result.pop("prompt_positions")
        result["similarity"] = similarities[best_index].item()
        results.append(result)
    return results


def classify_input_dependence(delta_logprob: float, threshold: float) -> str:
    """Classify prompt dependence without making a correctness judgment."""
    if delta_logprob >= threshold:
        return "strong_input_dependence"
    if delta_logprob <= -threshold:
        return "negative_input_dependence"
    return "weak_input_dependence"


def classify_confidence(entropy: float, entropy_threshold: float) -> str:
    """Classify a word's original-prompt confidence, independent of the
    question. This says nothing about correctness by itself -- it only says
    whether the model would likely produce this word regardless of prompt.
    Needs experimental calibration per model/vocab, same as
    classify_input_dependence's threshold.
    """
    return "confident" if entropy <= entropy_threshold else "uncertain"


def combine_input_dependence_and_confidence(dependence_classification: str,
                                            confidence_classification: str) -> str:
    """Layer per-word confidence on top of input-dependence.

    Input-dependence alone cannot separate two very different situations
    that both show up as "the question didn't matter for this word":
    the model already knew the fact (parametric knowledge), or the model
    was confabulating regardless of what was asked. Confidence from the
    original, unmodified prompt distinguishes these. A word that already
    showed strong dependence on the question is left alone -- the
    dependence signal is doing the explaining there, and splitting it
    further by confidence would not add information about the question's
    role.
    """
    if dependence_classification == "strong_input_dependence":
        return "input_dependent"
    if confidence_classification == "confident":
        return "parametric_knowledge"
    return "possible_hallucination"


def final_hallucination_label(combined_classification: str) -> str:
    """Collapse the three-way combined label into the binary call the task
    actually asks for. Everything that isn't flagged as a possible
    hallucination -- input-dependent words and words read as confident,
    prompt-independent parametric knowledge -- is reported as not a
    hallucination.

    This inherits every limitation of `combined_classification` verbatim:
    a word the model states confidently and consistently, but which is
    still wrong (a contested or fabricated fact stated with low entropy),
    will be labeled `not_hallucination` here. Confidence regardless of the
    prompt is evidence of parametric knowledge, not proof it's correct --
    telling those apart would need something like resampling the same
    question and checking whether the word is stable, which this label
    does not do.
    """
    if combined_classification == "possible_hallucination":
        return "hallucination"
    return "not_hallucination"