"""Content-word grouping for token-level input-grounding analysis."""

import re

import torch
import torch.nn.functional as F

from model_utils import input_device


_CONTENT_POS_TAGS = {"ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"}
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def load_content_tagger(model_name: str = "en_core_web_sm"):
    """Load a POS/NER tagger so content words are selected linguistically."""
    import spacy

    try:
        return spacy.load(model_name)
    except OSError as error:
        raise RuntimeError(
            f"Could not load spaCy model {model_name!r}. Install it with "
            f"`python -m spacy download {model_name}`."
        ) from error


def content_word_groups(tokenizer, generated_ids, generated_text: str,
                        prompt_len: int, pos_tagger) -> list[dict]:
    """Return POS/NER-selected content words with their generated positions.

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
    for word in pos_tagger(generated_text):
        if word.is_space or word.is_punct:
            continue
        if word.pos_ not in _CONTENT_POS_TAGS and not word.ent_type_:
            continue
        start = word.idx
        end = start + len(word.text)
        piece_indices = [index for index, (start, end) in enumerate(offsets)
                         if end > word.idx and start < word.idx + len(word.text)]
        if not piece_indices:
            continue
        groups.append({
            "token": word.text,
            "char_start": start,
            "char_end": end,
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