"""Content-word grouping for token-level input-grounding analysis."""

import re


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
    "other", "another", "few", "many", "much", "less", "least",
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
            "positions": [prompt_len + ordinary[index][0] for index in piece_indices],
            "subtoken_count": len(piece_indices),
        })
    return groups


def grounding_strength(alignment_score: float, threshold: float) -> str:
    """Describe directional grounding strength without assigning correctness."""
    return "strong" if abs(alignment_score) >= threshold else "weak"