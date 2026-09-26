"""Check grouped-word counterfactual scores and token-level CSV output."""

import csv
import io

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from grounding import classify_input_dependence, content_word_groups
from jacobian import compute_counterfactual_logprobs


class OffsetTokenizer:
    all_special_ids = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text == "The Internationalization 42."
        return {
            "input_ids": [11, 21, 22, 23, 24],
            "offset_mapping": [(0, 3), (4, 12), (12, 24), (25, 27), (27, 28)],
        }


torch.manual_seed(0)
config = Qwen2Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
model = AutoModelForCausalLM.from_config(config)
model.eval()

prompt_ids = torch.tensor([10, 12, 13])
counterfactual_prompt_ids = torch.tensor([10, 13])
generated_ids = torch.tensor([11, 21, 22, 23, 24])
sentence = "The Internationalization 42."
groups = content_word_groups(
    OffsetTokenizer(), generated_ids, sentence, len(prompt_ids)
)
assert [group["token"] for group in groups] == ["Internationalization", "42"]
assert groups[0]["token_indices"] == [1, 2]
assert groups[0]["subtoken_count"] == 2
assert groups[1]["token_indices"] == [3]

logprobs = compute_counterfactual_logprobs(
    model, prompt_ids, counterfactual_prompt_ids, generated_ids
)
rows = []
for token_index, group in enumerate(groups):
    indices = group["token_indices"]
    original_logprob = sum(logprobs["original_token_logprobs"][i].item() for i in indices)
    counterfactual_logprob = sum(
        logprobs["counterfactual_token_logprobs"][i].item() for i in indices
    )
    delta_logprob = original_logprob - counterfactual_logprob
    rows.append({
        "generated_text": sentence,
        "token_index_in_answer": token_index,
        "token": group["token"],
        "evidence_span": "example evidence",
        "original_logprob": original_logprob,
        "counterfactual_logprob": counterfactual_logprob,
        "delta_logprob": delta_logprob,
        "classification": classify_input_dependence(delta_logprob, 0.1),
    })

assert classify_input_dependence(0.5, 0.1) == "strong_input_dependence"
assert classify_input_dependence(-0.5, 0.1) == "negative_input_dependence"
assert classify_input_dependence(0.05, 0.1) == "weak_input_dependence"
assert all(row["generated_text"] == sentence for row in rows)

buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
buffer.seek(0)
read_back = list(csv.DictReader(buffer))
assert len(read_back) == 2
assert read_back[0]["generated_text"] == sentence
assert {"evidence_span", "original_logprob", "counterfactual_logprob",
        "delta_logprob", "classification"}.issubset(read_back[0])

print("Grouped counterfactual score smoke test passed:", rows)