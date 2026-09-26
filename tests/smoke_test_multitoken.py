"""Check grouped-word counterfactual scores and token-level CSV output."""

import csv
import io
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from grounding import (
    classify_confidence,
    classify_input_dependence,
    combine_input_dependence_and_confidence,
    content_word_groups,
    final_hallucination_label,
    select_semantic_evidence_spans,
)
from jacobian import compute_counterfactual_logprobs, compute_token_confidence


class OffsetTokenizer:
    all_special_ids = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text == "The Internationalization 42."
        return {
            "input_ids": [11, 21, 22, 23, 24],
            "offset_mapping": [(0, 3), (4, 12), (12, 24), (25, 27), (27, 28)],
        }


class CharacterOffsetTokenizer:
    all_special_ids = []

    def __call__(self, text, return_offsets_mapping=False):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class CueRepresentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 2)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:, 1] = 1.0
            for character in "1984":
                self.embedding.weight[ord(character)] = torch.tensor([1.0, 0.0])
            self.embedding.weight[200] = torch.tensor([1.0, 0.0])

    def get_input_embeddings(self):
        return self.embedding

    def get_base_model(self):
        return self

    def forward(self, input_ids, use_cache=False, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


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
assert (sentence[:4] + sentence[4 + len("Internationalization"):]
        == "The  42.")

semantic_question = "Who wrote 1984?"
semantic_prompt = f"Question: {semantic_question}"
semantic_tokenizer = CharacterOffsetTokenizer()
semantic_prompt_ids = torch.tensor(semantic_tokenizer(semantic_prompt)["input_ids"])
semantic_match = select_semantic_evidence_spans(
    CueRepresentationModel(),
    semantic_tokenizer,
    semantic_prompt,
    semantic_prompt_ids,
    torch.tensor([200]),
    semantic_question,
    [{"token": "Orwell", "token_indices": [0]}],
)[0]
assert semantic_match["text"] == "1984"
assert semantic_match["similarity"] > 0.99
counterfactual_question = (
    semantic_question[:semantic_match["start"]]
    + semantic_question[semantic_match["end"]:]
)
assert counterfactual_question == "Who wrote ?"
assert counterfactual_question != semantic_question
assert (semantic_prompt.replace(semantic_question, counterfactual_question, 1)
        != semantic_prompt)
tiny_model_match = select_semantic_evidence_spans(
    model,
    semantic_tokenizer,
    semantic_prompt,
    semantic_prompt_ids,
    torch.tensor([201]),
    semantic_question,
    [{"token": "Orwell", "token_indices": [0]}],
)[0]
assert tiny_model_match["text"]
assert -1.0 <= tiny_model_match["similarity"] <= 1.0

logprobs = compute_counterfactual_logprobs(
    model, prompt_ids, counterfactual_prompt_ids, generated_ids
)
assert not torch.equal(
    logprobs["original_token_logprobs"],
    logprobs["counterfactual_token_logprobs"],
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

assert classify_confidence(0.5, 1.0) == "confident"
assert classify_confidence(1.5, 1.0) == "uncertain"

# strong dependence is left alone regardless of confidence
assert combine_input_dependence_and_confidence(
    "strong_input_dependence", "confident"
) == "input_dependent"
assert combine_input_dependence_and_confidence(
    "strong_input_dependence", "uncertain"
) == "input_dependent"
# low/no dependence is where confidence does the separating
assert combine_input_dependence_and_confidence(
    "weak_input_dependence", "confident"
) == "parametric_knowledge"
assert combine_input_dependence_and_confidence(
    "weak_input_dependence", "uncertain"
) == "possible_hallucination"
assert combine_input_dependence_and_confidence(
    "no_matching_evidence_span", "confident"
) == "parametric_knowledge"

assert final_hallucination_label("possible_hallucination") == "hallucination"
assert final_hallucination_label("parametric_knowledge") == "not_hallucination"
assert final_hallucination_label("input_dependent") == "not_hallucination"

confidence = compute_token_confidence(model, prompt_ids, generated_ids)
assert confidence["logprob"].shape == generated_ids.shape
assert confidence["entropy"].shape == generated_ids.shape
assert confidence["margin"].shape == generated_ids.shape
assert torch.all(confidence["entropy"] >= 0)
assert torch.all(confidence["margin"] >= 0)

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