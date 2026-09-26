"""Check content-word grouping and grouped-subword scoring on a tiny model."""

import csv
import io
import math

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from grounding import content_word_groups, grounding_strength
from jacobian import compute_directional_sensitivity


class OffsetTokenizer:
    all_special_ids = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text == "The Internationalization 42."
        result = {
            "input_ids": [11, 21, 22, 23, 24],
            "offset_mapping": [(0, 3), (4, 12), (12, 24), (25, 27), (27, 28)],
        }
        return result


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
for parameter in model.parameters():
    parameter.requires_grad_(False)

prompt_ids = torch.tensor([10, 12, 13])
generated_ids = torch.tensor([11, 21, 22, 23, 24])
full_ids = torch.cat([prompt_ids, generated_ids])
groups = content_word_groups(
    OffsetTokenizer(), generated_ids, "The Internationalization 42.", len(prompt_ids)
)
assert [group["token"] for group in groups] == ["Internationalization", "42"]
assert groups[0]["positions"] == [len(prompt_ids) + 1, len(prompt_ids) + 2]
assert groups[0]["subtoken_count"] == 2
assert groups[1]["positions"] == [len(prompt_ids) + 3]

rows = []
for index, group in enumerate(groups):
    score = compute_directional_sensitivity(
        model, full_ids, len(prompt_ids), group["positions"]
    )
    alignment = score["alignment_score"]
    assert math.isfinite(alignment)
    assert -1.0 <= alignment <= 1.0
    rows.append({
        "token_index_in_answer": index,
        "token": group["token"],
        "alignment_score": alignment,
        "grounding_strength": grounding_strength(alignment, 0.1),
    })

assert grounding_strength(-0.5, 0.2) == "strong"
assert grounding_strength(-0.05, 0.2) == "weak"
assert "hallucination" not in rows[0]

buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
buffer.seek(0)
assert len(list(csv.DictReader(buffer))) == 2

print("Content-word alignment smoke test passed:", rows)