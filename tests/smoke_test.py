"""Check input-direction sensitivity on a tiny random-init Qwen model."""

import math

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from jacobian import compute_directional_sensitivity

torch.manual_seed(0)

config = Qwen2Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
model = AutoModelForCausalLM.from_config(config)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

prompt_ids = torch.tensor([10, 11, 20, 12, 13])
generated_ids = torch.tensor([21, 22])
full_ids = torch.cat([prompt_ids, generated_ids])

score = compute_directional_sensitivity(
    model, full_ids, prompt_ids.numel(), [prompt_ids.numel()]
)
assert math.isfinite(score["alignment_score"])
assert -1.0 <= score["alignment_score"] <= 1.0
assert math.isfinite(score["target_logit"])

print("Input-direction sensitivity smoke test passed:", score)