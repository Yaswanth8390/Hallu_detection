"""Check counterfactual token log-probabilities on a tiny Qwen model."""

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from jacobian import compute_counterfactual_logprobs

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

original_prompt_ids = torch.tensor([10, 11, 20, 12])
counterfactual_prompt_ids = torch.tensor([10, 12])
generated_ids = torch.tensor([21, 22, 23])
scores = compute_counterfactual_logprobs(
    model, original_prompt_ids, counterfactual_prompt_ids, generated_ids
)

assert scores["original_token_logprobs"].shape == generated_ids.shape
assert scores["counterfactual_token_logprobs"].shape == generated_ids.shape
assert torch.isfinite(scores["original_token_logprobs"]).all()
assert torch.isfinite(scores["counterfactual_token_logprobs"]).all()
assert torch.all(scores["original_token_logprobs"] <= 0)
assert torch.all(scores["counterfactual_token_logprobs"] <= 0)

print("Counterfactual conditional-logprob smoke test passed:", scores)