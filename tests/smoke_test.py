"""
Structural smoke test using a tiny randomly-initialized Qwen2-architecture
model (same model class as Qwen2.5-7B, just small + random weights, built
from a config so no network access / download is needed). This validates
that the hook/backward/shape logic in jacobian.py and grounding.py is
correct -- it does NOT validate anything about real hallucination signal
(random weights carry no semantics).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from jacobian import compute_layerwise_jacobian, trajectory_features
from grounding import find_entity_span, gradient_grounding_score, ablation_grounding_score

torch.manual_seed(0)

# Tiny config, same architecture family as Qwen2.5-7B.
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

device = "cpu"

# Build a fake "prompt" + "generated" sequence: pretend token ids 10..14 are
# the prompt (with a fake "entity" at position 2), and 20, 21 are generated.
prompt_ids = torch.tensor([10, 11, 20, 12, 13])  # "entity" at index 2 (token id 20)
gen_ids = torch.tensor([21, 22])
full_ids = torch.cat([prompt_ids, gen_ids])
prompt_len = prompt_ids.shape[0]

print("full_ids:", full_ids.tolist())

# --- Jacobian trace ---
positions = list(range(prompt_len, full_ids.shape[0]))
traces = compute_layerwise_jacobian(model, full_ids, positions=positions, device=device)
assert len(traces) == len(positions)
for t in traces:
    assert len(t.layer_grad_norms) == config.num_hidden_layers + 1
    feats = trajectory_features(t)
    print(f"position={t.position} token_id={t.token_id} "
          f"norms={[round(n, 4) for n in t.layer_grad_norms]} feats={feats}")

# --- Grounding: gradient attribution ---
entity_span = (2, 3)  # token id 20 is our fake "entity" at index 2
target_position = prompt_len
grad_score = gradient_grounding_score(model, full_ids, target_position, entity_span, device=device)
print("gradient_grounding_score:", grad_score)
assert 0.0 <= grad_score["grounding_fraction"] <= 1.0 + 1e-6

# --- Grounding: ablation (needs a tokenizer for the baseline token; fake a
# minimal one via a real small tokenizer's encode of a space+word, or just
# monkeypatch since we're on a random-vocab tiny model) ---
class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [5]  # arbitrary in-vocab id as "baseline" token
        return R()

dummy_tok = DummyTokenizer()
ablation_score = ablation_grounding_score(model, dummy_tok, full_ids, target_position,
                                           entity_span, device=device)
print("ablation_grounding_score:", ablation_score)
assert "logit_drop" in ablation_score

print("\nAll smoke tests passed: shapes, hooks, and backward passes behave as expected.")
