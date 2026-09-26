"""
Integration smoke test for the multi-token loop added to run_pipeline.py:
simulates one example's worth of (multi-position) Jacobian + grounding
extraction and row-building on the same tiny random-init model as
smoke_test.py, to catch bugs in the aggregation/CSV logic without needing a
real model download or TruthfulQA access.
"""

import csv
import io

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from jacobian import compute_layerwise_jacobian, trajectory_features
from grounding import find_entity_span, gradient_grounding_score, ablation_grounding_score
from run_pipeline import mean_aggregate

torch.manual_seed(0)

config = Qwen2Config(
    vocab_size=1000, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=128,
)
model = AutoModelForCausalLM.from_config(config)
model.eval()
device = "cpu"

prompt_ids = torch.tensor([10, 11, 20, 12, 13])   # fake "entity" token 20 at index 2
gen_ids = torch.tensor([21, 22, 23])                # pretend a 3-token answer this time
full_ids = torch.cat([prompt_ids, gen_ids])
prompt_len = prompt_ids.shape[0]
positions = list(range(prompt_len, full_ids.shape[0]))
assert len(positions) == 3, "expected 3 generated-token positions"

traces = compute_layerwise_jacobian(model, full_ids, positions=positions, device=device)
assert len(traces) == 3
per_token_traj_feats = [trajectory_features(t) for t in traces]

span = (2, 3)


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [5]
        return R()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


dummy_tok = DummyTokenizer()

per_token_grad_scores = []
per_token_ablation_scores = []
for p in positions:
    per_token_grad_scores.append(gradient_grounding_score(model, full_ids, p, span, device=device))
    per_token_ablation_scores.append(
        ablation_grounding_score(model, dummy_tok, full_ids, p, span, device=device))

# --- per-token rows (mirrors run_pipeline.py's inner loop) ---
token_rows = []
for idx, p in enumerate(positions):
    row = {
        "token_index_in_answer": idx,
        **per_token_traj_feats[idx],
        **{f"grad_{k}": v for k, v in per_token_grad_scores[idx].items()},
        **{f"ablate_{k}": v for k, v in per_token_ablation_scores[idx].items()},
    }
    token_rows.append(row)
assert len(token_rows) == 3
print("per-token rows OK, e.g. row[0]:", token_rows[0])

# --- aggregated example row (mirrors run_pipeline.py's aggregation) ---
agg_row = {
    "num_generated_tokens": len(positions),
    **mean_aggregate(per_token_traj_feats),
    **{f"grad_{k}": v for k, v in mean_aggregate(per_token_grad_scores).items()},
    **{f"ablate_{k}": v for k, v in mean_aggregate(per_token_ablation_scores).items()},
}
print("aggregated row OK:", agg_row)

# sanity: aggregated grad_norm_mean should sit between the min and max of the
# per-token values (basic mean-is-between-bounds check, catches transposition bugs)
per_token_vals = [f["grad_norm_mean"] for f in per_token_traj_feats]
assert min(per_token_vals) - 1e-6 <= agg_row["grad_norm_mean"] <= max(per_token_vals) + 1e-6

# --- CSV round-trip (mirrors write_csv) ---
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=sorted(token_rows[0].keys()))
writer.writeheader()
writer.writerows(token_rows)
buf.seek(0)
read_back = list(csv.DictReader(buf))
assert len(read_back) == 3

print("\nAll multi-token integration checks passed: positions, aggregation, "
      "and CSV round-trip all behave as expected.")
