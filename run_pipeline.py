"""
End-to-end run: for each TruthfulQA question ->
  1. generate an answer with Qwen2.5-7B-Instruct (greedy)
  2. compute layer-wise Jacobian trajectories for EVERY generated token
     (jacobian.py already accepts a list of positions -- this just uses that)
  3. compute gradient + ablation grounding scores against a candidate entity
     from the question, for every generated token
  4. label the generation as correct/hallucinated (heuristic, whole-answer level)
  5. write two CSVs:
       --out            one row per EXAMPLE (mean of each feature across its
                         generated tokens) -- this is what evaluate.py expects.
       --per-token-out   one row per (example, generated token) with the raw,
                         un-aggregated values -- for inspecting whether signal
                         is concentrated in specific tokens (e.g. the token
                         that names the hallucinated entity) rather than
                         smeared evenly across the answer.

Run:
    python run_pipeline.py --n 50 --out results.csv --per-token-out results_tokens.csv

This will NOT run on CPU in reasonable time for a 7B model -- use a GPU.
Multi-token tracing costs one forward+backward per generated token per score
type (Jacobian, gradient-grounding, ablation-grounding), so cost scales
linearly with answer length -- use --max-new-tokens to cap it if answers run
long.
"""

import argparse
import csv
from typing import Dict, List

from model_utils import load_model, generate_answer
from jacobian import compute_layerwise_jacobian, trajectory_features
from grounding import find_entity_span, gradient_grounding_score, ablation_grounding_score
from dataset import load_truthfulqa, label_correctness


def mean_aggregate(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each key across a list of per-token feature dicts. All dicts
    must share the same keys (true here since each token goes through the
    same feature functions).
    """
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of TruthfulQA examples")
    parser.add_argument("--out", type=str, default="results.csv",
                         help="per-example CSV (mean-aggregated across generated tokens)")
    parser.add_argument("--per-token-out", type=str, default="results_tokens.csv",
                         help="per-generated-token CSV (raw, un-aggregated)")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                         help="cap on generated answer length -- controls cost, since "
                              "every generated token now gets its own Jacobian+grounding pass")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model(device=args.device)
    examples = load_truthfulqa(limit=args.n)

    example_rows = []
    token_rows = []

    for i, ex in enumerate(examples):
        gen = generate_answer(model, tokenizer, ex.question, device=args.device,
                               max_new_tokens=args.max_new_tokens)
        full_ids = gen.full_ids
        prompt_len = gen.prompt_ids.shape[0]
        gen_len = gen.generated_ids.shape[0]
        if gen_len == 0:
            continue

        # Every generated position, not just the first -- this is the full
        # answer span the model produced.
        positions = list(range(prompt_len, full_ids.shape[0]))

        traces = compute_layerwise_jacobian(model, full_ids, positions=positions,
                                             device=args.device)
        per_token_traj_feats = [trajectory_features(t) for t in traces]

        span = None
        if ex.candidate_entity:
            span = find_entity_span(tokenizer, gen.prompt_ids, ex.candidate_entity)

        per_token_grad_scores = []
        per_token_ablation_scores = []
        if span is not None:
            for p in positions:
                per_token_grad_scores.append(
                    gradient_grounding_score(model, full_ids, p, span, device=args.device))
                per_token_ablation_scores.append(
                    ablation_grounding_score(model, tokenizer, full_ids, p, span,
                                              device=args.device))

        label = label_correctness(gen.generated_text, ex)

        # --- per-token detail rows ---
        for idx, p in enumerate(positions):
            token_id = traces[idx].token_id
            token_text = tokenizer.decode([token_id])
            row = {
                "question": ex.question,
                "generated_text": gen.generated_text,
                "candidate_entity": ex.candidate_entity,
                "token_index_in_answer": idx,
                "token_text": token_text,
                **per_token_traj_feats[idx],
            }
            if per_token_grad_scores:
                row.update({f"grad_{k}": v for k, v in per_token_grad_scores[idx].items()})
                row.update({f"ablate_{k}": v for k, v in per_token_ablation_scores[idx].items()})
            row.update(label)
            token_rows.append(row)

        # --- per-example aggregated row (mean across generated tokens) ---
        agg_row = {
            "question": ex.question,
            "generated_text": gen.generated_text,
            "candidate_entity": ex.candidate_entity,
            "num_generated_tokens": gen_len,
            **mean_aggregate(per_token_traj_feats),
        }
        if per_token_grad_scores:
            agg_row.update({f"grad_{k}": v for k, v in
                            mean_aggregate(per_token_grad_scores).items()})
            agg_row.update({f"ablate_{k}": v for k, v in
                            mean_aggregate(per_token_ablation_scores).items()})
        agg_row.update(label)
        example_rows.append(agg_row)

        print(f"[{i+1}/{len(examples)}] {ex.question!r} -> {gen.generated_text!r} "
              f"({gen_len} tokens, correct={label['is_correct_heuristic']})")

    def write_csv(path, rows):
        if not rows:
            print(f"No rows to write for {path}.")
            return
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")

    write_csv(args.out, example_rows)
    write_csv(args.per_token_out, token_rows)


if __name__ == "__main__":
    main()
