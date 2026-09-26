"""Generate a per-content-token semantic, counterfactual, and HARP dataset."""

import argparse
import csv
import json

import torch

from dataset import label_correctness, load_truthfulqa
from grounding import content_word_groups, load_content_tagger
from harp import build_reasoning_basis
from model_utils import MODEL_NAME, generate_answer, load_model
from token_features import extract_token_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of TruthfulQA examples")
    parser.add_argument("--out", type=str, default="results_tokens.csv",
                        help="per-content-token feature dataset")
    parser.add_argument("--per-token-out", type=str, default=None,
                        help="optional compatibility copy of the same token rows")
    parser.add_argument("--harp-basis-out", type=str, default="harp_basis.pt",
                        help="save the HARP projection basis for classifier training")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="maximum generated subword-token count")
    parser.add_argument("--overlap-threshold", type=float, default=0.3,
                        help="lexical-overlap threshold for TruthfulQA's noisy answer label")
    parser.add_argument("--content-tagger", type=str, default="en_core_web_sm",
                        help="spaCy POS/NER model used to select content tokens")
    parser.add_argument("--device", type=str, default="cuda",
                        help='"cuda" for one GPU, "auto" to split across visible GPUs')
    parser.add_argument("--load-in-8bit", action="store_true", default=True,
                        help="quantize weights to 8-bit (default)")
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false",
                        help="load unquantized FP16 weights")
    args = parser.parse_args()
    if not 0.0 <= args.overlap_threshold <= 1.0:
        parser.error("--overlap-threshold must be between 0 and 1")

    model, tokenizer = load_model(device=args.device, load_in_8bit=args.load_in_8bit)
    pos_tagger = load_content_tagger(args.content_tagger)
    reasoning_basis = build_reasoning_basis(model)
    torch.save({
        "basis": reasoning_basis.cpu(),
        "model_name": MODEL_NAME,
        "hidden_size": reasoning_basis.shape[0],
        "reasoning_size": reasoning_basis.shape[1],
    }, args.harp_basis_out)
    examples = load_truthfulqa(limit=args.n)
    token_rows = []

    for example_index, example in enumerate(examples, start=1):
        generation = generate_answer(
            model, tokenizer, example.question, device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        if generation.prompt.count(example.question) != 1:
            raise ValueError("Expected the question exactly once in the generation prompt")
        word_groups = content_word_groups(
            tokenizer, generation.generated_ids, generation.generated_text,
            generation.prompt_ids.numel(), pos_tagger,
        )
        ground_truth = label_correctness(
            generation.generated_text, example,
            overlap_threshold=args.overlap_threshold,
        )
        token_features = extract_token_features(
            model, tokenizer, generation, example.question, word_groups,
            reasoning_basis,
        )
        label = int(not ground_truth["is_correct_heuristic"])

        for token_index, features in enumerate(token_features):
            row = {
                "example_index": example_index,
                "question": example.question,
                "generated_text": generation.generated_text,
                "token_index_in_answer": token_index,
                "token": features["token"],
                "evidence_span": features["evidence_span"],
                "counterfactual_score": features["counterfactual_score"],
                "semantic_similarity": features["semantic_similarity"],
                "HARP_features": json.dumps(features["HARP_features"]),
                "label": label,
                "label_source": "truthfulqa_answer_level_lexical_overlap_weak_label",
                "is_correct_heuristic": ground_truth["is_correct_heuristic"],
                "max_correct_overlap": ground_truth["max_correct_overlap"],
                "max_incorrect_overlap": ground_truth["max_incorrect_overlap"],
                "overlap_threshold": args.overlap_threshold,
                "subtoken_count": features["subtoken_count"],
                "token_char_start": features["char_start"],
                "token_char_end": features["char_end"],
            }
            token_rows.append(row)
            print(
                f"[{example_index}/{len(examples)}] {features['token']!r}: "
                f"delta_logP={features['counterfactual_score']} "
                f"similarity={features['semantic_similarity']} label={label}",
                flush=True,
            )

    if not token_rows:
        raise RuntimeError("No content-token rows were generated; dataset was not written")

    def write_csv(path):
        with open(path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(token_rows[0]))
            writer.writeheader()
            writer.writerows(token_rows)
        print(f"Wrote {len(token_rows)} token rows to {path}")

    write_csv(args.out)
    if args.per_token_out and args.per_token_out != args.out:
        write_csv(args.per_token_out)


if __name__ == "__main__":
    main()
