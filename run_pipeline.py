"""Generate answers and score input-direction alignment for content words."""

import argparse
import csv

from dataset import load_truthfulqa
from grounding import content_word_groups, grounding_strength
from jacobian import compute_directional_sensitivity
from model_utils import generate_answer, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of TruthfulQA examples")
    parser.add_argument("--out", type=str, default="results.csv",
                        help="token-level CSV output (one row per content word)")
    parser.add_argument("--per-token-out", type=str, default="results_tokens.csv",
                        help="compatibility output path; contains the same token-level rows")
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="maximum generated subword-token count")
    parser.add_argument("--grounding-threshold", type=float, default=0.1,
                        help="absolute cosine threshold for weak/strong grounding status")
    parser.add_argument("--device", type=str, default="cuda",
                        help='"cuda" for one GPU, "auto" to split across visible GPUs')
    parser.add_argument("--load-in-8bit", action="store_true", default=True,
                        help="quantize weights to 8-bit (default)")
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false",
                        help="load full bf16 weights")
    args = parser.parse_args()
    if not 0.0 <= args.grounding_threshold <= 1.0:
        parser.error("--grounding-threshold must be between 0 and 1")

    model, tokenizer = load_model(device=args.device, load_in_8bit=args.load_in_8bit)
    examples = load_truthfulqa(limit=args.n)
    token_rows = []

    for example_index, example in enumerate(examples, start=1):
        generation = generate_answer(
            model, tokenizer, example.question, device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        prompt_len = generation.prompt_ids.shape[0]
        word_groups = content_word_groups(
            tokenizer, generation.generated_ids, generation.generated_text, prompt_len
        )

        for token_index, group in enumerate(word_groups):
            score = compute_directional_sensitivity(
                model, generation.full_ids, prompt_len, group["positions"]
            )
            alignment = score["alignment_score"]
            token_rows.append({
                "question": example.question,
                "generated_text": generation.generated_text,
                "token_index_in_answer": token_index,
                "token": group["token"],
                "subtoken_count": group["subtoken_count"],
                "alignment_score": alignment,
                "grounding_strength": grounding_strength(
                    alignment, args.grounding_threshold
                ),
                "grounding_threshold": args.grounding_threshold,
                "grouped_target_logit": score["target_logit"],
            })

        print(f"[{example_index}/{len(examples)}] {example.question!r} -> "
              f"{generation.generated_text!r} ({len(word_groups)} content words scored)")

    def write_csv(path):
        if not token_rows:
            print(f"No content-token rows to write for {path}.")
            return
        fieldnames = list(token_rows[0])
        with open(path, "w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(token_rows)
        print(f"Wrote {len(token_rows)} token-level rows to {path}")

    write_csv(args.out)
    if args.per_token_out != args.out:
        write_csv(args.per_token_out)


if __name__ == "__main__":
    main()