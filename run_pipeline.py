"""Generate answers and score input-direction alignment for content words."""

import argparse
import csv

from dataset import label_correctness, load_truthfulqa
from grounding import (
    classify_confidence,
    classify_input_dependence,
    combine_input_dependence_and_confidence,
    content_word_groups,
    final_hallucination_label,
    select_semantic_evidence_spans,
)
from jacobian import compute_counterfactual_logprobs, compute_token_confidence
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
    parser.add_argument("--dependence-threshold", "--grounding-threshold",
                        dest="dependence_threshold", type=float, default=0.1,
                        help="log-probability-delta threshold for input-dependence labels")
    parser.add_argument("--entropy-threshold", type=float, default=1.0,
                        help="nats; original-prompt entropy at/below this counts as "
                             "'confident' when separating parametric knowledge from "
                             "possible hallucination for low-dependence words. Should "
                             "be calibrated experimentally, same as --dependence-threshold.")
    parser.add_argument("--overlap-threshold", type=float, default=0.3,
                        help="lexical-overlap threshold for TruthfulQA's own noisy "
                             "correctness heuristic (dataset.label_correctness), "
                             "attached to every row as ground truth for benchmarking "
                             "in evaluate.py. This is not human judgment -- see "
                             "dataset.py's module docstring.")
    parser.add_argument("--device", type=str, default="cuda",
                        help='"cuda" for one GPU, "auto" to split across visible GPUs')
    parser.add_argument("--load-in-8bit", action="store_true", default=True,
                        help="quantize weights to 8-bit (default)")
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false",
                        help="load unquantized FP16 weights")
    args = parser.parse_args()
    if args.dependence_threshold < 0.0:
        parser.error("--dependence-threshold must be nonnegative")
    if args.entropy_threshold < 0.0:
        parser.error("--entropy-threshold must be nonnegative")
    if not 0.0 <= args.overlap_threshold <= 1.0:
        parser.error("--overlap-threshold must be between 0 and 1")

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
        if generation.prompt.count(example.question) != 1:
            raise ValueError("Expected the question exactly once in the generation prompt")
        ground_truth = label_correctness(
            generation.generated_text, example, overlap_threshold=args.overlap_threshold
        )
        evidence_spans = select_semantic_evidence_spans(
            model, tokenizer, generation.prompt, generation.prompt_ids,
            generation.generated_ids, example.question, word_groups,
        )
        original_stats = compute_token_confidence(
            model, generation.prompt_ids, generation.generated_ids
        )
        original_scores = original_stats["logprob"]
        original_entropy = original_stats["entropy"]
        original_margin = original_stats["margin"]
        counterfactual_cache = {}

        for token_index, group in enumerate(word_groups):
            evidence = evidence_spans[token_index]
            token_indices = group["token_indices"]
            original_logprob = sum(original_scores[index].item() for index in token_indices)
            counterfactual_logprob = ""
            delta_logprob = ""
            if evidence is not None:
                span_key = (evidence["start"], evidence["end"])
                if span_key not in counterfactual_cache:
                    counterfactual_question = (
                        example.question[:evidence["start"]]
                        + example.question[evidence["end"]:]
                    )
                    counterfactual_prompt = generation.prompt.replace(
                        example.question, counterfactual_question, 1
                    )
                    counterfactual_prompt_ids = tokenizer(
                        counterfactual_prompt, return_tensors="pt"
                    ).input_ids[0]
                    if counterfactual_prompt_ids.tolist() == generation.prompt_ids.tolist():
                        raise RuntimeError(
                            f"Removing evidence span {evidence['text']!r} did not change "
                            "the prompt token IDs; refusing to report a no-op counterfactual."
                        )
                    counterfactual_cache[span_key] = compute_counterfactual_logprobs(
                        model, generation.prompt_ids, counterfactual_prompt_ids,
                        generation.generated_ids,
                        original_token_logprobs=original_scores,
                    )["counterfactual_token_logprobs"]
                counterfactual_scores = counterfactual_cache[span_key]
                counterfactual_logprob = sum(
                    counterfactual_scores[index].item() for index in token_indices
                )
                delta_logprob = original_logprob - counterfactual_logprob
                classification = classify_input_dependence(
                    delta_logprob, args.dependence_threshold
                )
                evidence_text = evidence["text"]
                counterfactual_change = f"removed:{evidence_text}"
            else:
                classification = "no_matching_evidence_span"
                evidence_text = ""
                counterfactual_change = "not_scored_no_span"

            word_entropy = sum(
                original_entropy[index].item() for index in token_indices
            ) / len(token_indices)
            word_margin = sum(
                original_margin[index].item() for index in token_indices
            ) / len(token_indices)
            confidence_classification = classify_confidence(
                word_entropy, args.entropy_threshold
            )
            combined_classification = combine_input_dependence_and_confidence(
                classification, confidence_classification
            )
            hallucination_label = final_hallucination_label(combined_classification)

            token_rows.append({
                "example_index": example_index,
                "question": example.question,
                "generated_text": generation.generated_text,
                "token_index_in_answer": token_index,
                "token": group["token"],
                "subtoken_count": group["subtoken_count"],
                "evidence_span": evidence_text,
                "evidence_similarity": evidence["similarity"] if evidence is not None else "",
                "counterfactual_change": counterfactual_change,
                "original_logprob": original_logprob,
                "counterfactual_logprob": counterfactual_logprob,
                "delta_logprob": delta_logprob,
                "classification": classification,
                "dependence_threshold": args.dependence_threshold,
                "word_entropy": word_entropy,
                "word_margin": word_margin,
                "entropy_threshold": args.entropy_threshold,
                "confidence_classification": confidence_classification,
                "combined_classification": combined_classification,
                "hallucination_label": hallucination_label,
                "is_correct_heuristic": ground_truth["is_correct_heuristic"],
                "max_correct_overlap": ground_truth["max_correct_overlap"],
                "max_incorrect_overlap": ground_truth["max_incorrect_overlap"],
                "overlap_threshold": args.overlap_threshold,
            })

            print(f"[{example_index}/{len(examples)}] {group['token']!r}: "
                  f"evidence={evidence_text!r} delta_logprob={delta_logprob} "
                  f"classification={classification} "
                  f"confidence={confidence_classification} (entropy={word_entropy:.3f}) "
                  f"combined={combined_classification} label={hallucination_label}", flush=True)

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