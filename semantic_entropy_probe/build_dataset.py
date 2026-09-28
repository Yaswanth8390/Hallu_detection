"""Build semantic-entropy targets and single-response hidden-state features."""

import argparse
import csv
import json
import random

import numpy as np
import torch

from .entropy import (
    NLIEntailment,
    extract_response_representation,
    semantic_entropy,
)
from .runtime import (
    MODEL_NAME,
    encode_question,
    input_device,
    load_model,
    load_questions,
    sequence_log_probability,
)

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def sample_responses(model, tokenizer, prompt_ids: torch.Tensor,
                     sample_count: int, max_new_tokens: int,
                     temperature: float, top_p: float) -> list[dict]:
    samples = []
    input_ids = prompt_ids.to(input_device(model)).unsqueeze(0)
    for _ in range(sample_count):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        generated_ids = output[input_ids.shape[1]:].detach().cpu()
        text = tokenizer.decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text or generated_ids.numel() == 0:
            raise RuntimeError("Model generated an empty answer; cannot build SEP target")
        samples.append({
            "text": text,
            "generated_ids": generated_ids,
            "log_probability": sequence_log_probability(
                model, prompt_ids, generated_ids
            ),
        })
    return samples


def build_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model(
        device=args.device, load_in_8bit=args.load_in_8bit
    )
    nli = NLIEntailment(args.nli_model, device=args.nli_device,
                        batch_size=args.nli_batch_size)
    questions = load_questions(args.n, split=args.split)
    rows = []
    for example_index, question in enumerate(questions, start=1):
        prompt_ids = encode_question(tokenizer, question)
        samples = sample_responses(
            model, tokenizer, prompt_ids, args.num_samples,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        responses = [sample["text"] for sample in samples]
        entropy, cluster_count, cluster_ids = semantic_entropy(
            responses,
            [sample["log_probability"] for sample in samples],
            nli.matrix(responses, threshold=args.entailment_threshold),
        )
        representation_sample = samples[0]
        hidden = extract_response_representation(
            model, prompt_ids, representation_sample["generated_ids"],
            layer=args.layer,
        )
        rows.append({
            "example_index": example_index,
            "question": question,
            "generated_text": representation_sample["text"],
            "hidden_features": json.dumps(hidden.tolist()),
            "semantic_entropy": entropy,
            "cluster_count": cluster_count,
            "num_samples": len(samples),
            "sampled_responses": json.dumps(responses, ensure_ascii=False),
            "sample_cluster_ids": json.dumps(cluster_ids),
            "model_name": MODEL_NAME,
            "nli_model": args.nli_model,
            "hidden_layer": args.layer,
        })
        print(
            f"[{example_index}/{len(questions)}] H_SE={entropy:.4f} nats "
            f"clusters={cluster_count}/{len(samples)}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No SEP examples were produced")
    with open(args.out, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} SEP examples to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-device", default="cuda")
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--layer", type=int, default=-1,
                        help="hidden_states tuple index; -1 selects the final layer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="sep_dataset.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-8bit", action="store_true", default=True)
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false")
    args = parser.parse_args()
    if args.n < 1 or args.num_samples < 1 or args.max_new_tokens < 1:
        parser.error("--n, --num-samples, and --max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if not 0 <= args.entailment_threshold <= 1:
        parser.error("--entailment-threshold must be between 0 and 1")
    if args.nli_batch_size < 1:
        parser.error("--nli-batch-size must be positive")
    build_dataset(args)


if __name__ == "__main__":
    main()
