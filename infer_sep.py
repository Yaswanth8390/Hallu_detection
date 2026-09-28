"""Predict semantic uncertainty from one generated answer using a trained SEP."""

import argparse
import json

import joblib

from model_utils import MODEL_NAME, generate_answer, load_model
from semantic_entropy import extract_response_representation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--probe", default="sep_probe.joblib")
    parser.add_argument("--threshold", type=float, default=None,
                        help="semantic-entropy alert threshold in nats")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-8bit", action="store_true", default=True)
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false")
    args = parser.parse_args()
    if args.threshold is not None and args.threshold < 0:
        parser.error("--threshold must be non-negative")

    detector = joblib.load(args.probe)
    if detector.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Probe expects {detector.get('model_name')!r}, but inference loads {MODEL_NAME!r}"
        )
    if detector.get("target") != "sampled_response_semantic_entropy_nats":
        raise ValueError("Probe does not contain a semantic-entropy training target")

    model, tokenizer = load_model(
        device=args.device, load_in_8bit=args.load_in_8bit
    )
    generation = generate_answer(
        model, tokenizer, args.question, device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    representation = extract_response_representation(
        model, generation.prompt_ids, generation.generated_ids,
        layer=detector["hidden_layer"],
    )
    if representation.shape != (detector["hidden_size"],):
        raise ValueError("Inference hidden-state dimension does not match the trained probe")
    predicted_entropy = max(0.0, float(
        detector["probe"].predict(
            detector["scaler"].transform(representation.reshape(1, -1))
        )[0]
    ))
    threshold = (
        detector["alert_threshold"] if args.threshold is None else args.threshold
    )
    print(json.dumps({
        "generated_text": generation.generated_text,
        "predicted_semantic_entropy_nats": predicted_entropy,
        "uncertainty_threshold_nats": threshold,
        "elevated_semantic_uncertainty": predicted_entropy >= threshold,
        "interpretation": (
            "This is a semantic-uncertainty signal, not a factuality judgment."
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
