"""Generate a complete answer and flag its individual content tokens."""

import argparse
import json

import joblib
import numpy as np
import torch

from grounding import content_word_groups, load_content_tagger
from model_utils import MODEL_NAME, generate_answer, load_model
from token_features import extract_token_features
from train_detector import feature_vector, token_probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--detector", type=str, default="token_detector.joblib")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="token hallucination probability threshold")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--content-tagger", type=str, default="en_core_web_sm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-in-8bit", action="store_true", default=True)
    parser.add_argument("--full-precision", dest="load_in_8bit", action="store_false")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    detector = joblib.load(args.detector)
    if detector["model_name"] != MODEL_NAME:
        raise ValueError(
            f"Detector expects {detector['model_name']!r}, but this inference script "
            f"loads {MODEL_NAME!r}"
        )
    if detector.get("training_objective") != "answer_level_max_pool_binary_cross_entropy":
        raise ValueError(
            "Detector was not trained with the answer-level max-pooling objective; "
            "retrain it with the current train_detector.py before inference."
        )
    reasoning_basis = torch.as_tensor(detector["reasoning_basis"])
    if reasoning_basis.shape[1] != detector["harp_feature_size"]:
        raise ValueError("Detector HARP metadata does not match its saved projection basis")

    model, tokenizer = load_model(device=args.device, load_in_8bit=args.load_in_8bit)
    generation = generate_answer(
        model, tokenizer, args.question, device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    groups = content_word_groups(
        tokenizer, generation.generated_ids, generation.generated_text,
        generation.prompt_ids.numel(), load_content_tagger(args.content_tagger),
    )
    features = extract_token_features(
        model, tokenizer, generation, args.question, groups, reasoning_basis,
    )

    records = []
    feature_vectors = []
    for feature in features:
        row = {
            "counterfactual_score": feature["counterfactual_score"],
            "semantic_similarity": feature["semantic_similarity"],
            "HARP_features": json.dumps(feature["HARP_features"]),
        }
        feature_vectors.append(
            feature_vector(row, expected_harp_size=detector["harp_feature_size"])
        )

    probabilities = token_probabilities(detector, np.asarray(feature_vectors))
    for feature, probability in zip(features, probabilities):
        records.append({
            "token": feature["token"],
            "char_start": feature["char_start"],
            "char_end": feature["char_end"],
            "evidence_span": feature["evidence_span"],
            "counterfactual_score": feature["counterfactual_score"],
            "semantic_similarity": feature["semantic_similarity"],
            "hallucination_probability": float(probability),
            "flagged": probability >= args.threshold,
        })

    answer_probability = float(max(probabilities, default=0.0))
    print(json.dumps({
        "generated_text": generation.generated_text,
        "content_token_predictions": records,
        "answer_hallucination_probability": answer_probability,
        "answer_flagged": answer_probability >= args.threshold,
        "threshold": args.threshold,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
