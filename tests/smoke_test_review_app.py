"""Smoke test CSV grouping and persistence used by the annotation UI."""

import csv
import tempfile
from pathlib import Path

from prepare_annotations import export_annotation_template, load_answer_groups
from review_app import annotation_csv, read_annotations, token_table_rows


with tempfile.TemporaryDirectory() as temporary_directory:
    directory = Path(temporary_directory)
    features_path = directory / "results_tokens.csv"
    annotations_path = directory / "human_labels.csv"
    feature_rows = [
        {
            "example_index": "1",
            "question": "Who wrote the book?",
            "generated_text": "A person wrote it.",
            "token": "person",
            "evidence_span": "Who wrote",
            "counterfactual_score": "0.2",
            "semantic_similarity": "0.8",
            "label": "0",
            "max_correct_overlap": "0.7",
            "max_incorrect_overlap": "0.1",
        },
        {
            "example_index": "1",
            "question": "Who wrote the book?",
            "generated_text": "A person wrote it.",
            "token": "wrote",
            "evidence_span": "wrote the book",
            "counterfactual_score": "0.3",
            "semantic_similarity": "0.9",
            "label": "0",
            "max_correct_overlap": "0.7",
            "max_incorrect_overlap": "0.1",
        },
        {
            "example_index": "2",
            "question": "What is the capital?",
            "generated_text": "It is Atlantis.",
            "token": "Atlantis",
            "evidence_span": "capital",
            "counterfactual_score": "-0.2",
            "semantic_similarity": "0.1",
            "label": "1",
            "max_correct_overlap": "0.0",
            "max_incorrect_overlap": "0.5",
        },
    ]
    with features_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    answers = load_answer_groups(str(features_path))
    assert len(answers) == 2
    assert answers[0]["machine_label"] == "0"
    assert len(answers[0]["token_rows"]) == 2
    table = token_table_rows(answers[1])
    assert table[0]["Existing auto label"] == "Hallucinated (heuristic)"

    assert export_annotation_template(str(features_path), str(annotations_path)) == 2
    annotations = read_annotations(str(annotations_path))
    annotations["1"] = {
        "human_label": "supported",
        "annotator_notes": "Matches the reference.",
    }
    annotations["2"] = {
        "human_label": "abstain",
        "annotator_notes": "",
    }
    annotations_path.write_text(
        annotation_csv(answers, annotations), encoding="utf-8"
    )
    loaded = read_annotations(str(annotations_path))
    assert loaded["1"]["human_label"] == "supported"
    assert loaded["2"]["human_label"] == "abstain"
    assert "Matches the reference." in annotations_path.read_text(encoding="utf-8")

print("Review UI answer grouping and annotation persistence smoke test passed.")
