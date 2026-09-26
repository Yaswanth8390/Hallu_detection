"""Export one human answer-label row per generated question/answer pair."""

import argparse
import csv


def export_annotation_template(data_path: str, output_path: str) -> int:
    answers = {}
    with open(data_path, newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            example_id = row["example_index"]
            answer = (row["question"], row["generated_text"])
            if example_id in answers and answers[example_id] != answer:
                raise ValueError(
                    f"Example {example_id!r} has inconsistent question/answer rows"
                )
            answers[example_id] = answer

    if not answers:
        raise ValueError(f"No feature rows found in {data_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        fields = [
            "example_index", "question", "generated_text",
            "human_label", "annotator_notes",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for example_id, (question, generated_text) in answers.items():
            writer.writerow({
                "example_index": example_id,
                "question": question,
                "generated_text": generated_text,
                "human_label": "",
                "annotator_notes": "",
            })

    return len(answers)


def main():
    parser = argparse.ArgumentParser(
        description="Create an answer-level human annotation template from token features."
    )
    parser.add_argument("--data", default="results_tokens.csv")
    parser.add_argument("--out", default="human_labels.csv")
    args = parser.parse_args()
    answer_count = export_annotation_template(args.data, args.out)
    print(f"Wrote {answer_count} answer annotation rows to {args.out}")
    print("Fill human_label with supported, hallucinated, or uncertain.")


if __name__ == "__main__":
    main()
